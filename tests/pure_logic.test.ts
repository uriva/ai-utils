import { assert, assertEquals } from "@std/assert";
import { z } from "zod/v4";
import { tool } from "../mod.ts";
import {
  createSkillTools,
  guardNovelOpaqueIdentifiers,
  type HistoryEvent,
  injectAccessHistory,
  injectOutputEvent,
  learnSkillToolName,
  maxNovelOpaqueIdentifierCorrections,
  novelOpaqueIdentifierThought,
  ownThoughtTurn,
  ownUtteranceTurn,
  participantUtteranceTurn,
  runAbstractAgent,
  runCommandToolName,
} from "../src/agent.ts";
import {
  filterOrphanedToolResults,
  stripEmbeddedThoughtPatterns,
} from "../src/geminiAgent.ts";

Deno.test("filterOrphanedToolResults logic", () => {
  const baseAuth = { isOwn: true, id: "msg-id", timestamp: 100 } as const;
  const mkCall = (
    id: string,
    name: string,
    timestamp: number,
  ) => ({
    ...baseAuth,
    type: "tool_call" as const,
    id,
    name,
    timestamp,
    parameters: {},
    modelMetadata: { type: "gemini", responseId: "r1", thoughtSignature: "" },
  });
  const mkResult = (
    name: string,
    timestamp: number,
    toolCallId?: string,
  ) => ({
    ...baseAuth,
    type: "tool_result" as const,
    name,
    timestamp,
    toolCallId,
    result: "res",
    modelMetadata: { type: "gemini", responseId: "r1", thoughtSignature: "" },
  });

  // Case 1: Legacy match
  const h1 = [
    mkCall("c1", "toolA", 100),
    mkResult("toolA", 101),
  ];
  // @ts-ignore - types match roughly
  assertEquals(filterOrphanedToolResults(h1).length, 2);

  // Case 2: Orphaned legacy
  const h2 = [
    mkResult("toolA", 101),
  ];
  // @ts-ignore - types match roughly
  assertEquals(filterOrphanedToolResults(h2).length, 0);

  // Case 3: Mixed strict and legacy
  const h3 = [
    mkCall("c1", "toolA", 100),
    mkCall("c2", "toolA", 102),
    mkResult("toolA", 103, "c2"), // Claims c2
    mkResult("toolA", 104), // Should claim c1
  ];
  // @ts-ignore - types match roughly
  assertEquals(filterOrphanedToolResults(h3).length, 4);

  // Case 4: Stealing prevention
  const h4 = [
    mkCall("c1", "toolA", 100),
    mkResult("toolA", 103, "c1"), // Claims c1
    mkResult("toolA", 104), // Orphan, because c1 is taken
  ];
  // @ts-ignore - types match roughly
  assertEquals(filterOrphanedToolResults(h4).length, 2);

  // Case 5: Excess legacy results (1 call, 2 results)
  const h5 = [
    mkCall("c1", "toolA", 100),
    mkResult("toolA", 101),
    mkResult("toolA", 102),
  ];
  // @ts-ignore - types match roughly
  assertEquals(filterOrphanedToolResults(h5).length, 2);
});

Deno.test(
  "skills: run_command actually executes the skill tool handler",
  async () => {
    let handlerWasCalled = false;

    const testTool = tool({
      name: "test_tool",
      description: "Test tool",
      parameters: z.object({ value: z.number() }),
      handler: ({ value }) => {
        handlerWasCalled = true;
        return Promise.resolve(`Result: ${value * 2}`);
      },
    });

    const skillTools = createSkillTools([{
      name: "test",
      description: "Test skill",
      instructions: "Test instructions",
      tools: [testTool],
    }]);
    const runCommandTool = skillTools.find((t) =>
      t.name === runCommandToolName
    );

    if (!runCommandTool) {
      throw new Error("run_command tool not found");
    }

    const result = await runCommandTool.handler({
      command: "test/test_tool",
      params: { value: 5 },
    });

    assert(handlerWasCalled, "Handler should have been called");
    assertEquals(result, "Result: 10");
  },
);

Deno.test(
  `skills: ${learnSkillToolName} returns actual skill details`,
  async () => {
    const weatherSkill = {
      name: "weather",
      description: "Weather information service",
      instructions: "Always ask for location before checking weather",
      tools: [
        {
          name: "get_forecast",
          description: "Get weather forecast",
          parameters: z.object({ location: z.string() }),
          handler: () => Promise.resolve("Sunny"),
        },
      ],
    };

    const skillTools = createSkillTools([weatherSkill]);
    const learnSkillTool = skillTools.find((t) =>
      t.name === learnSkillToolName
    );

    if (!learnSkillTool) {
      throw new Error(`${learnSkillToolName} tool not found`);
    }

    const result = await learnSkillTool.handler({ skillName: "weather" });

    assert(typeof result === "string", "Result should be a string");
    const parsed = JSON.parse(result);
    assertEquals(parsed.name, "weather");
    assertEquals(parsed.description, "Weather information service");
    assertEquals(
      parsed.instructions,
      "Always ask for location before checking weather",
    );
    assert(Array.isArray(parsed.tools), "Should have tools array");
    assertEquals(parsed.tools.length, 1);
    assertEquals(parsed.tools[0].name, "get_forecast");
  },
);

Deno.test(
  "direct skillName/toolName call is routed through run_command",
  async () => {
    let handlerCalledWith = "";
    const history: HistoryEvent[] = [];

    const testSkillTool = tool({
      name: "get_info",
      description: "Get info",
      parameters: z.object({ query: z.string() }),
      handler: ({ query }) => {
        handlerCalledWith = query;
        return Promise.resolve(`info for ${query}`);
      },
    });

    const mockCallModel = (_h: HistoryEvent[]): Promise<HistoryEvent[]> =>
      Promise.resolve([{
        type: "tool_call" as const,
        id: crypto.randomUUID(),
        timestamp: Date.now(),
        isOwn: true,
        name: "my_skill/get_info",
        parameters: { query: "test" },
        modelMetadata: undefined,
      }]);

    await injectAccessHistory(() => Promise.resolve(history))(
      injectOutputEvent((event) => {
        history.push(event);
        return Promise.resolve();
      })(runAbstractAgent),
    )(
      {
        maxIterations: 2,
        onMaxIterationsReached: () => {},
        tools: [],
        skills: [{
          name: "my_skill",
          description: "A skill",
          instructions: "Use it",
          tools: [testSkillTool],
        }],
        prompt: "test",
        rewriteHistory: async () => {},
        timezoneIANA: "UTC",
      },
      mockCallModel,
    );

    assertEquals(handlerCalledWith, "test");
    const toolResult = history.find((e) =>
      e.type === "tool_result" && e.result === "info for test"
    );
    assert(toolResult, "Should have tool result with correct output");
  },
);

Deno.test("novel opaque id guard logs but passes through in shadow mode", () => {
  const prompt = "No URL is available until the async notification arrives.";
  const offendingOutput = [
    ownUtteranceTurn(
      '<video controls><source src="https://api.example-fake.com/s/e53b21" type="video/mp4" /></video>',
    ),
  ];
  const baseHistory = [
    participantUtteranceTurn({ name: "user", text: "wait" }),
  ];

  // Shadow mode: novel IDs are logged but output passes through unchanged
  const firstResult = guardNovelOpaqueIdentifiers(
    prompt,
    baseHistory,
    offendingOutput,
  );
  assertEquals(firstResult.emit, offendingOutput);
  assertEquals(firstResult.internal, offendingOutput);

  // Even after many offenses, still passes through (no do_nothing)
  const historyWithMaxCorrections = [
    ...baseHistory,
    ...Array.from(
      { length: maxNovelOpaqueIdentifierCorrections },
      () => ownThoughtTurn(novelOpaqueIdentifierThought),
    ),
  ];
  const exhaustedResult = guardNovelOpaqueIdentifiers(
    prompt,
    historyWithMaxCorrections,
    offendingOutput,
  );
  assertEquals(exhaustedResult.emit, offendingOutput);
  assertEquals(exhaustedResult.internal, offendingOutput);

  // Non-novel output: passes through unchanged
  const legitimateOutput = [
    ownUtteranceTurn("I'll wait for the download to complete."),
  ];
  const passthroughResult = guardNovelOpaqueIdentifiers(
    prompt,
    baseHistory,
    legitimateOutput,
  );
  assertEquals(passthroughResult.emit, legitimateOutput);
  assertEquals(passthroughResult.internal, legitimateOutput);
});

Deno.test("novel opaque id guard shadow mode ignores correction count", () => {
  const prompt = "No URL is available until the async notification arrives.";
  const offendingOutput = [
    ownUtteranceTurn(
      '<video controls><source src="https://api.example-fake.com/s/e53b21" type="video/mp4" /></video>',
    ),
  ];
  // History with old corrections interrupted by a non-correction event
  const historyWithResetCount = [
    participantUtteranceTurn({ name: "user", text: "wait" }),
    ...Array.from(
      { length: maxNovelOpaqueIdentifierCorrections - 1 },
      () => ownThoughtTurn(novelOpaqueIdentifierThought),
    ),
    participantUtteranceTurn({ name: "user", text: "try again" }),
    ...Array.from(
      { length: maxNovelOpaqueIdentifierCorrections - 1 },
      () => ownThoughtTurn(novelOpaqueIdentifierThought),
    ),
  ];
  // Shadow mode: always passes through regardless of correction count
  const result = guardNovelOpaqueIdentifiers(
    prompt,
    historyWithResetCount,
    offendingOutput,
  );
  assertEquals(result.emit, offendingOutput);
  assertEquals(result.internal, offendingOutput);
});

Deno.test("stripEmbeddedThoughtPatterns removes thoughts from mixed text", () => {
  const mixed =
    'Great choice! Here are the scenes:\n<video controls><source src="https://fake-url.com/video" type="video/mp4" /></video>[Internal thought, visible only to you: DOWNLOAD COMPLETE. Confirmed media HTML:\n<video controls><source src="https://api.find-scene.com/s/7c2a10" type="video/mp4" /></video>] [Internal thought, visible only to you: DOWNLOAD COMPLETE. Confirmed media HTML:\n<video controls><source src="https://api.find-scene.com/s/9d4f32" type="video/mp4" /></video>]';
  assertEquals(
    stripEmbeddedThoughtPatterns(mixed),
    'Great choice! Here are the scenes:\n<video controls><source src="https://fake-url.com/video" type="video/mp4" /></video>',
  );
});

Deno.test("stripEmbeddedThoughtPatterns preserves text without thoughts", () => {
  const plain = "Hello! Here is a regular message with no special patterns.";
  assertEquals(stripEmbeddedThoughtPatterns(plain), plain);
});

Deno.test("stripEmbeddedThoughtPatterns returns empty for thought-only text", () => {
  const thoughtOnly =
    "[Internal thought, visible only to you: some thought content]";
  assertEquals(stripEmbeddedThoughtPatterns(thoughtOnly), "");
});
