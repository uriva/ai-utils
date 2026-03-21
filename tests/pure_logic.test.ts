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
import { filterOrphanedToolResults } from "../src/geminiAgent.ts";

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

Deno.test("novel opaque id guard corrects then emits do_nothing", () => {
  const prompt = "No URL is available until the async notification arrives.";
  const offendingOutput = [
    ownUtteranceTurn(
      '<video controls><source src="https://api.example-fake.com/s/e53b21" type="video/mp4" /></video>',
    ),
  ];
  const baseHistory = [
    participantUtteranceTurn({ name: "user", text: "wait" }),
  ];

  // First offense: returns correction thought internally
  const firstResult = guardNovelOpaqueIdentifiers(
    prompt,
    baseHistory,
    offendingOutput,
  );
  assertEquals(firstResult.internal.length, 1);
  assertEquals(firstResult.emit.length, 0);
  assertEquals(firstResult.internal[0].type, "own_thought");
  assert(
    "text" in firstResult.internal[0] && firstResult.internal[0].text ===
        novelOpaqueIdentifierThought,
  );

  // After max corrections: returns do_nothing emitted
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
  assertEquals(exhaustedResult.emit.length, 1);
  assertEquals(exhaustedResult.internal.length, 0);
  assertEquals(exhaustedResult.emit[0].type, "do_nothing");

  // Non-novel output: passes through unchanged emitted
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

Deno.test("novel opaque id correction count resets after non-correction event", () => {
  const prompt = "No URL is available until the async notification arrives.";
  const offendingOutput = [
    ownUtteranceTurn(
      '<video controls><source src="https://api.example-fake.com/s/e53b21" type="video/mp4" /></video>',
    ),
  ];
  // History with old corrections interrupted by a non-correction event, then fewer trailing corrections
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
  // Should still correct (not do_nothing) because only trailing consecutive count matters
  const result = guardNovelOpaqueIdentifiers(
    prompt,
    historyWithResetCount,
    offendingOutput,
  );
  assertEquals(result.internal.length, 1);
  assertEquals(result.emit.length, 0);
  assertEquals(result.internal[0].type, "own_thought");
  assert(
    "text" in result.internal[0] &&
      result.internal[0].text === novelOpaqueIdentifierThought,
  );
});
