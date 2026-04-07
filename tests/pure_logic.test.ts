import { assert, assertEquals } from "@std/assert";
import type { Content } from "@google/genai";
import { z } from "zod/v4";
import { tool } from "../mod.ts";
import {
  createSkillTools,
  getToolDetailsToolName,
  type HistoryEvent,
  injectAccessHistory,
  injectOutputEvent,
  learnSkillToolName,
  runAbstractAgent,
  runCommandToolName,
} from "../src/agent.ts";
import {
  buildReq,
  filterOrphanedToolResults,
  stripEmbeddedThoughtPatterns,
} from "../src/geminiAgent.ts";
import { zodToTypingString } from "../src/gemini.ts";

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
    timestamp: number,
    toolCallId?: string,
  ) => ({
    ...baseAuth,
    type: "tool_result" as const,
    timestamp,
    toolCallId,
    result: "res",
    modelMetadata: { type: "gemini", responseId: "r1", thoughtSignature: "" },
  });

  // Case 1: Result without toolCallId is filtered out
  const h1 = [
    mkCall("c1", "toolA", 100),
    mkResult(101),
  ];
  // @ts-ignore - types match roughly
  assertEquals(filterOrphanedToolResults(h1).length, 1);

  // Case 2: Orphaned (no call at all, no toolCallId)
  const h2 = [
    mkResult(101),
  ];
  // @ts-ignore - types match roughly
  assertEquals(filterOrphanedToolResults(h2).length, 0);

  // Case 3: Strict match by toolCallId
  const h3 = [
    mkCall("c1", "toolA", 100),
    mkCall("c2", "toolA", 102),
    mkResult(103, "c2"),
    mkResult(104, "c1"),
  ];
  // @ts-ignore - types match roughly
  assertEquals(filterOrphanedToolResults(h3).length, 4);

  // Case 4: Strict match with duplicate prevention
  const h4 = [
    mkCall("c1", "toolA", 100),
    mkResult(103, "c1"),
    mkResult(104), // No toolCallId, filtered out
  ];
  // @ts-ignore - types match roughly
  assertEquals(filterOrphanedToolResults(h4).length, 2);

  // Case 5: Multiple results without toolCallId all filtered
  const h5 = [
    mkCall("c1", "toolA", 100),
    mkResult(101),
    mkResult(102),
  ];
  // @ts-ignore - types match roughly
  assertEquals(filterOrphanedToolResults(h5).length, 1);
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
    }, "test-call-id");

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

    const result = await learnSkillTool.handler(
      { skillName: "weather" },
      "test-call-id",
    );

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

Deno.test("tool_call with empty thoughtSignature omits field from API request", () => {
  const events = [
    {
      type: "participant_utterance" as const,
      isOwn: false as const,
      id: "msg-1",
      timestamp: 100,
      name: "user",
      text: "do something",
    },
    {
      type: "tool_call" as const,
      isOwn: true as const,
      id: "tc-1",
      timestamp: 101,
      name: "my_tool",
      parameters: { arg: "val" },
      modelMetadata: {
        type: "gemini" as const,
        responseId: "r1",
        thoughtSignature: "",
      },
    },
    {
      type: "tool_result" as const,
      isOwn: true as const,
      id: "tr-1",
      timestamp: 102,
      toolCallId: "tc-1",
      result: "done",
    },
  ];
  const req = buildReq(false, true, "prompt", [], "UTC", undefined)(events);
  const contents = req.contents as Content[];
  const modelContents = contents.filter((c: Content) => c.role === "model");
  for (const content of modelContents) {
    for (const part of content.parts ?? []) {
      if (part.functionCall) {
        assert(
          !("thoughtSignature" in part),
          `Expected functionCall part to NOT have thoughtSignature field when it's empty, but found: ${
            JSON.stringify(part)
          }`,
        );
      }
    }
  }
});

Deno.test("tool_call with non-empty thoughtSignature preserves it in API request", () => {
  const events = [
    {
      type: "participant_utterance" as const,
      isOwn: false as const,
      id: "msg-1",
      timestamp: 100,
      name: "user",
      text: "do something",
    },
    {
      type: "tool_call" as const,
      isOwn: true as const,
      id: "tc-1",
      timestamp: 101,
      name: "my_tool",
      parameters: { arg: "val" },
      modelMetadata: {
        type: "gemini" as const,
        responseId: "r1",
        thoughtSignature: "abc123signature",
      },
    },
    {
      type: "tool_result" as const,
      isOwn: true as const,
      id: "tr-1",
      timestamp: 102,
      toolCallId: "tc-1",
      result: "done",
    },
  ];
  const req = buildReq(false, true, "prompt", [], "UTC", undefined)(events);
  const contents = req.contents as Content[];
  const modelContents = contents.filter((c: Content) => c.role === "model");
  let foundFc = false;
  for (const content of modelContents) {
    for (const part of content.parts ?? []) {
      if (part.functionCall) {
        foundFc = true;
        assertEquals(
          part.thoughtSignature,
          "abc123signature",
          "Expected thoughtSignature to be preserved when non-empty",
        );
      }
    }
  }
  assert(foundFc, "Expected to find a functionCall part");
});

Deno.test(
  "learn_skill returns lightweight payload without parameter schemas",
  async () => {
    const skillTools = createSkillTools([{
      name: "weather",
      description: "Weather information service",
      instructions: "Always ask for location before checking weather",
      tools: [
        {
          name: "get_forecast",
          description: "Get weather forecast",
          parameters: z.object({ location: z.string(), days: z.number() }),
          handler: () => Promise.resolve("Sunny"),
        },
        {
          name: "get_temperature",
          description: "Get current temperature",
          parameters: z.object({ location: z.string() }),
          handler: () => Promise.resolve("25°C"),
        },
      ],
    }]);
    const learnSkillTool = skillTools.find((t) =>
      t.name === learnSkillToolName
    )!;

    const result = await learnSkillTool.handler(
      { skillName: "weather" },
      "test-call-id",
    );

    const parsed = JSON.parse(result as string);
    assertEquals(parsed.name, "weather");
    assertEquals(
      parsed.instructions,
      "Always ask for location before checking weather",
    );
    assertEquals(parsed.tools.length, 2);
    assertEquals(parsed.tools[0].name, "get_forecast");
    assertEquals(parsed.tools[0].description, "Get weather forecast");
    assert(
      !("parameters" in parsed.tools[0]),
      "learn_skill should NOT include parameters",
    );
    assert(
      !("parameters" in parsed.tools[1]),
      "learn_skill should NOT include parameters",
    );
  },
);

Deno.test(
  "get_tool_details returns concise typing string for a specific tool",
  async () => {
    const skillTools = createSkillTools([{
      name: "weather",
      description: "Weather information service",
      instructions: "Always ask for location before checking weather",
      tools: [
        {
          name: "get_forecast",
          description: "Get weather forecast",
          parameters: z.object({ location: z.string(), days: z.number() }),
          handler: () => Promise.resolve("Sunny"),
        },
      ],
    }]);
    const getDetailsTool = skillTools.find((t) =>
      t.name === getToolDetailsToolName
    )!;

    assert(getDetailsTool, "get_tool_details tool should exist");

    const result = await getDetailsTool.handler(
      { toolPath: "weather/get_forecast" },
      "test-call-id",
    ) as string;

    assert(result.includes("get_forecast"), "Should include tool name");
    assert(
      result.includes("Get weather forecast"),
      "Should include description",
    );
    assert(
      result.includes("location: string"),
      "Should include location param",
    );
    assert(result.includes("days: number"), "Should include days param");
    assert(!result.includes('"type"'), "Should NOT be JSON Schema format");
  },
);

Deno.test(
  "get_tool_details returns error for nonexistent tool",
  async () => {
    const skillTools = createSkillTools([{
      name: "weather",
      description: "Weather information service",
      instructions: "test",
      tools: [{
        name: "get_forecast",
        description: "Get weather forecast",
        parameters: z.object({ location: z.string() }),
        handler: () => Promise.resolve("Sunny"),
      }],
    }]);
    const getDetailsTool = skillTools.find((t) =>
      t.name === getToolDetailsToolName
    )!;

    const result = await getDetailsTool.handler(
      { toolPath: "weather/nonexistent" },
      "test-call-id",
    );

    assert(
      (result as string).includes("not found"),
      "Should return not found message",
    );
  },
);

Deno.test("zodToTypingString renders flat object with descriptions", () => {
  const schema = z.object({
    location: z.string().describe("City name"),
    days: z.number(),
  });
  assertEquals(
    zodToTypingString(schema),
    "{ location: string /* City name */, days: number }",
  );
});

Deno.test("zodToTypingString renders optional, enum, array, boolean", () => {
  const schema = z.object({
    units: z.enum(["celsius", "fahrenheit"]).optional(),
    tags: z.array(z.string()),
    active: z.boolean(),
  });
  assertEquals(
    zodToTypingString(schema),
    '{ units?: "celsius" | "fahrenheit", tags: string[], active: boolean }',
  );
});

Deno.test("zodToTypingString renders nullable fields", () => {
  const schema = z.object({
    name: z.string().nullable(),
    ids: z.array(z.string()).nullable().optional(),
  });
  assertEquals(
    zodToTypingString(schema),
    "{ name: string | null, ids?: string[] | null }",
  );
});
