import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import type { Content } from "@google/genai";
import { z } from "zod/v4";
import { tool } from "../mod.ts";
import {
  createSkillTools,
  estimateAgentInputTokens,
  formatSkillsPrompt,
  type HistoryEvent,
  injectAccessHistory,
  injectOutputEvent,
  learnSkillToolName,
  maxUtteranceChars,
  ownUtteranceTurn,
  resolveToolDescription,
  runAbstractAgent,
  runCommandToolName,
  sanitizeModelOutput,
  type Skill,
} from "../src/agent.ts";
import {
  buildReq,
  filterAndRewriteInvalidToolCalls,
  filterOrphanedToolResults,
  geminiMalformedFunctionCallError,
  geminiOutputToHistoryEvents,
  rejectMalformedFunctionCall,
  stripEmbeddedThoughtPatterns,
} from "../src/geminiAgent.ts";
import {
  geminiFallbackVersion,
  geminiFlashVersion,
  geminiProVersion,
} from "../src/gemini.ts";
import {
  collapseDuplicatedText,
  isEmojiFlood,
  isRepetitionFlood,
  isRetryableError,
  isSyntheticTimeoutError,
  syntheticTimeoutMarker,
} from "../src/utils.ts";

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
  "skills: run_command handles tool names that include the skill prefix",
  async () => {
    let calledWith = "";
    const prefixedTool = tool({
      name: "browser/create",
      description: "Create browser",
      parameters: z.object({}),
      handler: () => {
        calledWith = "create";
        return Promise.resolve("session_id: abc");
      },
    });
    const skillTools = createSkillTools([{
      name: "browser",
      description: "Browser",
      instructions: "Use it",
      tools: [prefixedTool],
    }]);
    const runCommandTool = skillTools.find((t) =>
      t.name === runCommandToolName
    );
    if (!runCommandTool) throw new Error("run_command not found");
    const result = await runCommandTool.handler(
      { command: "browser/create", params: {} },
      "test-call-id",
    );
    assertEquals(calledWith, "create");
    assertEquals(result, "session_id: abc");
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

    const mockCallModel = (h: HistoryEvent[]): Promise<HistoryEvent[]> => {
      const hasCall = h.some((e) => e.type === "tool_call");
      if (hasCall) return Promise.resolve([ownUtteranceTurn("Done")]);
      return Promise.resolve([{
        type: "tool_call" as const,
        id: crypto.randomUUID(),
        timestamp: Date.now(),
        isOwn: true,
        name: "my_skill/get_info",
        parameters: { query: "test" },
        modelMetadata: undefined,
      }]);
    };

    await injectAccessHistory(() => Promise.resolve(history))(
      injectOutputEvent((event) => {
        history.push(event);
        return Promise.resolve();
      })(runAbstractAgent),
    )(
      {
        maxIterations: 2,
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

Deno.test(
  "direct skillName:toolName call (colon form) is routed through run_command",
  async () => {
    let handlerCalledWith = "";
    const history: HistoryEvent[] = [];

    const testSkillTool = tool({
      name: "read_url",
      description: "Read a URL",
      parameters: z.object({ url: z.string() }),
      handler: ({ url }) => {
        handlerCalledWith = url;
        return Promise.resolve(`content of ${url}`);
      },
    });

    const mockCallModel = (h: HistoryEvent[]): Promise<HistoryEvent[]> => {
      const hasCall = h.some((e) => e.type === "tool_call");
      if (hasCall) return Promise.resolve([ownUtteranceTurn("Done")]);
      return Promise.resolve([{
        type: "tool_call" as const,
        id: crypto.randomUUID(),
        timestamp: Date.now(),
        isOwn: true,
        name: "web:read_url",
        parameters: { url: "https://example.com" },
        modelMetadata: undefined,
      }]);
    };

    await injectAccessHistory(() => Promise.resolve(history))(
      injectOutputEvent((event) => {
        history.push(event);
        return Promise.resolve();
      })(runAbstractAgent),
    )(
      {
        maxIterations: 2,
        tools: [],
        skills: [{
          name: "web",
          description: "Web skill",
          instructions: "Use it",
          tools: [testSkillTool],
        }],
        prompt: "test",
        rewriteHistory: async () => {},
        timezoneIANA: "UTC",
      },
      mockCallModel,
    );

    assertEquals(handlerCalledWith, "https://example.com");
    const toolResult = history.find((e) =>
      e.type === "tool_result" && e.result === "content of https://example.com"
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

Deno.test("invalid Gemini tool calls rewrite to useful thoughts without internals", () => {
  const replacements: Record<string, HistoryEvent> = {};
  const filter = filterAndRewriteInvalidToolCalls((r) => {
    Object.assign(replacements, r);
    return Promise.resolve();
  });
  const history: Parameters<typeof filter>[0] = [
    {
      type: "tool_call",
      isOwn: true,
      id: "call-id",
      timestamp: 1,
      name: "run_command",
      parameters: { command: "x", params: { query: "abc" } },
      modelMetadata: {
        type: "gemini",
        responseId: "resp",
        thoughtSignature: "",
      },
    },
    {
      type: "tool_result",
      isOwn: true,
      id: "result-id",
      timestamp: 2,
      toolCallId: "call-id",
      result: "useful result",
    },
  ];

  const filtered = filter(history);

  assertEquals(filtered.length, 0);
  assertEquals(replacements["call-id"].type, "own_thought");
  assertEquals(replacements["result-id"].type, "own_thought");
  const replacementText = JSON.stringify(replacements);
  assert(replacementText.includes("I previously called the tool run_command"));
  assert(replacementText.includes("useful result"));
  assertEquals(replacementText.includes("thought signature"), false);
  assertEquals(replacementText.includes("Removed tool"), false);
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
  const req = buildReq(true, "prompt", [], "UTC", undefined)(events);
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
  const req = buildReq(true, "prompt", [], "UTC", undefined)(events);
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

Deno.test("isEmojiFlood returns false for normal text", () => {
  assertEquals(isEmojiFlood("Hello world! This is normal text."), false);
  assertEquals(isEmojiFlood("Some emojis are fine 😀🎉👍"), false);
});

Deno.test("isEmojiFlood returns true for emoji flood", () => {
  const flood = "😀".repeat(101);
  assertEquals(isEmojiFlood(flood), true);
});

Deno.test("isEmojiFlood returns false at exactly 100 emojis", () => {
  assertEquals(isEmojiFlood("😀".repeat(100)), false);
});

Deno.test("isEmojiFlood returns false for long text with scattered emojis", () => {
  const normalTextWithManyEmojis = "party ".repeat(500) + "🎉".repeat(120);
  assertEquals(isEmojiFlood(normalTextWithManyEmojis), false);
});

Deno.test("isEmojiFlood returns true for high density of emojis exceeding limit", () => {
  const highDensityFlood = "party " + "🎉".repeat(120);
  assertEquals(isEmojiFlood(highDensityFlood), true);
});

Deno.test(
  "emoji flood in model response causes retry and eventual throw",
  async () => {
    const history: HistoryEvent[] = [];
    const emojiFlood = "🤖".repeat(200);
    let callCount = 0;

    const mockCallModel = (_h: HistoryEvent[]): Promise<HistoryEvent[]> => {
      callCount++;
      return Promise.resolve([{
        type: "own_utterance" as const,
        id: crypto.randomUUID(),
        timestamp: Date.now(),
        isOwn: true as const,
        text: emojiFlood,
      }]);
    };

    await assertRejects(
      () =>
        injectAccessHistory(() => Promise.resolve(history))(
          injectOutputEvent((event) => {
            history.push(event);
            return Promise.resolve();
          })(runAbstractAgent),
        )(
          {
            maxIterations: 10,
            tools: [],
            prompt: "test",
            rewriteHistory: async () => {},
            timezoneIANA: "UTC",
          },
          mockCallModel,
        ),
      Error,
      "model keeps producing emoji flood responses",
    );

    assertEquals(callCount, 3, "should have retried exactly 3 times");
    assertEquals(history.length, 0, "no events should have been emitted");
  },
);

Deno.test(
  "emoji flood recovery: model succeeds after initial flood",
  async () => {
    const history: HistoryEvent[] = [];
    let callCount = 0;

    const mockCallModel = (_h: HistoryEvent[]): Promise<HistoryEvent[]> => {
      callCount++;
      if (callCount <= 2) {
        return Promise.resolve([{
          type: "own_utterance" as const,
          id: crypto.randomUUID(),
          timestamp: Date.now(),
          isOwn: true as const,
          text: "🤖".repeat(200),
        }]);
      }
      return Promise.resolve([{
        type: "own_utterance" as const,
        id: crypto.randomUUID(),
        timestamp: Date.now(),
        isOwn: true as const,
        text: "A normal response",
      }]);
    };

    await injectAccessHistory(() => Promise.resolve(history))(
      injectOutputEvent((event) => {
        history.push(event);
        return Promise.resolve();
      })(runAbstractAgent),
    )(
      {
        maxIterations: 10,
        tools: [],
        prompt: "test",
        rewriteHistory: async () => {},
        timezoneIANA: "UTC",
      },
      mockCallModel,
    );

    assertEquals(callCount, 3, "should have called model 3 times");
    assertEquals(history.length, 1, "should have emitted the normal response");
    const emitted = history[0];
    assertEquals(emitted.type, "own_utterance");
    if (emitted.type === "own_utterance") {
      assertEquals(emitted.text, "A normal response");
    }
  },
);

Deno.test("isRepetitionFlood returns false for normal text", () => {
  assertEquals(isRepetitionFlood("Hello world! This is normal text."), false);
  assertEquals(isRepetitionFlood("ha ha ha ha ha"), false);
  assertEquals(isRepetitionFlood("yes yes yes"), false);
});

Deno.test("collapseDuplicatedText behavior", () => {
  // 1. Should not interfere with short organic repetitions
  assertEquals(collapseDuplicatedText("yes yes yes"), "yes yes yes");
  assertEquals(
    collapseDuplicatedText("ha ha ha ha ha ha ha ha ha ha ha ha ha ha"),
    "ha ha ha ha ha ha ha ha ha ha ha ha ha ha",
  );

  // 2. Should collapse perfect duplicate > 80 chars
  const longSentence =
    "This is a very long sentence designed to test the duplication collapsing helper. It must be more than eighty characters long.";
  const duplicated = `${longSentence} ${longSentence}`;
  assertEquals(collapseDuplicatedText(duplicated), longSentence);

  // 3. Should not collapse non-perfect duplicates
  const almostDuplicated =
    `${longSentence} ${longSentence} slightly different ending.`;
  assertEquals(collapseDuplicatedText(almostDuplicated), almostDuplicated);
});

Deno.test("geminiOutputToHistoryEvents collapses duplicated text parts", () => {
  const longSentence =
    "This is a very long sentence designed to test the duplication collapsing helper. It must be more than eighty characters long.";
  const duplicated = `${longSentence} ${longSentence}`;

  const output = [
    {
      type: "text" as const,
      text: duplicated,
    },
  ];

  const events = geminiOutputToHistoryEvents(output);
  assertEquals(events.length, 1);
  assertEquals(events[0].type, "own_utterance");
  assertEquals(
    (events[0] as Extract<HistoryEvent, { type: "own_utterance" }>).text,
    longSentence,
  );
});

Deno.test("isRepetitionFlood returns true for </u> flood", () => {
  const flood = "Some text " + "</u>".repeat(70);
  assertEquals(isRepetitionFlood(flood), true);
});

Deno.test("isRepetitionFlood returns true for varied repeated tokens", () => {
  assertEquals(isRepetitionFlood("</b>".repeat(50)), true);
  assertEquals(isRepetitionFlood("ab".repeat(40)), true);
  assertEquals(isRepetitionFlood("x".repeat(30)), true);
});

Deno.test("isRepetitionFlood boundary: exactly 30 reps triggers, 29 does not", () => {
  assertEquals(isRepetitionFlood("</u>".repeat(30)), true);
  assertEquals(isRepetitionFlood("</u>".repeat(29)), false);
});

Deno.test(
  "repetition flood in model response causes retry and eventual throw",
  async () => {
    const history: HistoryEvent[] = [];
    const repetitionFlood = "Hello " + "</u>".repeat(100);
    let callCount = 0;

    const mockCallModel = (_h: HistoryEvent[]): Promise<HistoryEvent[]> => {
      callCount++;
      return Promise.resolve([{
        type: "own_utterance" as const,
        id: crypto.randomUUID(),
        timestamp: Date.now(),
        isOwn: true as const,
        text: repetitionFlood,
      }]);
    };

    await assertRejects(
      () =>
        injectAccessHistory(() => Promise.resolve(history))(
          injectOutputEvent((event) => {
            history.push(event);
            return Promise.resolve();
          })(runAbstractAgent),
        )(
          {
            maxIterations: 10,
            tools: [],
            prompt: "test",
            rewriteHistory: async () => {},
            timezoneIANA: "UTC",
          },
          mockCallModel,
        ),
      Error,
      "model keeps producing repetition flood responses",
    );

    assertEquals(callCount, 3, "should have retried exactly 3 times");
    assertEquals(history.length, 0, "no events should have been emitted");
  },
);

Deno.test(
  "repetition flood recovery: model succeeds after initial flood",
  async () => {
    const history: HistoryEvent[] = [];
    let callCount = 0;

    const mockCallModel = (_h: HistoryEvent[]): Promise<HistoryEvent[]> => {
      callCount++;
      if (callCount <= 2) {
        return Promise.resolve([{
          type: "own_utterance" as const,
          id: crypto.randomUUID(),
          timestamp: Date.now(),
          isOwn: true as const,
          text: "Hi " + "</u>".repeat(100),
        }]);
      }
      return Promise.resolve([{
        type: "own_utterance" as const,
        id: crypto.randomUUID(),
        timestamp: Date.now(),
        isOwn: true as const,
        text: "A normal response",
      }]);
    };

    await injectAccessHistory(() => Promise.resolve(history))(
      injectOutputEvent((event) => {
        history.push(event);
        return Promise.resolve();
      })(runAbstractAgent),
    )(
      {
        maxIterations: 10,
        tools: [],
        prompt: "test",
        rewriteHistory: async () => {},
        timezoneIANA: "UTC",
      },
      mockCallModel,
    );

    assertEquals(callCount, 3, "should have called model 3 times");
    assertEquals(history.length, 1, "should have emitted the normal response");
    const emitted = history[0];
    assertEquals(emitted.type, "own_utterance");
    if (emitted.type === "own_utterance") {
      assertEquals(emitted.text, "A normal response");
    }
  },
);

Deno.test(
  "empty own_utterance from model is not emitted to user",
  async () => {
    const history: HistoryEvent[] = [];
    let callCount = 0;

    const mockCallModel = (_h: HistoryEvent[]): Promise<HistoryEvent[]> => {
      callCount++;
      if (callCount === 1) {
        return Promise.resolve([{
          type: "own_utterance" as const,
          id: crypto.randomUUID(),
          timestamp: Date.now(),
          isOwn: true as const,
          text: "",
        }]);
      }
      return Promise.resolve([{
        type: "own_utterance" as const,
        id: crypto.randomUUID(),
        timestamp: Date.now(),
        isOwn: true as const,
        text: "Real response",
      }]);
    };

    await injectAccessHistory(() => Promise.resolve(history))(
      injectOutputEvent((event) => {
        history.push(event);
        return Promise.resolve();
      })(runAbstractAgent),
    )(
      {
        maxIterations: 5,
        tools: [],
        prompt: "test",
        rewriteHistory: async () => {},
        timezoneIANA: "UTC",
      },
      mockCallModel,
    );

    const emptyUtterances = history.filter(
      (e) => e.type === "own_utterance" && !e.text.trim(),
    );
    assertEquals(
      emptyUtterances.length,
      0,
      "should not emit empty own_utterance events",
    );
    assert(
      history.some((e) =>
        e.type === "own_utterance" && e.text === "Real response"
      ),
      "should emit the real response",
    );
  },
);

Deno.test(
  "whitespace-only own_utterance from model is not emitted to user",
  async () => {
    const history: HistoryEvent[] = [];
    let callCount = 0;

    const mockCallModel = (_h: HistoryEvent[]): Promise<HistoryEvent[]> => {
      callCount++;
      if (callCount === 1) {
        return Promise.resolve([{
          type: "own_utterance" as const,
          id: crypto.randomUUID(),
          timestamp: Date.now(),
          isOwn: true as const,
          text: "   \n  ",
        }]);
      }
      return Promise.resolve([{
        type: "own_utterance" as const,
        id: crypto.randomUUID(),
        timestamp: Date.now(),
        isOwn: true as const,
        text: "Real response",
      }]);
    };

    await injectAccessHistory(() => Promise.resolve(history))(
      injectOutputEvent((event) => {
        history.push(event);
        return Promise.resolve();
      })(runAbstractAgent),
    )(
      {
        maxIterations: 5,
        tools: [],
        prompt: "test",
        rewriteHistory: async () => {},
        timezoneIANA: "UTC",
      },
      mockCallModel,
    );

    const emptyUtterances = history.filter(
      (e) => e.type === "own_utterance" && !e.text.trim(),
    );
    assertEquals(
      emptyUtterances.length,
      0,
      "should not emit whitespace-only own_utterance events",
    );
  },
);

Deno.test(
  "own_utterance with empty text but attachments is preserved",
  async () => {
    const history: HistoryEvent[] = [];

    const mockCallModel = (_h: HistoryEvent[]): Promise<HistoryEvent[]> =>
      Promise.resolve([{
        type: "own_utterance" as const,
        id: crypto.randomUUID(),
        timestamp: Date.now(),
        isOwn: true as const,
        text: "",
        attachments: [{
          kind: "inline" as const,
          mimeType: "image/png",
          dataBase64: "iVBORw0KGgo=",
        }],
      }]);

    await injectAccessHistory(() => Promise.resolve(history))(
      injectOutputEvent((event) => {
        history.push(event);
        return Promise.resolve();
      })(runAbstractAgent),
    )(
      {
        maxIterations: 2,
        tools: [],
        prompt: "test",
        rewriteHistory: async () => {},
        timezoneIANA: "UTC",
      },
      mockCallModel,
    );

    const utterances = history.filter((e) => e.type === "own_utterance");
    assertEquals(
      utterances.length,
      1,
      "should preserve utterance with attachments",
    );
  },
);

Deno.test(
  "intentional do_nothing from model stops the agent",
  async () => {
    const history: HistoryEvent[] = [];
    let callCount = 0;

    const mockCallModel = (_h: HistoryEvent[]): Promise<HistoryEvent[]> => {
      callCount++;
      return Promise.resolve([{
        type: "do_nothing" as const,
        id: crypto.randomUUID(),
        timestamp: Date.now(),
        isOwn: true as const,
      }]);
    };

    await injectAccessHistory(() => Promise.resolve(history))(
      injectOutputEvent((event) => {
        history.push(event);
        return Promise.resolve();
      })(runAbstractAgent),
    )(
      {
        maxIterations: 5,
        tools: [],
        prompt: "test",
        rewriteHistory: async () => {},
        timezoneIANA: "UTC",
      },
      mockCallModel,
    );

    assertEquals(callCount, 1, "should call model only once and stop");
    assert(
      history.some((e) => e.type === "do_nothing"),
      "should emit the do_nothing event",
    );
  },
);

Deno.test(
  "calling do_nothing tool emits do_nothing event and stops the agent",
  async () => {
    const history: HistoryEvent[] = [];

    const mockCallModel = (_h: HistoryEvent[]): Promise<HistoryEvent[]> =>
      Promise.resolve([{
        type: "tool_call" as const,
        id: crypto.randomUUID(),
        timestamp: Date.now(),
        isOwn: true as const,
        name: "do_nothing",
        parameters: {},
        modelMetadata: undefined,
      }]);

    await injectAccessHistory(() => Promise.resolve(history))(
      injectOutputEvent((event) => {
        history.push(event);
        return Promise.resolve();
      })(runAbstractAgent),
    )(
      {
        maxIterations: 5,
        tools: [],
        prompt: "test",
        rewriteHistory: async () => {},
        timezoneIANA: "UTC",
      },
      mockCallModel,
    );

    const visibleUtterances = history.filter(
      (e) => e.type === "own_utterance" && e.text.trim() !== "",
    );
    assertEquals(
      visibleUtterances.length,
      0,
      "should have no visible utterance",
    );
    assert(
      history.some((e) => e.type === "do_nothing"),
      "should emit a do_nothing event",
    );
    assert(
      !history.some((e) => e.type === "tool_result"),
      "should not emit a tool_result for do_nothing",
    );
  },
);

const longParagraphs = (count: number, charsPerParagraph: number) =>
  Array.from(
    { length: count },
    (_, i) => `Paragraph ${i + 1}: ${"x".repeat(charsPerParagraph)}`,
  ).join("\n\n");

Deno.test("sanitizeModelOutput splits over-cap own_utterance into multiple", () => {
  const original = longParagraphs(6, 1500);
  assert(original.length > maxUtteranceChars);
  const { emit } = sanitizeModelOutput([], [ownUtteranceTurn(original)]);
  const utterances = emit.filter((e) => e.type === "own_utterance");
  assert(utterances.length > 1, "expected split into multiple utterances");
  utterances.forEach((u) => {
    if (u.type !== "own_utterance") throw new Error("unreachable");
    assert(
      u.text.length <= maxUtteranceChars,
      `chunk length ${u.text.length} exceeds cap ${maxUtteranceChars}`,
    );
    assert(u.text.length > 0, "chunk should be non-empty");
  });
  const ids = new Set(utterances.map((u) => u.id));
  assertEquals(ids.size, utterances.length, "chunks must have unique ids");
});

Deno.test("sanitizeModelOutput leaves under-cap own_utterance unchanged", () => {
  const event = ownUtteranceTurn("short message");
  const { emit } = sanitizeModelOutput([], [event]);
  assertEquals(emit.length, 1);
  assertEquals(emit[0], event);
});

Deno.test("sanitizeModelOutput hard-splits a single huge wordless run", () => {
  const original = "a".repeat(maxUtteranceChars * 2 + 137);
  const { emit } = sanitizeModelOutput([], [ownUtteranceTurn(original)]);
  const utterances = emit.filter((e) => e.type === "own_utterance");
  assert(utterances.length >= 3);
  utterances.forEach((u) => {
    if (u.type !== "own_utterance") throw new Error("unreachable");
    assert(u.text.length <= maxUtteranceChars);
  });
});

const buildSyntheticTimeoutError = () => {
  const err = new Error("Model call timed out");
  Object.assign(err, { status: 503, [syntheticTimeoutMarker]: true });
  return err;
};

const buildRealServerError = () => {
  const err = new Error("Internal server error");
  Object.assign(err, { status: 503 });
  return err;
};

Deno.test("isSyntheticTimeoutError identifies marked synthetic timeouts", () => {
  assert(isSyntheticTimeoutError(buildSyntheticTimeoutError()));
  assert(!isSyntheticTimeoutError(buildRealServerError()));
  assert(!isSyntheticTimeoutError(new Error("plain")));
  assert(!isSyntheticTimeoutError(null));
});

Deno.test("isRetryableError excludes synthetic timeouts to prevent retry-amplified hangs", () => {
  assert(
    !isRetryableError(buildSyntheticTimeoutError()),
    "synthetic 503 from withTimeout must not be retried; the underlying request may still complete and compound wall-clock under SIGTERM window",
  );
  assert(
    isRetryableError(buildRealServerError()),
    "real 503 from Gemini server must still be retried",
  );
  const rateLimit = new Error("too many requests");
  Object.assign(rateLimit, { status: 429 });
  assert(isRetryableError(rateLimit));
});

Deno.test("isSyntheticTimeoutError marks errors that should skip same-model retry but use fallback model", () => {
  const timeout = buildSyntheticTimeoutError();
  assert(isSyntheticTimeoutError(timeout));
  assert(!isRetryableError(timeout));
});

Deno.test("Gemini timeout fallback model differs from primary models", () => {
  assertEquals(new Set([geminiFallbackVersion, geminiFlashVersion]).size, 2);
  assertEquals(new Set([geminiFallbackVersion, geminiProVersion]).size, 2);
});

Deno.test("Gemini MALFORMED_FUNCTION_CALL is retryable instead of do_nothing", () => {
  const error = assertThrows(
    () => rejectMalformedFunctionCall("MALFORMED_FUNCTION_CALL", []),
    Error,
    "MALFORMED_FUNCTION_CALL",
  );
  assert(
    isRetryableError(error),
    "Gemini malformed function-call responses should use retry/fallback path, not empty-output do_nothing",
  );
  assert(isRetryableError(geminiMalformedFunctionCallError([])));
});

Deno.test("isRepetitionFlood detects repeated sub-strings", () => {
  assert(isRepetitionFlood("abc".repeat(30)));
  assert(isRepetitionFlood("X".repeat(30)));
  assert(!isRepetitionFlood("abcabc"));
  assert(!isRepetitionFlood("normal sentence with normal words"));
});

Deno.test("formatSkillsPrompt appends compact tool names and descriptions under each skill", () => {
  const dummyTool = tool({
    name: "add_numbers",
    description: "Add two numbers together",
    parameters: z.object({ a: z.number(), b: z.number() }),
    handler: () => Promise.resolve(""),
  });
  const skills: Skill[] = [{
    name: "calculator",
    description: "Mathematical operations skill",
    instructions: "Do math",
    tools: [dummyTool],
  }];
  const prompt = formatSkillsPrompt(skills);
  assertEquals(
    prompt,
    "- calculator: Mathematical operations skill\n  Tools:\n    - add_numbers: Add two numbers together",
  );
});

Deno.test("resolveToolDescription resolves description from run_command spinnerText", () => {
  const dummySkillTool = tool({
    name: "dummy_tool",
    description: "A dummy tool",
    parameters: z.object({ comment: z.string().optional() }),
    handler: () => Promise.resolve(""),
  });
  const skills: Skill[] = [{
    name: "dummy_skill",
    description: "A dummy skill",
    instructions: "Dummy",
    tools: [dummySkillTool],
  }];
  const allTools = createSkillTools(skills);
  const resolved = resolveToolDescription(
    allTools,
    "run_command",
    {
      command: "dummy_skill/dummy_tool",
      params: {},
      spinnerText: "Performing a background task",
    },
    skills,
  );
  assertEquals(resolved, "Performing a background task");
});

import { zodToGeminiParameters } from "../src/gemini.ts";

Deno.test("zodToGeminiParameters converts ZodLiteral (const) to enum with single value", () => {
  const schema = z.object({
    type: z.literal("custom_event"),
    value: z.string(),
  });
  const result = zodToGeminiParameters(schema);
  // deno-lint-ignore no-explicit-any
  const typeProp = (result as any).properties?.type;
  assertEquals(typeProp.const, undefined);
  assertEquals(typeProp.enum, ["custom_event"]);
});

Deno.test({
  name:
    "estimateAgentInputTokens accurately estimates non-ASCII (Hebrew) characters higher than English characters",
}, async () => {
  const spec = {
    prompt: "",
    tools: [],
    skills: [],
    maxIterations: 5,
    rewriteHistory: (_replacements: Record<string, HistoryEvent>) =>
      Promise.resolve(),
    timezoneIANA: "UTC",
  };

  const englishText =
    "hello world! this is a test of english token estimation.";
  const hebrewText =
    "שלום עולם! זהו מבחן של הערכת אסימונים בעברית לעומת אנגלית.";

  const englishEvent = {
    id: "1",
    timestamp: 1000,
    type: "participant_utterance" as const,
    text: englishText,
    isOwn: false as const,
    name: "user",
  };

  const hebrewEvent = {
    id: "2",
    timestamp: 1000,
    type: "participant_utterance" as const,
    text: hebrewText,
    isOwn: false as const,
    name: "user",
  };

  const englishTokens = await estimateAgentInputTokens(spec, [englishEvent]);
  const hebrewTokens = await estimateAgentInputTokens(spec, [hebrewEvent]);

  assertEquals(englishTokens < hebrewTokens, true);
  assertEquals(hebrewTokens > 50, true);
});
