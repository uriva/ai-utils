import { assert, assertEquals } from "@std/assert";
import { sleep } from "gamla";
import { z } from "zod/v4";
import { runAgent } from "../mod.ts";
import { runForAllProviders } from "../test_helpers.ts";
import {
  type DeferredTool,
  type HistoryEvent,
  injectCallModel,
  ownThoughtTurnWithMetadata,
  ownUtteranceTurn,
  participantUtteranceTurn,
} from "../src/agent.ts";
import {
  agentDeps,
  noopRewriteHistory,
  someTool,
  toolResult,
} from "../test_helpers.ts";

runForAllProviders(
  "runBot calls the tool and replies with its output",
  async (runAgentWithProvider) => {
    const mockHistory: HistoryEvent[] = [participantUtteranceTurn({
      name: "user",
      text:
        `Please call the doSomethingUnique tool now and only reply with its output.`,
    })];
    await agentDeps(mockHistory)(runAgentWithProvider)({
      maxIterations: 5,
      onMaxIterationsReached: () => {},
      tools: [someTool],
      prompt: `You are an AI assistant.`,
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
    assert(
      mockHistory.some((event) => (event.type === "own_utterance" &&
        event.text.includes(toolResult))
      ),
      `AI did not reply with tool output. History: ${
        JSON.stringify(mockHistory, null, 2)
      }`,
    );
  },
);

runForAllProviders(
  "ai returns text event before calling actions",
  async (runAgentWithProvider) => {
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text:
          `Please call the doSomethingUnique tool and explain what you're doing.`,
      }),
    ];
    await agentDeps(mockHistory)(runAgentWithProvider)({
      maxIterations: 5,
      onMaxIterationsReached: () => {},
      tools: [someTool],
      prompt:
        `You are an AI assistant. Always explain what you're doing before using tools.`,
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
    const firstTextIndex = mockHistory.findIndex((event) =>
      event.type === "own_utterance" && event.text
    );
    const firstToolIndex = mockHistory.findIndex((event) =>
      event.type === "tool_call"
    );
    assert(firstTextIndex >= 0, "AI should produce text output");
    assert(firstToolIndex >= 0, "AI should call the tool");
  },
);

runForAllProviders(
  "ai handles new history items while waiting for function calls",
  async (runAgentWithProvider) => {
    const mockHistory: HistoryEvent[] = [participantUtteranceTurn({
      name: "user",
      text:
        "Call the slowTool now. After it finishes, acknowledge my follow-up message.",
    })];

    const slowTool = {
      name: "slowTool",
      description: "A tool that takes time to execute",
      parameters: z.object({}),
      handler: async () => {
        await sleep(10);
        mockHistory.push(participantUtteranceTurn({
          name: "user",
          text: "While you're working, here's additional context.",
        }));
        return "slow result";
      },
    };

    await agentDeps(mockHistory)(runAgentWithProvider)({
      maxIterations: 10,
      onMaxIterationsReached: () => {},
      tools: [slowTool],
      prompt:
        "You are an AI assistant. When the user asks for slowTool, call slowTool before answering. If a new user message arrives while a tool is running, respond to that new message on the next iteration.",
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });

    const toolCall = mockHistory.find((e) =>
      e.type === "tool_call" && e.name === "slowTool"
    );
    assert(toolCall, "slowTool should be called");

    const addedContext = mockHistory.find((e) =>
      e.type === "participant_utterance" &&
      e.text?.includes("additional context")
    );
    assert(addedContext, "Additional message should be in history");

    const addedContextIndex = mockHistory.findIndex((event) =>
      event === addedContext
    );
    const responseAfterContext = mockHistory.slice(addedContextIndex + 1).find((
      event,
    ) => event.type === "own_utterance" && event.text);
    assert(
      responseAfterContext,
      "AI should respond to additional message in next iteration",
    );
  },
  8,
);

Deno.test(
  "maxIterationsReached aborts the loop",
  async () => {
    let callbackCalled = false;
    const mockHistory: HistoryEvent[] = [participantUtteranceTurn({
      name: "user",
      text:
        "Keep going forever. On every turn, call continueTalking again before anything else.",
    })];
    let n = 0;
    const fakeCallModel = () =>
      Promise.resolve([
        {
          type: "tool_call" as const,
          isOwn: true as const,
          name: "continueTalking",
          parameters: {},
          id: `fake-tool-${++n}`,
          timestamp: Date.now(),
        },
      ]);
    await injectCallModel(fakeCallModel)(async () => {
      await agentDeps(mockHistory)(runAgent)({
        maxIterations: 3,
        onMaxIterationsReached: () => {
          callbackCalled = true;
        },
        tools: [{
          name: "continueTalking",
          description: "A tool that keeps the conversation going",
          parameters: z.object({}),
          handler: async () => {
            await sleep(5);
            mockHistory.push(participantUtteranceTurn({
              name: "user",
              text: "Keep going, call the tool again!",
            }));
            return "continue";
          },
        }],
        prompt:
          "You are a chatty AI. In every response, call continueTalking before any text. Never stop the loop on your own.",
        rewriteHistory: noopRewriteHistory,
        timezoneIANA: "UTC",
      });
    })();
    assert(callbackCalled, "onMaxIterationsReached callback should be called");
  },
);

runForAllProviders(
  "agent triggers do nothing event after conversation ends with goodbye",
  async (runAgentWithProvider) => {
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({ name: "user", text: "What is 2+2?" }),
      ownUtteranceTurn("4"),
      participantUtteranceTurn({ name: "user", text: "Thanks, bye!" }),
      ownUtteranceTurn("Bye!"),
      participantUtteranceTurn({ name: "user", text: "\u{1F44D}" }),
    ];
    await agentDeps(mockHistory)(runAgentWithProvider)({
      maxIterations: 1,
      onMaxIterationsReached: () => {},
      tools: [],
      prompt:
        "You are a helpful but concise assistant. When a conversation has clearly ended (goodbyes exchanged), do not respond further. A thumbs up or similar acknowledgment after goodbyes does not require a response.",
      lightModel: true,
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
    assertEquals(mockHistory[mockHistory.length - 1].type, "do_nothing");
  },
);

runForAllProviders(
  "deferred tool handler is called with toolCallId and agent exits without emitting tool_result",
  async (runAgentWithProvider) => {
    let capturedToolCallId: string | undefined;
    const deferredTool: DeferredTool<z.ZodObject<{ ms: z.ZodNumber }>> = {
      name: "timeout-wakeup",
      description: "Set a timeout to wake up later",
      parameters: z.object({
        ms: z.number().describe("Milliseconds to wait"),
      }),
      // deno-lint-ignore require-await
      handler: async (_params, toolCallId) => {
        capturedToolCallId = toolCallId;
      },
    };
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text:
          "Please call the timeout-wakeup tool with ms=5000. Do not say anything else.",
      }),
    ];
    await agentDeps(mockHistory)(runAgentWithProvider)({
      maxIterations: 5,
      onMaxIterationsReached: () => {},
      tools: [deferredTool],
      prompt:
        "You are an assistant. When asked, call the timeout-wakeup tool with the requested parameters.",
      lightModel: true,
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
    assert(
      capturedToolCallId,
      "Deferred tool handler should have been called with a toolCallId",
    );
    assert(
      !mockHistory.some((e) => e.type === "tool_result"),
      "No tool_result should be emitted for deferred tools",
    );
    assert(
      mockHistory.some((e) =>
        e.type === "tool_call" && e.name === "timeout-wakeup"
      ),
      "tool_call should be in history",
    );
  },
);

runForAllProviders(
  "does not throw when there is an orphaned tool call",
  async (runAgent) => {
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Please do something.",
      }),
      {
        type: "tool_call",
        isOwn: true,
        timestamp: Date.now(),
        name: "doSomethingUnique",
        parameters: { param: "value" },
        id: "call_123",
      },
      participantUtteranceTurn({
        name: "user",
        text: "What were the contact names?",
      }),
    ];

    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 1,
      onMaxIterationsReached: () => {},
      tools: [someTool],
      prompt: "You are an AI assistant.",
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
  },
);

runForAllProviders(
  "does not throw when own_thought with modelMetadata is in history",
  async (runAgent) => {
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Hello, how are you?",
      }),
      ownThoughtTurnWithMetadata("The user is greeting me.", {
        type: "kimi",
        responseId: "test-response-1",
      }),
      ownUtteranceTurn("I'm doing well, thanks!"),
      participantUtteranceTurn({
        name: "user",
        text: "What were the contact names?",
      }),
    ];

    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 1,
      onMaxIterationsReached: () => {},
      tools: [someTool],
      prompt: "You are an AI assistant.",
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
  },
);

runForAllProviders(
  "zod .default() on a tool param is honored when the model omits it",
  async (runAgent) => {
    const received: { prefix: string; skip?: number }[] = [];
    const searchTool = {
      name: "search_items",
      description: "Search items by a required prefix, skipping past results.",
      parameters: z.object({
        prefix: z.string().describe("Prefix to search for"),
        skip: z.number().describe(
          "How many results to skip past. Defaults to 0.",
        ).default(0),
      }),
      handler: (args: { prefix: string; skip: number }) => {
        received.push(args);
        return Promise.resolve(`found 1 item starting with ${args.prefix}`);
      },
    };

    const mockHistory: HistoryEvent[] = [participantUtteranceTurn({
      name: "user",
      text:
        `Call the search_items tool with prefix "abc". Do not include any other fields.`,
    })];

    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 5,
      onMaxIterationsReached: () => {},
      tools: [searchTool],
      prompt: "You are an AI assistant. Call tools exactly as the user asks.",
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });

    assert(
      received.length > 0,
      `search_items handler was never called. History: ${
        JSON.stringify(mockHistory, null, 2)
      }`,
    );
    assertEquals(
      received[0].prefix,
      "abc",
      "handler should receive the prefix the model sent",
    );
    assertEquals(
      received[0].skip,
      0,
      "handler should receive the zod default (0) for the omitted skip param",
    );
  },
);

runForAllProviders(
  "handles orphaned tool_results between assistant messages without crashing",
  async (runAgent) => {
    const now = Date.now();
    const toolCallId1 = crypto.randomUUID();
    const toolCallId2 = crypto.randomUUID();
    const orphanedId1 = crypto.randomUUID();
    const orphanedId2 = crypto.randomUUID();
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Save contact John and look up info.",
      }),
      {
        type: "tool_call",
        isOwn: true,
        timestamp: now,
        name: "doSomethingUnique",
        parameters: { foo: "bar" },
        id: toolCallId1,
      },
      {
        type: "tool_result",
        isOwn: true,
        timestamp: now + 1,
        result: "orphaned result",
        toolCallId: orphanedId1,
        id: crypto.randomUUID(),
      },
      ownUtteranceTurn("I found the information."),
      {
        type: "tool_result",
        isOwn: true,
        timestamp: now + 3,
        result: "another orphaned result",
        toolCallId: orphanedId2,
        id: crypto.randomUUID(),
      },
      {
        type: "tool_call",
        isOwn: true,
        timestamp: now + 4,
        name: "doSomethingUnique",
        parameters: { foo: "baz" },
        id: toolCallId2,
      },
      {
        type: "tool_result",
        isOwn: true,
        timestamp: now + 5,
        result: "valid result",
        toolCallId: toolCallId2,
        id: crypto.randomUUID(),
      },
      participantUtteranceTurn({
        name: "user",
        text: "What did you find?",
      }),
    ];

    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 1,
      onMaxIterationsReached: () => {},
      tools: [someTool],
      prompt: "You are an AI assistant.",
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
  },
);

runForAllProviders(
  "does not throw when tool_result is not adjacent in raw history",
  async (runAgent) => {
    const now = Date.now();
    const toolCallId = crypto.randomUUID();
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Save contact John.",
      }),
      {
        type: "tool_call",
        isOwn: true,
        timestamp: now,
        name: "doSomethingUnique",
        parameters: {},
        id: toolCallId,
      },
      participantUtteranceTurn({
        name: "user",
        text: "Additional context before the tool result.",
      }),
      {
        type: "tool_result",
        isOwn: true,
        timestamp: now + 1,
        result: toolResult,
        toolCallId,
        id: crypto.randomUUID(),
      },
      participantUtteranceTurn({
        name: "user",
        text: "What was the result?",
      }),
    ];

    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 1,
      onMaxIterationsReached: () => {},
      tools: [someTool],
      prompt: "You are an AI assistant.",
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
  },
);

runForAllProviders(
  "anthropic streams text before tool calls when the model explains first",
  async (runAgentWithProvider) => {
    if (Deno.env.get("TEST_PROVIDER") !== "anthropic") return;

    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text:
          "First explain what you are about to do in one short sentence, then call the doSomethingUnique tool.",
      }),
    ];
    let streamedText = "";

    await agentDeps(mockHistory)(runAgentWithProvider)({
      maxIterations: 5,
      onMaxIterationsReached: () => {},
      tools: [someTool],
      prompt:
        "You are an AI assistant. Before any tool call, first say exactly 'Checking now.' in a normal message, then call the tool.",
      onStreamChunk: (chunk) => {
        streamedText += chunk;
      },
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });

    const firstTextIndex = mockHistory.findIndex((event) =>
      event.type === "own_utterance" && event.text.includes("Checking now")
    );
    const firstToolIndex = mockHistory.findIndex((event) =>
      event.type === "tool_call" && event.name === someTool.name
    );

    assert(firstTextIndex >= 0, "AI should emit the pre-tool text message");
    assert(firstToolIndex >= 0, "AI should call the tool");
    assert(
      streamedText.includes("Checking now."),
      "onStreamChunk should include the pre-tool text",
    );
  },
  3,
);

runForAllProviders(
  "does not throw when tool_call id mismatches tool_result toolCallId",
  async (runAgent) => {
    const now = Date.now();
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Save contact John.",
      }),
      {
        type: "tool_call",
        isOwn: true,
        timestamp: now,
        name: "upsert_contact",
        parameters: { name: "John" },
        id: "upsert_contact:3",
      },
      {
        type: "tool_result",
        isOwn: true,
        timestamp: now + 1,
        result: "Contact saved.",
        toolCallId: crypto.randomUUID(),
        id: crypto.randomUUID(),
      },
      participantUtteranceTurn({
        name: "user",
        text: "What was the contact name?",
      }),
    ];

    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 1,
      onMaxIterationsReached: () => {},
      tools: [someTool],
      prompt: "You are an AI assistant.",
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
  },
);

runForAllProviders(
  "does not throw when tool_result is missing (compaction dropped it)",
  async (runAgent) => {
    const now = Date.now();
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Save contact John.",
      }),
      {
        type: "tool_call",
        isOwn: true,
        timestamp: now,
        name: "upsert_contact",
        parameters: { name: "John" },
        id: "upsert_contact:3",
      },
      participantUtteranceTurn({
        name: "user",
        text: "What was the contact name?",
      }),
    ];

    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 1,
      onMaxIterationsReached: () => {},
      tools: [someTool],
      prompt: "You are an AI assistant.",
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
  },
);

runForAllProviders(
  "handles consecutive tool_calls with missing tool_results after compaction",
  async (runAgent) => {
    const now = Date.now();
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Save contacts John and Jane.",
      }),
      {
        type: "tool_call",
        isOwn: true,
        timestamp: now,
        name: "upsert_contact",
        parameters: { name: "John" },
        id: "upsert_contact:3",
      },
      {
        type: "tool_call",
        isOwn: true,
        timestamp: now + 1,
        name: "upsert_contact",
        parameters: { name: "Jane" },
        id: "upsert_contact:4",
      },
      {
        type: "tool_result",
        isOwn: true,
        timestamp: now + 2,
        result: "Contact Jane saved.",
        toolCallId: "upsert_contact:4",
        id: crypto.randomUUID(),
      },
      participantUtteranceTurn({
        name: "user",
        text: "What were the contact names?",
      }),
    ];

    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 1,
      onMaxIterationsReached: () => {},
      tools: [someTool],
      prompt: "You are an AI assistant.",
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
  },
);

runForAllProviders(
  "handles consecutive tool_calls with all tool_results present",
  async (runAgent) => {
    const now = Date.now();
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Save contacts John and Jane.",
      }),
      {
        type: "tool_call",
        isOwn: true,
        timestamp: now,
        name: "upsert_contact",
        parameters: { name: "John" },
        id: "upsert_contact:3",
      },
      {
        type: "tool_call",
        isOwn: true,
        timestamp: now + 1,
        name: "upsert_contact",
        parameters: { name: "Jane" },
        id: "upsert_contact:4",
      },
      {
        type: "tool_result",
        isOwn: true,
        timestamp: now + 2,
        result: "Contact John saved.",
        toolCallId: "upsert_contact:3",
        id: crypto.randomUUID(),
      },
      {
        type: "tool_result",
        isOwn: true,
        timestamp: now + 3,
        result: "Contact Jane saved.",
        toolCallId: "upsert_contact:4",
        id: crypto.randomUUID(),
      },
      participantUtteranceTurn({
        name: "user",
        text: "What were the contact names?",
      }),
    ];

    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 1,
      onMaxIterationsReached: () => {},
      tools: [someTool],
      prompt: "You are an AI assistant.",
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
  },
);
