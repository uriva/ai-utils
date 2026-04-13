import { assert, assertEquals } from "@std/assert";
import { sleep } from "gamla";
import { z } from "zod/v4";
import { runAgent } from "../mod.ts";
import { runForAllProviders } from "../test_helpers.ts";
import {
  type DeferredTool,
  type HistoryEvent,
  ownThoughtTurnWithMetadata,
  ownUtteranceTurn,
  participantUtteranceTurn,
} from "../src/agent.ts";
import {
  agentDeps,
  injectSecrets,
  llmTest,
  noopRewriteHistory,
  someTool,
  toolResult,
} from "../test_helpers.ts";

Deno.test(
  "runBot calls the tool and replies with its output",
  injectSecrets(async () => {
    const mockHistory: HistoryEvent[] = [participantUtteranceTurn({
      name: "user",
      text:
        `Please call the doSomethingUnique tool now and only reply with its output.`,
    })];
    await agentDeps(mockHistory)(runAgent)({
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
  }),
);

llmTest(
  "ai returns text event before calling actions",
  injectSecrets(async () => {
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text:
          `Please call the doSomethingUnique tool and explain what you're doing.`,
      }),
    ];
    await agentDeps(mockHistory)(runAgent)({
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
    assert(
      firstTextIndex < firstToolIndex,
      "Text should come before tool call",
    );
  }),
);

Deno.test(
  "ai handles new history items while waiting for function calls",
  injectSecrets(async () => {
    const mockHistory: HistoryEvent[] = [participantUtteranceTurn({
      name: "user",
      text: `Please call the slowTool.`,
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

    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 10,
      onMaxIterationsReached: () => {},
      tools: [slowTool],
      prompt: `You are an AI assistant. Always acknowledge new messages.`,
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

    const responseAfterContext = mockHistory.find((event) =>
      event.type === "own_utterance" &&
      event.text &&
      event.timestamp > addedContext.timestamp
    );
    assert(
      responseAfterContext,
      "AI should respond to additional message in next iteration",
    );
  }),
);

llmTest(
  "maxIterationsReached aborts the loop",
  injectSecrets(async () => {
    let callbackCalled = false;
    const mockHistory: HistoryEvent[] = [participantUtteranceTurn({
      name: "user",
      text: `Please keep talking and calling tools continuously.`,
    })];
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
        `You are a chatty AI. Always call the continueTalking tool in every response and keep the conversation going.`,
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
    assert(callbackCalled, "onMaxIterationsReached callback should be called");
  }),
);

llmTest(
  "agent triggers do nothing event after conversation ends with goodbye",
  injectSecrets(async () => {
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({ name: "user", text: "What is 2+2?" }),
      ownUtteranceTurn("4"),
      participantUtteranceTurn({ name: "user", text: "Thanks, bye!" }),
      ownUtteranceTurn("Bye!"),
      participantUtteranceTurn({ name: "user", text: "\u{1F44D}" }),
    ];
    await agentDeps(mockHistory)(runAgent)({
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
  }),
);

Deno.test(
  "deferred tool handler is called with toolCallId and agent exits without emitting tool_result",
  injectSecrets(async () => {
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
    await agentDeps(mockHistory)(runAgent)({
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
  }),
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
        text: "What is 1+1?",
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
