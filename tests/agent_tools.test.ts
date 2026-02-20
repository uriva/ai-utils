import { assert, assertEquals } from "@std/assert";
import { sleep } from "gamla";
import { z } from "zod/v4";
import { runAgent } from "../mod.ts";
import {
  type HistoryEvent,
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

Deno.test(
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
