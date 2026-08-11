import { assertEquals } from "@std/assert";
import { runAgent } from "../mod.ts";
import { agentDeps, noopRewriteHistory, someTool } from "../test_helpers.ts";
import {
  doNothingEvent,
  type HistoryEvent,
  injectCallModel,
  ownUtteranceTurn,
  participantUtteranceTurn,
} from "../src/agent.ts";

Deno.test(
  "runAgent - nudges model when do_nothing is emitted after tool calls without answering user message",
  async () => {
    const history: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Create a customer service AI agent for my business.",
      }),
    ];

    let turnCount = 0;
    const fakeCallModel = (): Promise<HistoryEvent[]> => {
      turnCount++;
      if (turnCount === 1) {
        return Promise.resolve([
          {
            type: "tool_call",
            id: "call-1",
            name: "test_tool",
            parameters: {},
            description: "Executing test tool",
            isOwn: true as const,
            timestamp: Date.now(),
          },
        ]);
      }
      if (turnCount === 2) {
        // Model returns do_nothing on turn 2 after tool execution (the production bug)
        return Promise.resolve([doNothingEvent()]);
      }
      return Promise.resolve([
        ownUtteranceTurn("Here is your created agent!"),
      ]);
    };

    await injectCallModel(fakeCallModel)(async () => {
      await agentDeps(history)(runAgent)({
        maxIterations: 10,
        tools: [someTool],
        prompt: "You are an agent builder.",
        rewriteHistory: noopRewriteHistory,
        timezoneIANA: "UTC",
      });
    })();

    const ownUtterances = history.filter((e) => e.type === "own_utterance");
    assertEquals(
      ownUtterances.length,
      1,
      "Expected agent to send an own_utterance to the user after being nudged",
    );
    assertEquals(ownUtterances[0].text, "Here is your created agent!");
  },
);
