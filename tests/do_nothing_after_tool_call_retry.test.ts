import { assertEquals } from "@std/assert";
import { runAgent } from "../mod.ts";
import { agentDeps, noopRewriteHistory, someTool } from "../test_helpers.ts";
import {
  doNothingEvent,
  doNothingToolName,
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

Deno.test(
  "runAgent - respects explicit do_nothing tool call when model chooses to stay silent",
  async () => {
    const history: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Check my email but if no emails don't answer me.",
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
            description: "Checking email",
            isOwn: true as const,
            timestamp: Date.now(),
          },
        ]);
      }
      // Model explicitly calls do_nothing tool because no emails were found per user instruction
      return Promise.resolve([
        {
          type: "tool_call",
          id: "call-2",
          name: doNothingToolName,
          parameters: {
            reason: "No emails found, remaining silent per user request.",
          },
          description: "Staying silent per user instruction",
          isOwn: true as const,
          timestamp: Date.now(),
        },
      ]);
    };

    await injectCallModel(fakeCallModel)(async () => {
      await agentDeps(history)(runAgent)({
        maxIterations: 10,
        tools: [someTool],
        prompt:
          "You check emails. If user asks to stay silent when no emails, call do_nothing.",
        rewriteHistory: noopRewriteHistory,
        timezoneIANA: "UTC",
      });
    })();

    const ownUtterances = history.filter((e) => e.type === "own_utterance");
    assertEquals(
      ownUtterances.length,
      0,
      "Expected agent to remain silent when explicitly calling do_nothing tool",
    );
    assertEquals(
      turnCount,
      2,
      "Expected agent to exit on turn 2 when calling do_nothing tool explicitly",
    );
  },
);

Deno.test(
  "runAgent - respects model choosing do_nothing twice (max retries reached)",
  async () => {
    const history: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Check my email but if no emails don't answer me.",
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
            description: "Checking email",
            isOwn: true as const,
            timestamp: Date.now(),
          },
        ]);
      }
      // Model persists in choosing do_nothing even after nudge
      return Promise.resolve([doNothingEvent()]);
    };

    await injectCallModel(fakeCallModel)(async () => {
      await agentDeps(history)(runAgent)({
        maxIterations: 10,
        tools: [someTool],
        prompt:
          "You check emails. If user asks to stay silent when no emails, stay silent.",
        rewriteHistory: noopRewriteHistory,
        timezoneIANA: "UTC",
      });
    })();

    const ownUtterances = history.filter((e) => e.type === "own_utterance");
    assertEquals(
      ownUtterances.length,
      0,
      "Expected agent to remain silent after max do_nothing retries",
    );
  },
);
