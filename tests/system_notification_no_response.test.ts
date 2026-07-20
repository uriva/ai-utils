import { assert, assertEquals } from "@std/assert";
import {
  doNothingTool,
  doNothingToolName,
  type HistoryEvent,
  ownThoughtTurn,
} from "../src/agent.ts";
import { agentDeps, runForAllProviders } from "../test_helpers.ts";

runForAllProviders(
  "agent should not send chat replies to system notifications and instead use do_nothing",
  async (runAgentWithProvider) => {
    // History contains a completed conversation where the last event is a behavioral correction (system notification)
    const mockHistory: HistoryEvent[] = [
      {
        type: "participant_utterance",
        id: crypto.randomUUID(),
        timestamp: Date.now() - 30000,
        text: "Greetings!",
        name: "user",
        isOwn: false,
      },
      {
        type: "own_utterance",
        id: crypto.randomUUID(),
        timestamp: Date.now() - 20000,
        text: "Hello! How can I help you?",
        isOwn: true,
      },
      // Injected system notification (correctional thought)
      ownThoughtTurn(
        "I should remain focused on my role as a bot builder assistant and help the administrator manage the 'Atlantis Attractions' bot, rather than outputting arbitrary titles like 'Israel - Sales Bot' which have no relevance to the conversation.",
      ),
    ];

    await agentDeps(mockHistory)(runAgentWithProvider)({
      maxIterations: 1,
      tools: [doNothingTool],
      prompt: "You are a helpful assistant.",
      rewriteHistory: () => Promise.resolve(),
      timezoneIANA: "UTC",
    });

    // Check if the agent outputted any new own_utterance (it should NOT have sent any message to the user!)
    const ownUtterancesAfterNotification = mockHistory.filter(
      (r: HistoryEvent) =>
        r.type === "own_utterance" &&
        r.timestamp > mockHistory[2].timestamp,
    );

    console.log("HISTORY AFTER RUNAGENT:");
    console.log(JSON.stringify(mockHistory, null, 2));

    const calledDoNothing = mockHistory.some(
      (r: HistoryEvent) =>
        r.type === "tool_call" && r.name === doNothingToolName,
    );

    assertEquals(
      ownUtterancesAfterNotification.length,
      0,
      `Agent should not send chat reply to system notification. Got: ${
        JSON.stringify(ownUtterancesAfterNotification)
      }`,
    );

    assert(
      calledDoNothing,
      "Agent should call do_nothing when receiving a system notification with no pending user message.",
    );
  },
  3,
  true,
);
