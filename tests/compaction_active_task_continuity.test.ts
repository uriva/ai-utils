import { assertEquals } from "@std/assert";
import { runAgent } from "../mod.ts";
import { agentDeps, noopRewriteHistory, someTool } from "../test_helpers.ts";
import {
  doNothingEvent,
  type HistoryEvent,
  injectCallModel,
  ownThoughtTurn,
  ownUtteranceTurn,
  participantUtteranceTurn,
} from "../src/agent.ts";

Deno.test(
  "runAgent - maintains task continuity across compaction when unanswered user prompt is preserved",
  async () => {
    // History after compaction: an own_thought summary followed by the preserved unanswered user prompt
    // and intermediate tool calls/results that ran before compaction.
    const summaryText =
      "Past conversation history was compacted into a structured summary.\n\n" +
      "## Key Entities\nUser, Database Service\n\n" +
      "## Decisions & Agreements\nAgreed to run database migration and verification.\n\n" +
      "## Actions Taken\nExecuted initial migration step.\n\n" +
      "## Pending Items\nComplete verification and finalize migration.";

    const history: HistoryEvent[] = [
      ownThoughtTurn(summaryText),
      participantUtteranceTurn({
        name: "user",
        text:
          "Please run the migration script, verify the results, and update me when done.",
      }),
      {
        id: "call-1",
        type: "tool_call",
        isOwn: true,
        name: "test_tool",
        parameters: { command: "run_step_1" },
        timestamp: Date.now() - 5000,
      },
      {
        id: "result-1",
        type: "tool_result",
        isOwn: true,
        toolCallId: "call-1",
        result: "Step 1 completed successfully",
        timestamp: Date.now() - 4000,
      },
    ];

    let turnCount = 0;
    const fakeCallModel = (): Promise<HistoryEvent[]> => {
      turnCount++;
      if (turnCount === 1) {
        // If the model initially considers do_nothing, the unanswered user gate should intercept it
        // and guide it to complete the task or reply to the user.
        return Promise.resolve([doNothingEvent()]);
      }
      return Promise.resolve([
        ownUtteranceTurn("Migration and verification completed successfully!"),
      ]);
    };

    await injectCallModel(fakeCallModel)(async () => {
      await agentDeps(history)(runAgent)({
        maxIterations: 10,
        tools: [someTool],
        prompt: "You are an automated maintenance assistant.",
        rewriteHistory: noopRewriteHistory,
        timezoneIANA: "UTC",
      });
    })();

    const ownUtterances = history.filter((e) => e.type === "own_utterance");
    assertEquals(
      ownUtterances.length,
      1,
      "Expected agent to send a completion utterance to the user instead of stalling",
    );
    assertEquals(
      ownUtterances[0].text,
      "Migration and verification completed successfully!",
    );
  },
);
