import { assertEquals } from "@std/assert";
import { runAgent } from "../mod.ts";
import { agentDeps, noopRewriteHistory, someTool } from "../test_helpers.ts";
import {
  type HistoryEvent,
  injectCallModel,
  ownUtteranceTurn,
} from "../src/agent.ts";
import { pipe } from "gamla";

Deno.test(
  "runAgent - preserves internal thoughts from past turns in model context across user replies",
  async () => {
    const baseTime = Date.now() - 10000;

    // Turn 1: Completed turn with internal thoughts and an answer
    const history: HistoryEvent[] = [
      {
        id: "p-1",
        type: "participant_utterance",
        isOwn: false,
        name: "user",
        text: "What is the capital of France?",
        timestamp: baseTime,
      },
      {
        id: "t-1-1",
        type: "own_thought",
        isOwn: true,
        text:
          "Internal reasoning: User is asking for capital of France. Checking European capitals database.",
        timestamp: baseTime + 100,
      },
      {
        id: "t-1-2",
        type: "own_thought",
        isOwn: true,
        text: "Internal reasoning: Confirmed Paris is the capital of France.",
        timestamp: baseTime + 200,
      },
      {
        id: "o-1",
        type: "own_utterance",
        isOwn: true,
        text: "The capital of France is Paris.",
        timestamp: baseTime + 300,
      },
      // Turn 2: Another completed turn with internal thoughts and an answer
      {
        id: "p-2",
        type: "participant_utterance",
        isOwn: false,
        name: "user",
        text: "What is its population?",
        timestamp: baseTime + 1000,
      },
      {
        id: "t-2-1",
        type: "own_thought",
        isOwn: true,
        text:
          "Internal reasoning: Looking up population of Paris. Estimated ~2.1 million.",
        timestamp: baseTime + 1100,
      },
      {
        id: "t-stop",
        type: "own_thought",
        isOwn: true,
        text:
          "I'm working on this for some time and not making progress. I should instead stop and ask the user for feedback.",
        timestamp: baseTime + 1150,
      },
      {
        id: "o-2",
        type: "own_utterance",
        isOwn: true,
        text: "The population of Paris is approximately 2.1 million people.",
        timestamp: baseTime + 1200,
      },
      // Turn 3: New active turn starting now
      {
        id: "p-3",
        type: "participant_utterance",
        isOwn: false,
        name: "user",
        text: "What language is spoken there?",
        timestamp: baseTime + 5000,
      },
    ];

    let receivedHistoryForTurn3: HistoryEvent[] | null = null;

    const fakeCallModel = (
      received: HistoryEvent[],
    ): Promise<HistoryEvent[]> => {
      receivedHistoryForTurn3 = [...received];
      return Promise.resolve([
        ownUtteranceTurn("The official language spoken in France is French."),
      ]);
    };

    await pipe(
      injectCallModel(fakeCallModel),
      agentDeps(history),
    )(async () => {
      await runAgent({
        provider: "moonshot",
        maxIterations: 5,
        tools: [someTool],
        prompt: "You are a geography assistant.",
        rewriteHistory: noopRewriteHistory,
        timezoneIANA: "UTC",
      });
    })();

    assertEquals(
      receivedHistoryForTurn3 !== null,
      true,
      "Model should have been called",
    );

    const thoughtsInModelContext = receivedHistoryForTurn3!.filter(
      (e) => e.type === "own_thought",
    );

    assertEquals(
      thoughtsInModelContext.length,
      3,
      "Expected normal internal thoughts from past completed turns to be preserved in model context, but found: " +
        JSON.stringify(thoughtsInModelContext, null, 2),
    );

    assertEquals(
      thoughtsInModelContext.some((e) =>
        e.text.includes("not making progress")
      ),
      false,
      "Expected stale loop-breaker stop thought from prior turn to be pruned, but it leaked into new turn",
    );

    // Verify all user utterances and bot answers are still intact in context
    const participantUtterances = receivedHistoryForTurn3!.filter(
      (e) => e.type === "participant_utterance",
    );
    const ownUtterances = receivedHistoryForTurn3!.filter(
      (e) => e.type === "own_utterance",
    );

    assertEquals(
      participantUtterances.length,
      3,
      "Expected all 3 user messages in model context",
    );
    assertEquals(
      ownUtterances.length,
      2,
      "Expected both prior bot responses in model context",
    );
  },
);
