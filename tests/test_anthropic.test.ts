import { assert } from "@std/assert";
import { type HistoryEvent, participantUtteranceTurn } from "../src/agent.ts";
import {
  agentDeps,
  findTextualAnswer,
  injectSecrets,
  noopRewriteHistory,
  runWithProvider,
  withRetries,
} from "../test_helpers.ts";

Deno.test(
  "anthropic agent responds to a simple question",
  injectSecrets(withRetries(3, async () => {
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "What is 2 + 2?",
      }),
    ];

    await agentDeps(mockHistory)(runWithProvider("anthropic"))({
      maxIterations: 1,
      onMaxIterationsReached: () => {},
      tools: [],
      prompt: "You are a helpful assistant. Answer briefly.",
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });

    const answer = findTextualAnswer(mockHistory);
    assert(answer, "Expected an own_utterance response from Anthropic");
    assert(
      answer.text.includes("4"),
      `Expected answer to contain '4' but got: "${answer.text}"`,
    );
  })),
);
