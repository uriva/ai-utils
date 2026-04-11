import { assert } from "@std/assert";
import { type HistoryEvent, participantUtteranceTurn } from "../src/agent.ts";
import {
  agentDeps,
  injectSecrets,
  noopRewriteHistory,
} from "../test_helpers.ts";
import { runAgent } from "../mod.ts";

Deno.test(
  "agent does not emit visible response when instructed to ignore irrelevant messages",
  injectSecrets(async () => {
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "User",
        text: "Hey what's up everyone?",
      }),
    ];
    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 3,
      onMaxIterationsReached: () => {},
      tools: [],
      prompt:
        `You are a registration bot for a soccer group. You ONLY respond to messages about signing up for games or canceling attendance. For any other message, use the do_nothing tool to signal you have nothing to say. Never write HTML comments or empty responses.`,
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
    const visibleResponses = mockHistory.filter(
      (e): e is Extract<HistoryEvent, { type: "own_utterance" }> =>
        e.type === "own_utterance" && e.text.trim() !== "",
    );
    assert(
      visibleResponses.length === 0,
      `Expected no visible own_utterance, but got: ${
        visibleResponses.map((e) => e.text.slice(0, 100)).join(" | ")
      }`,
    );
  }),
);
