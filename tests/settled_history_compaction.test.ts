import { assertEquals } from "@std/assert";
import { runAgent } from "../mod.ts";
import { agentDeps, noopRewriteHistory, someTool } from "../test_helpers.ts";
import {
  type HistoryEvent,
  injectCallModel,
  ownUtteranceTurn,
} from "../src/agent.ts";

Deno.test(
  "runAgent - triggers compaction for settled past sessions even when total history is well below 100k",
  async () => {
    const baseTime = Date.now() - 24 * 60 * 60 * 1000; // 24 hours ago
    const hourMs = 60 * 60 * 1000;
    const fillerText =
      "detailed discussion about previous task parameters and logs ".repeat(50); // ~500 tokens per message

    // Create 3 distinct past sessions separated by 1-hour gaps, totaling ~20k tokens
    const history: HistoryEvent[] = [];
    for (let session = 0; session < 3; session++) {
      const sessionTime = baseTime + session * 2 * hourMs;
      for (let turn = 0; turn < 6; turn++) {
        history.push({
          id: `p-${session}-${turn}`,
          type: "participant_utterance",
          isOwn: false,
          name: "user",
          text: `User message session ${session} turn ${turn}: ${fillerText}`,
          timestamp: sessionTime + turn * 1000,
        });
        history.push({
          id: `o-${session}-${turn}`,
          type: "own_utterance",
          isOwn: true,
          text:
            `Assistant response session ${session} turn ${turn}: ${fillerText}`,
          timestamp: sessionTime + turn * 1000 + 500,
        });
      }
    }

    // New active turn starting now
    const now = Date.now();
    history.push({
      id: "p-active",
      type: "participant_utterance",
      isOwn: false,
      name: "user",
      text: "What is the summary of our previous decisions?",
      timestamp: now,
    });

    let compactionInvokedCount = 0;

    const fakeCompactHistory = (
      _passedHistory: HistoryEvent[],
    ): Promise<void> => {
      compactionInvokedCount++;
      return Promise.resolve();
    };

    const fakeCallModel = (): Promise<HistoryEvent[]> => {
      return Promise.resolve([
        ownUtteranceTurn("Here is the summary based on past records."),
      ]);
    };

    await injectCallModel(fakeCallModel)(async () => {
      await agentDeps(history)(runAgent)({
        maxIterations: 5,
        tools: [someTool],
        prompt: "You are a helpful assistant.",
        rewriteHistory: noopRewriteHistory,
        compactHistory: fakeCompactHistory,
        // Notice: even if the threshold is left default or 100k, settled past sessions must trigger compaction
        historyCompactionTokenThreshold: 100_000,
        timezoneIANA: "UTC",
      });
    })();

    await new Promise((resolve) => setTimeout(resolve, 100));

    assertEquals(
      compactionInvokedCount > 0,
      true,
      "Expected compaction to be triggered for settled past sessions (~20k tokens) even when threshold is 100k",
    );
  },
);
