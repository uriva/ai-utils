import { assertEquals } from "@std/assert";
import { map, sum } from "gamla";
import {
  estimateTokensLocal,
  ownUtteranceTurn,
  participantUtteranceTurn,
} from "../../src/agent.ts";
import {
  partitionSegments,
  segmentHistoryEvents,
} from "../../src/compaction.ts";
import { appendEvalRecord } from "./eval_logger.ts";

const segmentGapMs = 30 * 60 * 1000;

const makeUtterance = (text: string, isOwn: boolean) =>
  isOwn
    ? ownUtteranceTurn(text)
    : participantUtteranceTurn({ name: "user", text });

Deno.test("eval compaction savings stay within budget", async () => {
  const start = Date.now();
  const longText = "weekly report discussion point ";
  const events = Array.from(
    { length: 80 },
    (_, i) => makeUtterance(longText.repeat(100), i % 2 === 0),
  );
  const totalTokens = sum(map(estimateTokensLocal)(events));
  const segments = segmentHistoryEvents(events, segmentGapMs);
  const { kept, toSummarize } = await partitionSegments(30000, segments);
  const keptTokens = sum(
    map(estimateTokensLocal)(kept.flatMap((s) => s.events)),
  );
  const pass = keptTokens <= 30000 &&
    kept.flatMap((s) => s.events).length +
          toSummarize.flatMap((s) => s.events).length === events.length;
  assertEquals(pass, true, `kept=${keptTokens} total=${totalTokens}`);
  await appendEvalRecord({
    suite: "compaction",
    task: "savings_within_budget",
    provider: "local",
    pass,
    turns: events.length,
    toolCalls: 0,
    historyTokens: totalTokens,
    timeMs: Date.now() - start,
  });
});
