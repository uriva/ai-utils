import { assert } from "@std/assert";
import { sum } from "gamla";
import { estimateTokensLocal, type HistoryEvent } from "../src/agent.ts";
import { partitionSegments, segmentHistoryEvents } from "../src/compaction.ts";
import { buildReq } from "../src/geminiAgent.ts";

// A Gemini thoughtSignature is an opaque base64 blob attached to model turns
// (own_utterance / tool_call / own_thought). It is re-sent to the model on every
// call (see geminiAgent.ts optionalThoughtSignature) and therefore counts as
// billed input tokens. Before the fix, estimateTokensLocal -> eventToPlainTextLocal
// only looked at text/parameters/result and never read modelMetadata, so the
// compaction threshold silently undercounted a bloated history and never fired.
//
// A realistic base64 signature (varied bytes, ~1300 chars ≈ ~325 tokens each).
const base64Alphabet =
  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const makeSignature = (seed: number) =>
  Array.from(
    { length: 1300 },
    (_, i) => base64Alphabet[(seed * 31 + i * 7) % base64Alphabet.length],
  ).join("");

// Token estimate over the full serialized payload (chars/4 is the standard rough
// token heuristic and is faithful for opaque base64).
const roughTokens = (text: string) => Math.ceil(text.length / 4);

const compactionThreshold = 100_000;
const segmentGapMs = 30 * 60 * 1000;

const signedTurn = (i: number): HistoryEvent[] => {
  const base = i * (segmentGapMs + 60_000);
  const metadata = {
    type: "gemini" as const,
    responseId: `resp-${i}`,
    thoughtSignature: makeSignature(i),
  };
  return [
    {
      id: `thought-${i}`,
      type: "own_thought",
      isOwn: true,
      timestamp: base,
      text: `Thinking about turn ${i}.`,
      modelMetadata: metadata,
    },
    {
      id: `utter-${i}`,
      type: "own_utterance",
      isOwn: true,
      timestamp: base + 1000,
      text: `Reply for turn ${i}.`,
      modelMetadata: metadata,
    },
  ];
};

Deno.test(
  "estimateTokensLocal counts thoughtSignature metadata so compaction fires on a bloated history",
  async () => {
    const history: HistoryEvent[] = Array.from(
      { length: 200 },
      (_, i) => signedTurn(i),
    ).flat();

    // Plain-text-only token count (thoughtSignatures excluded) — this is what
    // the pre-fix estimator returned. Tiny, and would have hidden the bloat.
    const textOnlyTokens = sum(
      history.map((e) =>
        roughTokens(
          e.type === "own_thought" || e.type === "own_utterance" ? e.text : "",
        )
      ),
    );

    // The estimator used by the compaction threshold check.
    const estimatedTokens = sum(history.map(estimateTokensLocal));

    // What the model actually receives: build the real Gemini request and count
    // tokens over the serialized contents (thoughtSignatures included).
    // deno-lint-ignore no-explicit-any
    const geminiHistory = history as any;
    const req = buildReq(false, "system prompt", [], "UTC", undefined)(
      geminiHistory,
    );
    const serializedTokens = roughTokens(JSON.stringify(req.contents));

    console.log(
      `textOnlyTokens=${textOnlyTokens} estimateTokensLocal=${estimatedTokens} serializedRequestTokens=${serializedTokens}`,
    );

    // The plain-text projection alone is far under the threshold (the old bug).
    assert(
      textOnlyTokens < compactionThreshold,
      `text-only tokens ${textOnlyTokens} should be under threshold (proving the metadata is what makes the payload big)`,
    );

    // The fixed estimator now reflects the true (bloated) payload that the model
    // actually receives, instead of the tiny text-only projection.
    assert(
      estimatedTokens > compactionThreshold,
      `expected estimate ${estimatedTokens} to exceed threshold ${compactionThreshold} after counting metadata`,
    );
    assert(
      serializedTokens > compactionThreshold,
      `sanity: the real request ${serializedTokens} is genuinely over threshold`,
    );
    assert(
      estimatedTokens > textOnlyTokens * 10,
      `estimate ${estimatedTokens} should be dominated by metadata, not text ${textOnlyTokens}`,
    );

    // ...and because the estimate now exceeds the threshold, compaction actually
    // summarizes old segments instead of silently doing nothing.
    const segments = segmentHistoryEvents(history, segmentGapMs);
    const { toSummarize } = await partitionSegments(
      compactionThreshold,
      segments,
    );
    assert(
      toSummarize.length > 0,
      `expected compaction to summarize old segments, but got ${toSummarize.length}`,
    );
  },
);
