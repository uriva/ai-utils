import { assertEquals } from "@std/assert";
import type { HistoryEvent } from "../src/agent.ts";
import {
  externalEventTurn,
  ownEditMessageTurn,
  participantUtteranceTurn,
} from "../src/agent.ts";
import { segmentHistoryEvents } from "../src/compaction.ts";

type HistorySegmentRef = {
  events: HistoryEvent[];
  start: number;
  end: number;
};

const lastOf = <T>(xs: T[]): T => xs[xs.length - 1];

const isOwnUtterance = (e: HistoryEvent): boolean =>
  e.type === "own_utterance" || e.type === "own_edit_message";

const isParticipantUtterance = (e: HistoryEvent): boolean =>
  e.type === "participant_utterance" || e.type === "participant_edit_message";

// Reference implementation: the original O(n^2) algorithm that recomputed the
// whole accumulated segment's unanswered participants on every group. The
// incremental rewrite must produce identical segmentation.
const referenceSegmentHistoryEvents = (
  events: HistoryEvent[],
  gap: number,
): HistorySegmentRef[] => {
  if (events.length === 0) return [];
  const sorted = [...events].sort((a, b) => a.timestamp - b.timestamp);
  const groups = groupToolCallPairsReference(sorted);
  const segments: HistorySegmentRef[] = [];
  let currentGroups: HistoryEvent[][] = [groups[0]];
  for (let i = 1; i < groups.length; i++) {
    const prevEnd = lastOf(lastOf(currentGroups)).timestamp;
    const currStart = groups[i][0].timestamp;
    const gapOk = currStart - prevEnd >= gap;
    const currentEvents = currentGroups.flat();
    const hasUnanswered = nonemptyUnanswered(currentEvents);
    if (gapOk && !hasUnanswered && currentEvents.length >= 2) {
      segments.push({
        events: currentEvents,
        start: currentEvents[0].timestamp,
        end: lastOf(currentEvents).timestamp,
      });
      currentGroups = [groups[i]];
    } else {
      currentGroups.push(groups[i]);
    }
  }
  const remaining = currentGroups.flat();
  if (remaining.length > 0) {
    segments.push({
      events: remaining,
      start: remaining[0].timestamp,
      end: lastOf(remaining).timestamp,
    });
  }
  return segments;
};

const nonemptyUnanswered = (events: HistoryEvent[]): boolean => {
  const lastOwnIndex = events.findLastIndex(isOwnUtterance);
  return events.slice(lastOwnIndex + 1).some(isParticipantUtterance);
};

const groupToolCallPairsReference = (
  events: HistoryEvent[],
): HistoryEvent[][] => {
  const callIdToResults = new Map<string, HistoryEvent[]>();
  for (const e of events) {
    if (e.type === "tool_result" && e.toolCallId) {
      callIdToResults.set(e.toolCallId, [
        ...(callIdToResults.get(e.toolCallId) ?? []),
        e,
      ]);
    }
  }
  const matchedResultIds = new Set<string>();
  const result: HistoryEvent[][] = [];
  for (const e of events) {
    if (e.type === "tool_result" && matchedResultIds.has(e.id)) continue;
    if (e.type === "tool_call") {
      const results = callIdToResults.get(e.id) ?? [];
      if (results.length > 0) {
        result.push([e, ...results]);
        results.forEach((r) => matchedResultIds.add(r.id));
      } else result.push([e]);
    } else result.push([e]);
  }
  return result;
};

// Deterministic pseudo-random event streams mixing gap-straddling timestamps
// with unanswered participants right before gaps — exactly where an
// incremental rewrite tends to diverge from the reference.
let seed = 42;
const rand = () => {
  seed = (seed * 1103515245 + 12345) % 2147483648;
  return seed / 2147483648;
};

Deno.test("incremental segmentation matches the reference algorithm", () => {
  for (let trial = 0; trial < 200; trial++) {
    const events: HistoryEvent[] = [];
    let t = 1000;
    for (let i = 0; i < 40; i++) {
      t += Math.floor(rand() * 40 * 60 * 1000); // up to ~2 gaps worth
      const roll = rand();
      if (roll < 0.25) {
        events.push(participantUtteranceTurn({
          name: "user",
          text: `msg ${trial}-${i}`,
        }));
      } else if (roll < 0.45) {
        events.push(externalEventTurn(`event ${trial}-${i}`));
      } else if (roll < 0.65) {
        events.push(ownEditMessageTurn({
          text: `edit ${trial}-${i}`,
          onMessage: "m",
        }));
      } else {
        events.push({
          type: "own_utterance",
          isOwn: true,
          text: `hi ${trial}-${i}`,
          id: `own-${trial}-${i}`,
          timestamp: t,
        });
      }
      events[events.length - 1].timestamp = t;
    }
    const actual = JSON.stringify(segmentHistoryEvents(events, 30 * 60 * 1000));
    const expected = JSON.stringify(
      referenceSegmentHistoryEvents(events, 30 * 60 * 1000),
    );
    assertEquals(actual, expected, `divergence in trial ${trial}`);
  }
});
