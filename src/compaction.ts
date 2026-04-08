import {
  empty,
  head,
  join,
  last,
  map,
  pipe,
  reverse,
  sortCompare,
  sum,
} from "gamla";
import { estimateTokens, type HistoryEvent } from "./agent.ts";

export type HistorySegment = {
  events: HistoryEvent[];
  start: number;
  end: number;
};

const sortEventsChronologically = sortCompare(
  (a: HistoryEvent, b: HistoryEvent) => a.timestamp - b.timestamp,
);

const isToolCall = (
  e: HistoryEvent,
): e is Extract<HistoryEvent, { type: "tool_call" }> => e.type === "tool_call";

const isToolResult = (
  e: HistoryEvent,
): e is Extract<HistoryEvent, { type: "tool_result" }> =>
  e.type === "tool_result";

export const groupToolCallPairs = (
  events: HistoryEvent[],
): HistoryEvent[][] => {
  // Build a map from tool_call id to its matching tool_results
  const callIdToResults = new Map<string, HistoryEvent[]>();
  const matchedResultIds = new Set<string>();
  for (const e of events) {
    if (isToolResult(e) && e.toolCallId) {
      const existing = callIdToResults.get(e.toolCallId);
      if (existing) {
        existing.push(e);
      } else {
        callIdToResults.set(e.toolCallId, [e]);
      }
    }
  }
  // Walk events in order, emitting groups
  const result: HistoryEvent[][] = [];
  for (const e of events) {
    if (matchedResultIds.has(e.id)) continue;
    if (isToolCall(e)) {
      const results = callIdToResults.get(e.id);
      if (results && results.length > 0) {
        result.push([e, ...results]);
        for (const r of results) matchedResultIds.add(r.id);
      } else {
        result.push([e]);
      }
    } else {
      result.push([e]);
    }
  }
  return result;
};

export const segmentHistoryEvents = (
  events: HistoryEvent[],
  gap: number,
): HistorySegment[] => {
  if (empty(events)) return [];
  const sorted = sortEventsChronologically(events);
  const groups = groupToolCallPairs(sorted);
  const segments: HistorySegment[] = [];
  let currentGroups: HistoryEvent[][] = [groups[0]];

  const groupTimestamp = (g: HistoryEvent[]): number => head(g).timestamp;
  const groupEndTimestamp = (g: HistoryEvent[]): number => last(g).timestamp;
  const flattenGroups = (gs: HistoryEvent[][]): HistoryEvent[] =>
    gs.flatMap((g) => g);

  for (let i = 1; i < groups.length; i++) {
    const prevEnd = groupEndTimestamp(last(currentGroups));
    const currStart = groupTimestamp(groups[i]);
    const gapOk = currStart - prevEnd >= gap;
    const currentEvents = flattenGroups(currentGroups);
    if (gapOk && currentEvents.length >= 2) {
      segments.push({
        events: currentEvents,
        start: head(currentEvents).timestamp,
        end: last(currentEvents).timestamp,
      });
      currentGroups = [groups[i]];
    } else {
      currentGroups.push(groups[i]);
    }
  }
  const remaining = flattenGroups(currentGroups);
  if (remaining.length > 0) {
    segments.push({
      events: remaining,
      start: head(remaining).timestamp,
      end: last(remaining).timestamp,
    });
  }
  return segments;
};

export const eventToPlainText = (e: HistoryEvent): string =>
  e.type === "participant_utterance" || e.type === "own_utterance"
    ? e.text
    : e.type === "tool_call"
    ? `TOOL CALL ${e.name} ${JSON.stringify(e.parameters)}`
    : e.type === "tool_result"
    ? `TOOL RESULT ${JSON.stringify(e.result)}`
    : JSON.stringify(e);

const estimateSegmentTokens = ({ events }: HistorySegment) =>
  sum(map(estimateTokens)(events));

const makeSegmentFromEvents = (events: HistoryEvent[]): HistorySegment => ({
  events,
  start: head(events).timestamp,
  end: last(events).timestamp,
});

const trimSegmentToTokenBudget = (
  maxTokens: number,
  { events }: HistorySegment,
) => {
  const sorted = sortEventsChronologically(events);
  const groups = groupToolCallPairs(sorted);
  const reversedGroups = [...groups].reverse();
  const kept: HistoryEvent[][] = [];
  let tokens = 0;
  for (const group of reversedGroups) {
    const groupTokens = sum(map(estimateTokens)(group));
    if (tokens + groupTokens > maxTokens && kept.length > 0) break;
    kept.unshift(group);
    tokens += groupTokens;
  }
  const keptEvents = kept.flatMap((g) => g);
  const keptIds = new Set(keptEvents.map((e) => e.id));
  return {
    kept: keptEvents,
    overflow: sorted.filter((e) => !keptIds.has(e.id)),
    keptTokens: tokens,
  };
};

export const partitionSegments = (
  maxTokens: number,
  segments: HistorySegment[],
): { kept: HistorySegment[]; toSummarize: HistorySegment[] } => {
  const segmentsNewestFirst = reverse(segments);
  const [kept, , newestOverflow] = segmentsNewestFirst.reduce<
    [HistorySegment[], number, HistorySegment | null]
  >(
    ([acc, used, overflow], seg, i) => {
      const tks = estimateSegmentTokens(seg);
      if (i === 0) {
        if (tks <= maxTokens) return [[seg, ...acc], tks, null];
        const trimmed = trimSegmentToTokenBudget(maxTokens, seg);
        return [
          [makeSegmentFromEvents(trimmed.kept), ...acc],
          trimmed.keptTokens,
          empty(trimmed.overflow)
            ? null
            : makeSegmentFromEvents(trimmed.overflow),
        ];
      }
      return used + tks > maxTokens
        ? [acc, used, overflow]
        : [[...acc, seg], used + tks, overflow];
    },
    [[], 0, null],
  );
  const keptIds = new Set(kept.flatMap((s) => s.events.map((e) => e.id)));
  const toSummarize = [
    ...segments.filter((s) => !s.events.some((e) => keptIds.has(e.id))),
    ...(newestOverflow ? [newestOverflow] : []),
  ];
  return { kept, toSummarize };
};

export const eventsToPlainText: (events: HistoryEvent[]) => string = pipe(
  map(eventToPlainText),
  join("\n\n"),
);
