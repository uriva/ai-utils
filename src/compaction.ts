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
import { z } from "zod/v4";
import { estimateTokens, type HistoryEvent } from "./agent.ts";
import { geminiGenJson } from "./gemini.ts";

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

const structuredSummarySchema = z.object({
  entities: z.string().describe(
    "Key people, organizations, or named things mentioned. One line per entity.",
  ),
  decisions: z.string().describe(
    "Agreements, choices, or conclusions reached during the conversation.",
  ),
  actions: z.string().describe(
    "Actions taken via tools or by the assistant, with outcomes.",
  ),
  pendingItems: z.string().describe(
    "Open questions, unresolved requests, or next steps the user expects.",
  ),
  abandonedItems: z.string().describe(
    "Proposals, suggestions, or options that were raised but the user moved on from without confirming or rejecting. Include the specific name of the abandoned item and why it was not pursued.",
  ),
  context: z.string().describe(
    "Any other important context needed to continue the conversation coherently.",
  ),
  skillsToReLearn: z.string().describe(
    "List of active/used skills from the history that were learned (via the learn_skill tool) and are now compacted away, which the assistant must call learn_skill on immediately on the next turn to reload. If no skills were learned/active, write 'None'.",
  ),
});

const formatStructuredSummary = ({
  entities,
  decisions,
  actions,
  pendingItems,
  abandonedItems,
  context,
  skillsToReLearn,
}: z.infer<typeof structuredSummarySchema>) => {
  const parts = [
    "Past conversation history was compacted into a structured summary.",
    "",
    "## Key Entities",
    entities,
    "",
    "## Decisions & Agreements",
    decisions,
    "",
    "## Actions Taken",
    actions,
    "",
    "## Pending Items",
    pendingItems,
    "",
    "## Abandoned Items",
    abandonedItems,
    "",
    "## Context",
    context,
  ];

  if (
    skillsToReLearn && skillsToReLearn.trim() &&
    skillsToReLearn.toLowerCase() !== "none"
  ) {
    parts.push(
      "",
      "## Active Skills to Re-Learn",
      "The following skills were active in the history but their instructions were compacted away. You MUST call learn_skill immediately for each of them on the next turn to recover your guidelines before taking any other action:",
      skillsToReLearn,
    );
  }

  return parts.join("\n");
};

export const summarizeEvents = async (
  events: HistoryEvent[],
): Promise<string> =>
  formatStructuredSummary(
    await geminiGenJson(
      { mini: false },
      `Summarize the following conversation into structured sections. Write from the assistant's perspective. Be concise but preserve all important details, especially names, numbers, and specific facts that would be needed to continue the conversation.

Critical anti-fabrication rules:
- Use ONLY information that is explicitly present in the source events. Never introduce specific proper-noun entities (hotel names, person names, document titles, restaurant or landmark names, brand names, etc.) that the source did not state.
- If the source refers to something generically (e.g. "the user's hotel", "the document she sent", "the second hotel near X", "the photography museum"), preserve that generic phrasing verbatim. Do NOT upgrade a generic reference into a specific name even if you can guess what it might be.
- Any section may be empty or contain only generic descriptions. Empty/generic is strictly better than invented specifics. An empty section is a correct answer when the source contained no concrete items for it.
- When in doubt about whether a name appeared in the source, leave it out.

Important rules for Pending Items vs Abandoned Items:
- If the user explicitly confirmed or rejected a proposal, note that in Decisions.
- If the user moved on to a different topic or chose an alternative WITHOUT explicitly confirming or rejecting a proposal, treat the original proposal as ABANDONED (put it in Abandoned Items, not Pending Items).
- Only put something in Pending Items if there is a clear open question, unresolved request, or next step the user still expects.
- Never keep a specific proposal as pending just because the user did not explicitly say no to it.

Important rule for Active Skills to Re-Learn:
- Under skillsToReLearn, identify any skills that were actively learned or used in the history (look for learn_skill tool calls and results) and list them. If no skills were learned or used, write 'None'.`,
      structuredSummarySchema,
    )(eventsToPlainText(events)),
  );
