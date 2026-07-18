import {
  empty,
  head,
  join,
  last,
  map,
  pipe,
  reverse,
  sortCompare,
} from "gamla";
import { z } from "zod/v4";
import { accessTokenCounter, type HistoryEvent } from "./agent.ts";
import { genJson } from "./genJson.ts";
import { formatInternalSentTimestamp } from "./internalMessageMetadata.ts";
import { cleanActiveMemoryToolName } from "./utils.ts";

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
    : e.type === "external_event"
    ? `EXTERNAL EVENT ${e.text}`
    : e.type === "own_thought"
    ? `OWN THOUGHT ${e.text}`
    : JSON.stringify(e);

const estimateSegmentTokens = async (
  { events }: HistorySegment,
): Promise<number> => await accessTokenCounter(events);

const makeSegmentFromEvents = (events: HistoryEvent[]): HistorySegment => ({
  events,
  start: head(events).timestamp,
  end: last(events).timestamp,
});

const trimSegmentToTokenBudget = async (
  maxTokens: number,
  { events }: HistorySegment,
) => {
  const sorted = sortEventsChronologically(events);
  const groups = groupToolCallPairs(sorted);
  const reversedGroups = [...groups].reverse();
  const groupTokensList = await Promise.all(
    reversedGroups.map((group) => accessTokenCounter(group)),
  );
  const kept: HistoryEvent[][] = [];
  let tokens = 0;
  for (let i = 0; i < reversedGroups.length; i++) {
    const group = reversedGroups[i];
    const groupTokens = groupTokensList[i];
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

// Fraction of the compaction TRIGGER threshold that history is compacted DOWN
// to when compaction fires. Using the trigger value itself as the retention
// budget pins history permanently at the ceiling: it grows a little, trips the
// trigger, gets trimmed right back to the ceiling, and every subsequent model
// call re-sends a near-ceiling history. Retaining a small fraction gives real
// headroom so compaction is infrequent and the per-call input stays small.
export const compactionRetentionRatio = 0.1;

export const compactionRetentionTokens = (triggerTokens: number): number =>
  Math.floor(triggerTokens * compactionRetentionRatio);

export const partitionSegments = async (
  maxTokens: number,
  segments: HistorySegment[],
): Promise<{ kept: HistorySegment[]; toSummarize: HistorySegment[] }> => {
  const segmentsNewestFirst = reverse(segments);
  const kept: HistorySegment[] = [];
  let used = 0;
  let newestOverflow: HistorySegment | null = null;

  for (let i = 0; i < segmentsNewestFirst.length; i++) {
    const seg = segmentsNewestFirst[i];
    const tks = await estimateSegmentTokens(seg);
    if (i === 0) {
      if (tks <= maxTokens) {
        kept.push(seg);
        used = tks;
      } else {
        const trimmed = await trimSegmentToTokenBudget(maxTokens, seg);
        kept.push(makeSegmentFromEvents(trimmed.kept));
        used = trimmed.keptTokens;
        if (!empty(trimmed.overflow)) {
          newestOverflow = makeSegmentFromEvents(trimmed.overflow);
        }
      }
    } else {
      if (used + tks > maxTokens) {
        break;
      } else {
        kept.push(seg);
        used += tks;
      }
    }
  }

  const keptChronological = reverse(kept);
  const keptIds = new Set(
    keptChronological.flatMap((s) => s.events.map((e) => e.id)),
  );
  const toSummarize = [
    ...segments.filter((s) => !s.events.some((e) => keptIds.has(e.id))),
    ...(newestOverflow ? [newestOverflow] : []),
  ];
  return { kept: keptChronological, toSummarize };
};

export const eventsToPlainText: (events: HistoryEvent[]) => string = pipe(
  map(eventToPlainText),
  join("\n\n"),
);

const structuredSummarySchema = z.object({
  entities: z.string().describe(
    "Key people, organizations, or named things mentioned. One line per entity. Max 50 words.",
  ),
  decisions: z.string().describe(
    "Agreements, choices, or conclusions reached during the conversation. Max 150 words.",
  ),
  actions: z.string().describe(
    "Actions taken via tools or by the assistant, with outcomes. Max 150 words.",
  ),
  pendingItems: z.string().describe(
    "Open questions, unresolved requests, or next steps the user expects. Max 150 words.",
  ),
  abandonedItems: z.string().describe(
    "Proposals, suggestions, or options that were raised but the user moved on from without confirming or rejecting. Include the specific name of the abandoned item and why it was not pursued. Max 150 words.",
  ),
  context: z.string().describe(
    "Any other important context needed to continue the conversation coherently. CRITICAL: Do NOT declare the dates/times of this historical segment as the 'current' simulated date, 'today', or 'now', as this summary will be read in the future where those dates are in the past. If you must describe dates, refer to them explicitly as the dates of the segment (e.g., 'The segment took place on May 5, 2026'). Max 150 words.",
  ),
  skillsToReLearn: z.string().describe(
    "List of active/used skills from the history that were learned (via the learn_skill tool) and are now compacted away, which the assistant must call learn_skill on immediately on the next turn to reload. If no skills were learned/active, write 'None'. Max 50 words.",
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
    await genJson(
      { provider: "google", mini: false },
      `Summarize the following conversation into structured sections. Write from the assistant's perspective. Be concise but preserve all important details, especially names, numbers, and specific facts that would be needed to continue the conversation.

Critical length constraints to prevent response truncation:
- Every section must be brief, dense, and highly compacted.
- Maximum 50 words for Key Entities and Active Skills to Re-Learn.
- Maximum 150 words per section for Decisions, Actions, Pending Items, Abandoned Items, and Context.

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
- Under skillsToReLearn, identify any skills that were actively learned or used in the history (look for learn_skill tool calls and results) and list them. If no skills were learned or used, write 'None'.

Critical Date and Time Grounding Rules:
- NEVER refer to the timestamps, dates, or days of this conversation segment as "the current simulated date", "the current date", "today", or "now" under Decisions, Context, or any other section.
- Because this summary is saved as a historical record and will be read in future conversation turns where this segment is in the past, absolute dates from the segment must always be described as past historical events (e.g., "The user inquired about events on Tuesday, May 5, 2026", instead of "Today is Tuesday, May 5, 2026").
- The assistant runs in real-time in the future, so framing a historical date as "today" or "now" will severely confuse the agent's temporal grounding on its subsequent turns.`,
      structuredSummarySchema,
    )(eventsToPlainText(events)),
  );

const formatSegmentRange = (start: number, end: number, timezone: string) => {
  const startStr = formatInternalSentTimestamp(start, timezone);
  const endStr = formatInternalSentTimestamp(end, timezone);
  return `[This summary covers the period from ${startStr} to ${endStr}]`;
};

export const summarizeSegmentToHistoryEvent = (timezone: string) =>
async (
  segment: HistorySegment,
): Promise<HistoryEvent & { type: "own_thought"; text: string }> => {
  const summaryText = await summarizeEvents(segment.events);
  const rangeHeader = formatSegmentRange(segment.start, segment.end, timezone);
  return {
    id: crypto.randomUUID(),
    type: "own_thought",
    text: `${rangeHeader}\n\n${summaryText}`,
    timestamp: Math.round((segment.start + segment.end) / 2),
    isOwn: true,
  };
};

export const cleanActiveMemoryToolRaw = (
  rewriteHistory: (replacements: Record<string, HistoryEvent>) => Promise<void>,
  getHistory: () => Promise<HistoryEvent[]>,
) => ({
  name: cleanActiveMemoryToolName,
  description:
    "Clean up your active conversation history by deleting or summarizing a specific time span. Excellent for compressing a series of failed trials or redundant logs into a single summary line.",
  parameters: z.object({
    start_time: z.string().describe(
      "The exact timestamp of the first event in the span to clean, e.g. 'Jun 21, 2026, 1:10 AM' or '2026-06-21T01:10:00Z'.",
    ),
    end_time: z.string().describe(
      "The exact timestamp of the last event in the span to clean, e.g. 'Jun 21, 2026, 1:15 AM' or '2026-06-21T01:15:00Z'.",
    ),
    summary: z.string().optional().describe(
      "Optional. A short 1-line description of what transpired during that time span to replace the verbose logs. If omitted, the entire time span is permanently deleted.",
    ),
  }),
  handler: async ({ start_time, end_time, summary }: {
    start_time: string;
    end_time: string;
    summary?: string;
  }) => {
    const start = Date.parse(start_time);
    const end = Date.parse(end_time);
    if (Number.isNaN(start) || Number.isNaN(end)) {
      return "Invalid date format provided for start_time or end_time.";
    }
    if (start > end) {
      return "start_time cannot be greater than end_time.";
    }

    const history = await getHistory();
    const match = history.filter(
      (e) => e.timestamp >= start && e.timestamp <= end,
    );
    if (match.length === 0) {
      return "No events found within the specified time range. Check your timestamps and try again.";
    }

    const hasUserUtterance = match.some(
      (e) => e.type === "participant_utterance",
    );
    if (hasUserUtterance) {
      return "Memory cleanup aborted: You cannot delete or summarize user messages (participant_utterance). Only select time spans containing tool results, tool calls, or thoughts.";
    }

    const replacements: Record<string, HistoryEvent> = {};
    const removedSkillNames: string[] = [];
    for (const e of match) {
      if (e.type === "tool_call" && e.name === "learn_skill") {
        // deno-lint-ignore no-explicit-any
        const skillName = (e.parameters as any)?.skillName;
        if (skillName) {
          removedSkillNames.push(skillName);
        }
      }
    }

    let resultMsg = "";
    if (summary) {
      const first = match[0];
      replacements[first.id] = {
        id: first.id,
        type: "own_thought",
        isOwn: true,
        text:
          `[SYSTEM SUMMARY of events from ${start_time} to ${end_time}]: ${summary}`,
        timestamp: first.timestamp,
      };
      for (let j = 1; j < match.length; j++) {
        const ev = match[j];
        replacements[ev.id] = {
          id: ev.id,
          type: "do_nothing",
          isOwn: true,
          timestamp: ev.timestamp,
        };
      }
      resultMsg =
        `Successfully summarized ${match.length} events from ${start_time} to ${end_time} with summary: "${summary}"`;
    } else {
      for (const ev of match) {
        replacements[ev.id] = {
          id: ev.id,
          type: "do_nothing",
          isOwn: true,
          timestamp: ev.timestamp,
        };
      }
      resultMsg =
        `Successfully deleted ${match.length} events from ${start_time} to ${end_time}.`;
    }

    if (removedSkillNames.length > 0) {
      const uniqueNames = [...new Set(removedSkillNames)];
      resultMsg +=
        `\n\n⚠️ NOTE: This memory cleanup has permanently removed the following active skills from your system prompt: ${
          uniqueNames.join(", ")
        }. If you still need to use any of these skills, you must call "learn_skill" on them again to reload their tools and instructions.`;
    }

    await rewriteHistory(replacements);
    return resultMsg;
  },
});
