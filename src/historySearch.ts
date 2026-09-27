import { empty, nonempty } from "gamla";
import { z } from "zod/v4";
import { compileGrepPattern, type HistoryEvent } from "./agent.ts";

export const searchPastHistoryToolName = "search_past_history";

export const searchPastHistoryParameters = z.object({
  query: z.string().optional().describe(
    "Optional keyword or regex pattern to search for across past messages, thoughts, or tool calls/results.",
  ),
  start_time: z.string().optional().describe(
    "Optional start timestamp (ISO string or parseable date like '2026-09-04' or 'Sep 4, 2026'). Only events at or after this time are returned.",
  ),
  end_time: z.string().optional().describe(
    "Optional end timestamp (ISO string or parseable date). Only events at or before this time are returned.",
  ),
  types: z.array(z.enum([
    "participant_utterance",
    "own_utterance",
    "tool_call",
    "tool_result",
    "own_thought",
  ])).optional().describe(
    "Optional list of event types to include. If omitted, all types are included.",
  ),
  order: z.enum(["asc", "desc"]).optional().default("asc").describe(
    "Sort order by timestamp: 'asc' (chronological, oldest first, default) or 'desc' (newest first).",
  ),
  limit: z.number().int().min(1).optional().default(20).describe(
    "Maximum number of matching events to return (default: 20, max: 50).",
  ),
  offset: z.number().int().nonnegative().optional().default(0).describe(
    "Pagination offset for matching events (default: 0).",
  ),
});

const maxEventTextDisplayChars = 1000;
const maxAllowedLimit = 50;

const eventToSearchableText = (e: HistoryEvent): string => {
  if (e.type === "participant_utterance" || e.type === "own_utterance") {
    return e.text;
  }
  if (e.type === "own_thought") {
    return e.text;
  }
  if (e.type === "tool_call") {
    return `${e.name} ${JSON.stringify(e.parameters ?? {})}`;
  }
  if (e.type === "tool_result") {
    return typeof e.result === "string"
      ? e.result
      : JSON.stringify(e.result ?? "");
  }
  if (e.type === "external_event") {
    return e.text;
  }
  return "";
};

const extractRelevantSnippet = (text: string, re?: RegExp): string => {
  if (text.length <= maxEventTextDisplayChars) return text;
  if (!re) {
    return `${
      text.slice(0, maxEventTextDisplayChars)
    }... [truncated, ${text.length} chars total]`;
  }
  const m = re.exec(text);
  if (!m) {
    return `${
      text.slice(0, maxEventTextDisplayChars)
    }... [truncated, ${text.length} chars total]`;
  }
  const matchIdx = m.index;
  const start = Math.max(
    0,
    matchIdx - Math.floor(maxEventTextDisplayChars / 2),
  );
  const end = Math.min(text.length, start + maxEventTextDisplayChars);
  const prefix = start > 0 ? "..." : "";
  const suffix = end < text.length
    ? `... [truncated, ${text.length} chars total]`
    : "";
  return `${prefix}${text.slice(start, end)}${suffix}`;
};

const formatEventForDisplay = (e: HistoryEvent, re?: RegExp): string => {
  const dateStr = new Date(e.timestamp).toISOString();
  const getPayload = (): { sender: string; rawBody: string } => {
    if (e.type === "participant_utterance") {
      return { sender: `User (${e.name ?? "User"})`, rawBody: e.text };
    }
    if (e.type === "own_utterance") {
      return { sender: "Assistant", rawBody: e.text };
    }
    if (e.type === "own_thought") {
      return { sender: "Thought / Summary", rawBody: e.text };
    }
    if (e.type === "tool_call") {
      return {
        sender: `Tool Call (${e.name})`,
        rawBody: JSON.stringify(e.parameters ?? {}),
      };
    }
    if (e.type === "tool_result") {
      return {
        sender: "Tool Result",
        rawBody: typeof e.result === "string"
          ? e.result
          : JSON.stringify(e.result ?? ""),
      };
    }
    return {
      sender: e.type,
      rawBody: "text" in e && typeof e.text === "string"
        ? e.text
        : JSON.stringify(e),
    };
  };

  const { sender, rawBody } = getPayload();
  const body = extractRelevantSnippet(rawBody, re);
  return `[${dateStr}] ${sender}:\n${body}`;
};

export type SearchPastHistoryArgs = {
  query?: string;
  start_time?: string;
  end_time?: string;
  types?: (
    | "participant_utterance"
    | "own_utterance"
    | "tool_call"
    | "tool_result"
    | "own_thought"
  )[];
  order?: "asc" | "desc";
  limit?: number;
  offset?: number;
};

export const searchPastHistoryToolRaw = (
  getHistory: () => Promise<HistoryEvent[]>,
) => ({
  name: searchPastHistoryToolName,
  description:
    "Search and inspect conversation history events from earlier in this conversation or past sessions. " +
    "Use this to find exact uncompacted tool outputs, codes, quotes, or details from earlier turns that were summarized, truncated, or scrolled out of context, or to search past messages by keyword or regex.",
  parameters: searchPastHistoryParameters,
  handler: async ({
    query,
    start_time,
    end_time,
    types,
    order = "asc",
    limit = 20,
    offset = 0,
  }: SearchPastHistoryArgs): Promise<string> => {
    const startMs = start_time ? Date.parse(start_time) : undefined;
    if (start_time && (startMs === undefined || Number.isNaN(startMs))) {
      return `Invalid date format for start_time: "${start_time}". Use ISO 8601 (e.g. "2026-09-04T12:00:00Z") or YYYY-MM-DD.`;
    }

    const endMs = end_time ? Date.parse(end_time) : undefined;
    if (end_time && (endMs === undefined || Number.isNaN(endMs))) {
      return `Invalid date format for end_time: "${end_time}". Use ISO 8601 (e.g. "2026-09-04T12:00:00Z") or YYYY-MM-DD.`;
    }

    if (startMs !== undefined && endMs !== undefined && startMs > endMs) {
      return "start_time cannot be greater than end_time.";
    }

    let compiledRe: RegExp | undefined;
    if (typeof query === "string" && query.trim().length > 0) {
      const compiled = compileGrepPattern(query);
      if (!compiled.ok) {
        return `Invalid search regex /${query}/: ${compiled.error}. Use valid JavaScript regex syntax.`;
      }
      compiledRe = compiled.re;
    }

    const history = await getHistory();
    if (empty(history)) {
      return "No past conversation events found.";
    }

    const typesSet = types && nonempty(types)
      ? new Set<string>(types)
      : undefined;

    const matches = history.filter((e) => {
      if (startMs !== undefined && e.timestamp < startMs) return false;
      if (endMs !== undefined && e.timestamp > endMs) return false;
      if (typesSet && !typesSet.has(e.type)) return false;
      if (compiledRe) {
        const text = eventToSearchableText(e);
        if (!compiledRe.test(text)) return false;
      }
      return true;
    });

    if (empty(matches)) {
      return "No past conversation events matched your search criteria.";
    }

    const sortedMatches = order === "desc"
      ? [...matches].sort((a, b) => b.timestamp - a.timestamp)
      : [...matches].sort((a, b) => a.timestamp - b.timestamp);

    const effectiveLimit = Math.min(limit, maxAllowedLimit);
    const paged = sortedMatches.slice(offset, offset + effectiveLimit);

    if (empty(paged)) {
      return `Offset ${offset} is beyond the total ${sortedMatches.length} matching events.`;
    }

    const formatted = paged.map((e) => formatEventForDisplay(e, compiledRe))
      .join("\n\n---\n\n");
    const remaining = sortedMatches.length - (offset + paged.length);
    const suffix = remaining > 0
      ? `\n\n[${remaining} more matching events available. Call again with offset=${
        offset + paged.length
      } to view the next page.]`
      : "";

    return `Found ${sortedMatches.length} matching events (showing ${paged.length} events from offset ${offset}):\n\n${formatted}${suffix}`;
  },
});
