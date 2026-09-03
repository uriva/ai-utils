import {
  type HistoryEvent,
  learnSkillToolName,
  readScratchFileToolName,
  unlearnSkillToolName,
} from "./agent.ts";
import { groupToolCallPairs } from "./compaction.ts";

export const getSpillThreshold = (
  timestamp: number,
  turnDistance = 0,
): number => {
  if (turnDistance >= 3) {
    return 1500;
  }
  const ageMs = Date.now() - timestamp;
  const ageMins = ageMs / (60 * 1000);

  const minThreshold = 1500;
  const maxThreshold = 15000;
  const decayConstant = 0.09; // Threshold reaches ~5,000 characters at 15 minutes

  const decayFactor = Math.exp(-decayConstant * ageMins);
  return Math.round(minThreshold + (maxThreshold - minThreshold) * decayFactor);
};

const isCompactedToolResult = (text: string | undefined): boolean =>
  typeof text === "string" &&
  (text.includes(
    "Because time has passed, this tool result has been compacted",
  ) ||
    text.startsWith("[Because time has passed") ||
    text.includes("Memory TLDR:"));

const isSpillNotice = (text: string | undefined): boolean =>
  typeof text === "string" && text.includes("[Tool output was truncated");

export const defaultDeterministicTLDR = (
  toolCall: HistoryEvent & { type: "tool_call" },
  resultText: string,
): string => {
  const lineCount = resultText.split("\n").length;
  const paramEntries =
    toolCall.parameters && typeof toolCall.parameters === "object"
      ? Object.entries(toolCall.parameters)
      : [];
  const paramSummary = paramEntries.length > 0
    ? paramEntries
      .map(([k, v]) =>
        `${k}: ${typeof v === "object" ? JSON.stringify(v) : String(v)}`
      )
      .join(", ")
      .slice(0, 100)
    : "";
  const firstNonEmptyLine = resultText
    .split("\n")
    .map((l) => l.trim())
    .find((l) => l.length > 0)
    ?.slice(0, 120) ?? "";
  const paramPart = paramSummary ? ` (${paramSummary})` : "";
  const outputPart = firstNonEmptyLine ? ` Result: "${firstNonEmptyLine}"` : "";
  return `Command "${toolCall.name}"${paramPart} completed.${outputPart} (${lineCount} lines, ${resultText.length} chars).`;
};

export type CompactionOptions = {
  setScratch: (id: string, content: string) => Promise<void>;
  generateTLDR?: (
    toolCall: HistoryEvent & { type: "tool_call" },
    resultText: string,
  ) => Promise<string>;
};

export const compactToolResultsInMemory = async (
  events: HistoryEvent[],
  { setScratch, generateTLDR }: CompactionOptions,
): Promise<HistoryEvent[]> => {
  const pairs = groupToolCallPairs(events);
  const candidates: {
    toolCall: HistoryEvent & { type: "tool_call" };
    toolResult: HistoryEvent & { type: "tool_result" };
  }[] = [];

  for (let i = 0; i < pairs.length; i++) {
    const pair = pairs[i];
    const turnDistance = pairs.length - 1 - i;
    const toolCall = pair.find((
      e,
    ): e is Extract<HistoryEvent, { type: "tool_call" }> =>
      e.type === "tool_call"
    );
    const toolResult = pair.find((
      e,
    ): e is Extract<HistoryEvent, { type: "tool_result" }> =>
      e.type === "tool_result"
    );

    if (toolCall && toolResult && toolResult.result) {
      const threshold = getSpillThreshold(toolResult.timestamp, turnDistance);

      // If it exceeds decaying threshold and has not been folded yet
      if (
        toolCall.name !== readScratchFileToolName &&
        toolCall.name !== learnSkillToolName &&
        toolCall.name !== unlearnSkillToolName &&
        toolResult.result.length > threshold &&
        !isCompactedToolResult(toolResult.result)
      ) {
        candidates.push({ toolCall, toolResult });
      }
    }
  }

  if (candidates.length === 0) return events;

  // Process all candidates in parallel using Promise.all
  const replacementsList = await Promise.all(
    candidates.map(async ({ toolCall, toolResult }) => {
      const originalResult = toolResult.result;
      const isAlreadySpilled = isSpillNotice(originalResult);
      const scratchId = isAlreadySpilled && toolResult.toolCallId
        ? toolResult.toolCallId
        : toolResult.id;

      // If not already spilled to scratchpad on turn 0, save original full text to scratchpad
      if (!isAlreadySpilled) {
        await setScratch(toolResult.id, originalResult);
      }

      let tldr = defaultDeterministicTLDR(toolCall, originalResult);
      if (generateTLDR) {
        try {
          tldr = await generateTLDR(toolCall, originalResult);
        } catch (_e) {
          // Fallback to deterministic TLDR if custom generateTLDR fails
        }
      }

      const lineCount = originalResult.split("\n").length;

      // Informative memory replacement text
      const replacementText = [
        `[Because time has passed, this tool result has been compacted to save space. This is what you remember from this execution:`,
        `Memory TLDR: ${tldr}`,
        `To refresh your memory on the full, raw output of this tool call, you can always read the scratchpad file by invoking \`read_scratch_file\` with the ID: "${scratchId}" (${lineCount} lines, ${originalResult.length} characters)]`,
      ].join("\n\n");

      return {
        id: toolResult.id,
        event: { ...toolResult, result: replacementText },
      };
    }),
  );

  const replacements = new Map<string, HistoryEvent>();
  for (const r of replacementsList) {
    replacements.set(r.id, r.event);
  }

  return events.map((e) => replacements.get(e.id) ?? e);
};

export const runToolResultCompaction = async (
  events: HistoryEvent[],
  options: CompactionOptions,
  rewriteHistory?: (
    replacements: Record<string, HistoryEvent>,
  ) => Promise<void>,
): Promise<void> => {
  const compacted = await compactToolResultsInMemory(events, options);
  if (rewriteHistory && compacted !== events) {
    const replacements: Record<string, HistoryEvent> = {};
    for (const e of compacted) {
      const orig = events.find((origE) => origE.id === e.id);
      if (orig && orig !== e) {
        replacements[e.id] = e;
      }
    }
    if (Object.keys(replacements).length > 0) {
      await rewriteHistory(replacements);
    }
  }
};
