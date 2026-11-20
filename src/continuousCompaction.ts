import {
  generateTimestamp,
  type HistoryEvent,
  learnSkillToolName,
  readScratchFileToolName,
  runCommandToolName,
  unlearnSkillToolName,
} from "./agent.ts";
import { groupToolCallPairs } from "./compaction.ts";
import {
  cleanActiveMemoryToolName,
  setThinkingLevelToolName,
} from "./utils.ts";

const minThreshold = 1500;
const maxThreshold = 15000;
const decayConstant = 0.09; // Threshold reaches ~5,000 characters at 15 minutes

export const getSpillThreshold = (
  timestamp: number,
  turnDistance = 0,
): number => {
  if (turnDistance >= 3) {
    return minThreshold;
  }
  const ageMs = generateTimestamp() - timestamp;
  const ageMins = ageMs / (60 * 1000);
  if (ageMins < 5) {
    return maxThreshold;
  }

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

const isObject = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null;

const toolCallCommandAndParams = (
  toolCall: HistoryEvent & { type: "tool_call" },
): { command: string; params: unknown } => {
  const p = toolCall.parameters;
  return toolCall.name === runCommandToolName && isObject(p) &&
      typeof p.command === "string"
    ? { command: p.command, params: p.params }
    : { command: toolCall.name, params: toolCall.parameters };
};

export const defaultDeterministicTLDR = (
  toolCall: HistoryEvent & { type: "tool_call" },
  resultText: string,
): string => {
  const lineCount = resultText.split("\n").length;
  const { command, params } = toolCallCommandAndParams(toolCall);
  const paramEntries = params && typeof params === "object"
    ? Object.entries(params)
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
  return `Command "${command}"${paramPart} completed.${outputPart} (${lineCount} lines, ${resultText.length} chars).`;
};

export type CompactionOptions = {
  setScratch: (id: string, content: string) => Promise<void>;
  generateTLDR?: (
    toolCall: HistoryEvent & { type: "tool_call" },
    resultText: string,
  ) => Promise<string>;
};

const maxSpilledIdsPerScratch = 2000;
const spilledIdsBySetter = new WeakMap<
  CompactionOptions["setScratch"],
  Set<string>
>();

const hasSpilledToSetter = (
  setScratch: CompactionOptions["setScratch"],
  id: string,
): boolean => spilledIdsBySetter.get(setScratch)?.has(id) ?? false;

const markSpilledToSetter = (
  setScratch: CompactionOptions["setScratch"],
  id: string,
): void => {
  let set = spilledIdsBySetter.get(setScratch);
  if (!set) {
    set = new Set<string>();
    spilledIdsBySetter.set(setScratch, set);
  }
  if (set.has(id)) set.delete(id);
  set.add(id);
  if (set.size > maxSpilledIdsPerScratch) {
    const oldest = set.keys().next().value;
    if (oldest !== undefined) set.delete(oldest);
  }
};

const unreadFreshWindowMs = 60_000;

const isModelActionEvent = (e: HistoryEvent): boolean =>
  e.type === "tool_call" ||
  e.type === "own_utterance" ||
  e.type === "own_edit_message";

const hasModelActedSinceResult = (
  events: HistoryEvent[],
  toolResult: HistoryEvent & { type: "tool_result" },
): boolean => {
  const idx = events.findIndex((e) => e.id === toolResult.id);
  return idx !== -1 &&
    events
      .slice(idx + 1)
      .some((e) =>
        e.timestamp >= toolResult.timestamp && isModelActionEvent(e)
      );
};

const isUnreadFreshToolResult = (
  events: HistoryEvent[],
  toolResult: HistoryEvent & { type: "tool_result" },
): boolean =>
  !hasModelActedSinceResult(events, toolResult) &&
  generateTimestamp() - toolResult.timestamp < unreadFreshWindowMs;

const isCompletedTurnEvent = (e: HistoryEvent): boolean =>
  e.type === "own_utterance" || e.type === "own_edit_message";

const countCompletedTurnsAfterResult = (
  events: HistoryEvent[],
  toolResult: HistoryEvent & { type: "tool_result" },
): number => {
  const idx = events.findIndex((e) => e.id === toolResult.id);
  return idx === -1 ? 0 : events
    .slice(idx + 1)
    .filter(
      (e) => e.timestamp >= toolResult.timestamp && isCompletedTurnEvent(e),
    ).length;
};

const isNonLifecycleToolPair = (pair: HistoryEvent[]): boolean => {
  const toolCall = pair.find((
    e,
  ): e is Extract<HistoryEvent, { type: "tool_call" }> =>
    e.type === "tool_call"
  );
  if (!toolCall) return false;
  const { command } = toolCallCommandAndParams(toolCall);
  return (
    command !== learnSkillToolName &&
    command !== unlearnSkillToolName &&
    command !== cleanActiveMemoryToolName &&
    command !== setThinkingLevelToolName
  );
};

export const compactedToolCallArgsMarker = "[Compacted argument";
const toolCallArgsScratchPrefix = "args-";
const maxInlineCompactedArgChars = 400;

export const isCompactedToolCallArgsPlaceholderString = (
  value: string,
): boolean =>
  value.includes(compactedToolCallArgsMarker) &&
  value.includes(`${readScratchFileToolName}("${toolCallArgsScratchPrefix}`);

export const containsCompactedToolCallArgsPlaceholder = (
  value: unknown,
): boolean => {
  if (typeof value === "string") {
    return isCompactedToolCallArgsPlaceholderString(value);
  }
  if (Array.isArray(value)) {
    return value.some(containsCompactedToolCallArgsPlaceholder);
  }
  if (isObject(value)) {
    return Object.values(value).some(containsCompactedToolCallArgsPlaceholder);
  }
  return false;
};

const formatCompactedArgPlaceholder = (
  charLength: number,
  scratchId: string,
): string =>
  `${compactedToolCallArgsMarker} (${charLength} chars — omitted from history; do NOT pass this placeholder into new tool calls; inspect original via ${readScratchFileToolName}("${scratchId}"))]`;

const compactArgValue = (
  value: unknown,
  scratchId: string,
): unknown => {
  if (
    value === null ||
    typeof value === "boolean" ||
    typeof value === "number"
  ) {
    return value;
  }
  if (typeof value === "string") {
    return value.length <= maxInlineCompactedArgChars
      ? value
      : formatCompactedArgPlaceholder(value.length, scratchId);
  }
  const serialized = JSON.stringify(value) ?? "";
  return serialized.length <= maxInlineCompactedArgChars
    ? value
    : formatCompactedArgPlaceholder(serialized.length, scratchId);
};

const compactParameterRecord = (
  record: Record<string, unknown>,
  scratchId: string,
  recurseParamsKey: boolean,
): Record<string, unknown> =>
  Object.fromEntries(
    Object.entries(record).map(([key, value]) => [
      key,
      recurseParamsKey && key === "params" && isObject(value)
        ? compactParameterRecord(value, scratchId, false)
        : compactArgValue(value, scratchId),
    ]),
  );

export const compactToolResultsInMemory = async (
  events: HistoryEvent[],
  { setScratch, generateTLDR }: CompactionOptions,
): Promise<HistoryEvent[]> => {
  const pairs = groupToolCallPairs(events).filter(isNonLifecycleToolPair);
  const resultCandidates: {
    toolCall: HistoryEvent & { type: "tool_call" };
    toolResult: HistoryEvent & { type: "tool_result" };
  }[] = [];
  const callArgsCandidates: {
    toolCall: HistoryEvent & { type: "tool_call" };
    serializedArgs: string;
  }[] = [];

  for (let i = 0; i < pairs.length; i++) {
    const pair = pairs[i];
    const toolStepDistance = pairs.length - 1 - i;
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

    if (
      toolCall &&
      toolResult &&
      toolResult.result &&
      !isUnreadFreshToolResult(events, toolResult)
    ) {
      const completedTurnsAfter = countCompletedTurnsAfterResult(
        events,
        toolResult,
      );
      const turnDistance = isSpillNotice(toolResult.result)
        ? toolStepDistance
        : completedTurnsAfter;
      const threshold = getSpillThreshold(
        toolResult.timestamp,
        turnDistance,
      );
      const { command } = toolCallCommandAndParams(toolCall);

      // If it exceeds decaying threshold and has not been folded yet
      if (
        command !== readScratchFileToolName &&
        command !== learnSkillToolName &&
        command !== unlearnSkillToolName &&
        toolResult.result.length > threshold &&
        !isCompactedToolResult(toolResult.result)
      ) {
        resultCandidates.push({ toolCall, toolResult });
      }

      if (
        command !== learnSkillToolName &&
        command !== unlearnSkillToolName &&
        isObject(toolCall.parameters)
      ) {
        const serializedArgs = JSON.stringify(toolCall.parameters);
        const argsThreshold = getSpillThreshold(
          toolCall.timestamp,
          Math.max(toolStepDistance, completedTurnsAfter),
        );
        if (
          serializedArgs.length > argsThreshold &&
          !serializedArgs.includes(compactedToolCallArgsMarker)
        ) {
          callArgsCandidates.push({ toolCall, serializedArgs });
        }
      }
    }
  }

  if (resultCandidates.length === 0 && callArgsCandidates.length === 0) {
    return events;
  }

  // Process all candidates in parallel using Promise.all
  const [resultReplacements, argReplacements] = await Promise.all([
    Promise.all(
      resultCandidates.map(async ({ toolCall, toolResult }) => {
        const originalResult = toolResult.result;
        const isAlreadySpilled = isSpillNotice(originalResult);
        const scratchId = isAlreadySpilled && toolResult.toolCallId
          ? toolResult.toolCallId
          : toolResult.id;

        // If not already spilled to scratchpad on turn 0, save original full text to scratchpad
        if (
          !isAlreadySpilled && !hasSpilledToSetter(setScratch, toolResult.id)
        ) {
          await setScratch(toolResult.id, originalResult);
          markSpilledToSetter(setScratch, toolResult.id);
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
    ),
    Promise.all(
      callArgsCandidates.map(async ({ toolCall, serializedArgs }) => {
        const scratchId = `${toolCallArgsScratchPrefix}${toolCall.id}`;
        if (!hasSpilledToSetter(setScratch, scratchId)) {
          await setScratch(scratchId, serializedArgs);
          markSpilledToSetter(setScratch, scratchId);
        }
        const compactedParams = isObject(toolCall.parameters)
          ? compactParameterRecord(
            toolCall.parameters,
            scratchId,
            toolCall.name === runCommandToolName,
          )
          : toolCall.parameters;
        return {
          id: toolCall.id,
          event: { ...toolCall, parameters: compactedParams },
        };
      }),
    ),
  ]);

  const replacements = new Map<string, HistoryEvent>();
  for (const r of [...resultReplacements, ...argReplacements]) {
    replacements.set(r.id, r.event);
  }

  return events.map((e) => replacements.get(e.id) ?? e);
};
