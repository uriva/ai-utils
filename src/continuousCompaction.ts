import { z } from "zod/v4";
import type { HistoryEvent } from "./agent.ts";
import { groupToolCallPairs } from "./compaction.ts";
import { genJson } from "./genJson.ts";

export const getSpillThreshold = (timestamp: number): number => {
  const ageMs = Date.now() - timestamp;
  const ageMins = ageMs / (60 * 1000);

  const minThreshold = 1500;
  const maxThreshold = 15000;
  const decayConstant = 0.09; // Threshold reaches ~5,000 characters at 15 minutes

  const decayFactor = Math.exp(-decayConstant * ageMins);
  return Math.round(minThreshold + (maxThreshold - minThreshold) * decayFactor);
};

const tldrSchema = z.object({
  tldr: z.string().describe(
    "A concise one-sentence technical summary of the command outcome (max 25 words).",
  ),
});

const makeTldrPrompt = (
  toolCall: HistoryEvent & { type: "tool_call" },
  resultText: string,
) => `
You are a technical logging utility.
Summarize the output of this technical command/tool call in exactly one concise sentence (maximum 25 words).
Focus strictly on what the tool was trying to achieve, and its final outcome, error status, or result.
Do not include any conversational preamble.

Tool Called:
Name: ${toolCall.name}
Parameters: ${JSON.stringify(toolCall.parameters, null, 2)}

Tool Output:
${resultText.slice(0, 8000)}
`;

export type CompactionOptions = {
  setScratch: (id: string, content: string) => Promise<void>;
  generateTLDR?: (
    toolCall: HistoryEvent & { type: "tool_call" },
    resultText: string,
  ) => Promise<string>;
};

export const runToolResultCompaction = async (
  events: HistoryEvent[],
  { setScratch, generateTLDR }: CompactionOptions,
  rewriteHistory: (replacements: Record<string, HistoryEvent>) => Promise<void>,
) => {
  const pairs = groupToolCallPairs(events);
  const candidates: {
    toolCall: HistoryEvent & { type: "tool_call" };
    toolResult: HistoryEvent & { type: "tool_result" };
  }[] = [];

  for (const pair of pairs) {
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
      const threshold = getSpillThreshold(toolResult.timestamp);

      // If it exceeds decaying threshold and has not been folded yet
      if (
        toolResult.result.length > threshold &&
        !toolResult.result.includes("read_scratch_file")
      ) {
        candidates.push({ toolCall, toolResult });
      }
    }
  }

  if (candidates.length === 0) return;

  // Process all candidates in parallel using Promise.all
  const replacementsList = await Promise.all(
    candidates.map(async ({ toolCall, toolResult }) => {
      const originalResult = toolResult.result;
      const lineCount = originalResult.split("\n").length;

      // Save original to scratchpad
      await setScratch(toolResult.id, originalResult);

      // Generate 1-sentence technical TLDR with call context
      let tldr = "Command completed.";
      try {
        if (generateTLDR) {
          tldr = await generateTLDR(toolCall, originalResult);
        } else {
          const res = await genJson(
            { provider: "google", mini: true },
            makeTldrPrompt(toolCall, originalResult),
            tldrSchema,
          )("");
          tldr = res.tldr;
        }
      } catch (_e) {
        // Fallback if TLDR generation fails
      }

      // Informative memory replacement text
      const replacementText = [
        `[Because time has passed, this tool result has been compacted to save space. This is what you remember from this execution:`,
        `Memory TLDR: ${tldr}`,
        `To refresh your memory on the full, raw output of this tool call, you can always read the scratchpad file by invoking \`read_scratch_file\` with the ID: "${toolResult.id}" (${lineCount} lines, ${originalResult.length} characters)]`,
      ].join("\n\n");

      return {
        id: toolResult.id,
        event: { ...toolResult, result: replacementText },
      };
    }),
  );

  const replacements: Record<string, HistoryEvent> = {};
  for (const r of replacementsList) {
    replacements[r.id] = r.event;
  }

  // Persist the changes back to Deno KV
  await rewriteHistory(replacements);
};
