import { sum } from "gamla";
import { estimateTokensLocal, type HistoryEvent } from "../../src/agent.ts";

export type EvalRecord = {
  suite: string;
  task: string;
  provider: string;
  pass: boolean;
  turns: number;
  toolCalls: number;
  historyTokens: number;
  timeMs: number;
};

export const summarizeHistory = (history: HistoryEvent[]) => ({
  turns: history.filter((e) => e.type === "tool_call").length,
  toolCalls: history.filter((e) => e.type === "tool_call").length,
  historyTokens: sum(history.map(estimateTokensLocal)),
});

export const currentProvider = (): string =>
  Deno.env.get("TEST_PROVIDER") ?? "unknown";

export const appendEvalRecord = async (record: EvalRecord): Promise<void> => {
  const line = `${JSON.stringify(record)}\n`;
  await Deno.writeTextFile("./tests/eval/results.jsonl", line, {
    append: true,
  });
};
