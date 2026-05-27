import { assert } from "@std/assert";
import { type HistoryEvent, participantUtteranceTurn } from "../src/agent.ts";
import {
  agentDeps,
  noopRewriteHistory,
  runForAllProviders,
  someTool,
} from "../test_helpers.ts";

const withCapturedConsoleLog = async <T>(
  fn: () => Promise<T>,
): Promise<{ result: T; lines: string[] }> => {
  const lines: string[] = [];
  const original = console.log;
  console.log = (...args: unknown[]) => {
    lines.push(
      args.map((a) => typeof a === "string" ? a : String(a)).join(" "),
    );
    original(...args);
  };
  const result = await fn().finally(() => {
    console.log = original;
  });
  return { result, lines };
};

runForAllProviders(
  "handleFunctionCalls logs [tool-call] with name and durationMs for each tool invocation",
  async (runAgentWithProvider) => {
    const mockHistory: HistoryEvent[] = [participantUtteranceTurn({
      name: "user",
      text:
        "Please call the doSomethingUnique tool now and only reply with its output.",
    })];
    const { lines } = await withCapturedConsoleLog(() =>
      agentDeps(mockHistory)(runAgentWithProvider)({
        maxIterations: 5,
        onMaxIterationsReached: () => {},
        tools: [someTool],
        prompt: "You are an AI assistant.",
        rewriteHistory: noopRewriteHistory,
        timezoneIANA: "UTC",
      })
    );
    const toolCallLogs = lines.filter((l) => l.includes("[tool-call]"));
    assert(
      toolCallLogs.some((l) =>
        l.includes("name=doSomethingUnique") && /durationMs=\d+/.test(l)
      ),
      `expected a [tool-call] log with name=doSomethingUnique and durationMs=<n>; got: ${
        JSON.stringify(toolCallLogs)
      }`,
    );
  },
);
