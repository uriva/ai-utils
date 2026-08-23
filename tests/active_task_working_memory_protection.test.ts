import { assertEquals } from "@std/assert";
import { runAgent } from "../mod.ts";
import { agentDeps, noopRewriteHistory } from "../test_helpers.ts";
import { pipe } from "gamla";
import { z } from "zod/v4";
import {
  type HistoryEvent,
  injectCallModel,
  ownUtteranceTurn,
  type Tool,
  type ToolOutputScratchPad,
} from "../src/agent.ts";

Deno.test(
  "runAgent - protects active coding task working memory while past settled sessions are compacted",
  async () => {
    const pastTime = Date.now() - 2 * 60 * 60 * 1000; // 2 hours ago
    const fillerText = "past conversation details and discussion logs ".repeat(
      40,
    ); // ~400 tokens

    // Past completed session
    const history: HistoryEvent[] = [
      {
        id: "p-old-1",
        type: "participant_utterance",
        isOwn: false,
        name: "user",
        text: `Previous discussion: ${fillerText}`,
        timestamp: pastTime,
      },
      {
        id: "o-old-1",
        type: "own_utterance",
        isOwn: true,
        text: `Previous answer: ${fillerText}`,
        timestamp: pastTime + 500,
      },
    ];

    // Current active coding task starting now
    const now = Date.now();
    history.push({
      id: "p-active",
      type: "participant_utterance",
      isOwn: false,
      name: "user",
      text:
        "Refactor auth.ts and session.ts, then run the test suite to verify.",
      timestamp: now,
    });

    const scratchStorage = new Map<string, string>();
    const scratchPad: ToolOutputScratchPad = {
      get: (id: string) => Promise.resolve(scratchStorage.get(id)),
      set: (id: string, content: string) => {
        scratchStorage.set(id, content);
        return Promise.resolve();
      },
    };

    const files: Record<string, string> = {
      "src/auth.ts":
        "export const authenticate = (token: string) => token.startsWith('Bearer ');\n"
          .repeat(30),
      "src/session.ts":
        "export const createSession = (userId: string) => ({ id: userId, expires: Date.now() + 3600 });\n"
          .repeat(30),
    };

    const readFileTool: Tool<z.ZodObject<{ path: z.ZodString }>> = {
      name: "read_file",
      description: "Read file content",
      parameters: z.object({ path: z.string() }),
      handler: ({ path }) => Promise.resolve(files[path] || "file not found"),
    };

    const runTestsTool: Tool<z.ZodObject<{ command: z.ZodString }>> = {
      name: "run_tests",
      description: "Run automated tests",
      parameters: z.object({ command: z.string() }),
      handler: ({ command }) =>
        Promise.resolve(`Tests passed successfully for ${command}`),
    };

    let step = 0;
    const modelReceivedHistories: HistoryEvent[][] = [];

    const fakeCallModel = (
      received: HistoryEvent[],
    ): Promise<HistoryEvent[]> => {
      modelReceivedHistories.push(JSON.parse(JSON.stringify(received)));
      step++;
      if (step === 1) {
        return Promise.resolve([
          {
            id: "call-1",
            type: "tool_call",
            isOwn: true,
            name: "read_file",
            parameters: { path: "src/auth.ts" },
            timestamp: Date.now(),
          },
        ]);
      }
      if (step === 2) {
        return Promise.resolve([
          {
            id: "call-2",
            type: "tool_call",
            isOwn: true,
            name: "read_file",
            parameters: { path: "src/session.ts" },
            timestamp: Date.now(),
          },
        ]);
      }
      if (step === 3) {
        return Promise.resolve([
          {
            id: "call-3",
            type: "tool_call",
            isOwn: true,
            name: "run_tests",
            parameters: { command: "deno test tests/auth.test.ts" },
            timestamp: Date.now(),
          },
        ]);
      }
      return Promise.resolve([
        ownUtteranceTurn("Refactoring completed and all tests pass!"),
      ]);
    };

    let _compactionInvoked = false;
    const fakeCompactHistory = (_h: HistoryEvent[]): Promise<void> => {
      _compactionInvoked = true;
      return Promise.resolve();
    };

    await pipe(
      injectCallModel(fakeCallModel),
      agentDeps(history),
    )(async () => {
      await runAgent({
        provider: "moonshot",
        maxIterations: 10,
        tools: [readFileTool, runTestsTool],
        prompt: "You are an automated software engineer.",
        rewriteHistory: noopRewriteHistory,
        compactHistory: fakeCompactHistory,
        toolOutputScratchPad: scratchPad,
        historyCompactionTokenThreshold: 64_000,
        timezoneIANA: "UTC",
      });
    })();

    assertEquals(modelReceivedHistories.length, 4, "Expected 4 agent steps");

    // Verify that the active coding prompt was preserved across all turns
    for (let s = 0; s < modelReceivedHistories.length; s++) {
      const h = modelReceivedHistories[s];
      const hasActivePrompt = h.some(
        (e) =>
          e.type === "participant_utterance" &&
          e.text.includes("Refactor auth.ts and session.ts"),
      );
      assertEquals(
        hasActivePrompt,
        true,
        `Expected active coding user prompt to be preserved in model context on step ${
          s + 1
        }`,
      );
    }

    // Verify the latest model step had access to recent test output
    const fourthStepHistory = modelReceivedHistories[3];
    const hasTestResult = fourthStepHistory.some(
      (e) =>
        e.type === "tool_result" &&
        e.result?.includes("Tests passed successfully"),
    );
    assertEquals(
      hasTestResult,
      true,
      "Expected step 4 to have immediate access to the test execution output",
    );

    // Verify final reply was emitted
    const finalUtterances = history.filter((e) => e.type === "own_utterance");
    assertEquals(
      finalUtterances.some((u) => u.text.includes("Refactoring completed")),
      true,
      "Expected final success reply in history",
    );
  },
);
