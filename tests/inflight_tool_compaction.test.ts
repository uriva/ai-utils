import { assertEquals } from "@std/assert";
import { runAgent } from "../mod.ts";
import { agentDeps } from "../test_helpers.ts";
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
  "runAgent - in-flight multi-turn tool loop compacts older tool results to Memory TLDR before subsequent model calls",
  async () => {
    const history: HistoryEvent[] = [
      {
        id: "p-1",
        type: "participant_utterance",
        isOwn: false,
        name: "user",
        text: "Please inspect these 4 modules and generate a summary report.",
        timestamp: Date.now() - 10000,
      },
    ];

    const scratchStorage = new Map<string, string>();
    const scratchPad: ToolOutputScratchPad = {
      get: (id: string) => Promise.resolve(scratchStorage.get(id)),
      set: (id: string, content: string) => {
        scratchStorage.set(id, content);
        return Promise.resolve();
      },
    };

    let step = 0;
    const modelReceivedHistories: HistoryEvent[][] = [];

    const largeFileContent = (moduleName: string) =>
      `=== MODULE ${moduleName} SOURCE CODE ===\n` +
      `function process${moduleName}() {\n  // implementation details\n}\n`
        .repeat(100); // ~5,000 chars

    const inspectTool: Tool<z.ZodObject<{ module: z.ZodString }>> = {
      name: "inspect_module",
      description: "Inspect source code of a module",
      parameters: z.object({
        module: z.string().describe("Module name to inspect"),
      }),
      handler: ({ module }) => {
        return Promise.resolve(largeFileContent(module));
      },
    };

    const fakeCallModel = (
      receivedHistory: HistoryEvent[],
    ): Promise<HistoryEvent[]> => {
      modelReceivedHistories.push(JSON.parse(JSON.stringify(receivedHistory)));
      step++;
      if (step <= 4) {
        return Promise.resolve([
          {
            id: `call-${step}`,
            type: "tool_call",
            isOwn: true,
            name: "inspect_module",
            parameters: {
              module: ["Auth", "Database", "Payment", "Billing"][step - 1],
            },
            timestamp: Date.now(),
          },
        ]);
      }
      return Promise.resolve([
        ownUtteranceTurn("All 4 modules inspected successfully."),
      ]);
    };

    const historyReplacements: Record<string, HistoryEvent> = {};
    const rewriteHistory = (replacements: Record<string, HistoryEvent>) => {
      Object.assign(historyReplacements, replacements);
      for (const [id, event] of Object.entries(replacements)) {
        const idx = history.findIndex((e) => e.id === id);
        if (idx !== -1) {
          history[idx] = event;
        }
      }
      return Promise.resolve();
    };

    await pipe(
      injectCallModel(fakeCallModel),
      agentDeps(history),
    )(async () => {
      await runAgent({
        provider: "moonshot",
        maxIterations: 10,
        tools: [inspectTool],
        prompt: "You are a code inspection agent.",
        rewriteHistory,
        toolOutputScratchPad: scratchPad,
        timezoneIANA: "UTC",
      });
    })();

    // In history, the oldest tool result (from turn 1) must have been compacted to Memory TLDR
    const toolResults = history.filter((e) =>
      e.type === "tool_result"
    ) as Extract<
      HistoryEvent,
      { type: "tool_result" }
    >[];
    assertEquals(toolResults.length, 4, "Expected 4 tool results in history");

    const firstResult = toolResults[0];
    const isFirstCompacted = firstResult.result?.includes(
      "Because time has passed, this tool result has been compacted",
    ) || firstResult.result?.includes("Memory TLDR:");

    assertEquals(
      isFirstCompacted,
      true,
      "Expected oldest in-flight tool result to be compacted to Memory TLDR, but found:\n" +
        firstResult.result?.slice(0, 200),
    );

    // The most recent tool result (from turn 4) should remain in full fidelity
    const latestResult = toolResults[3];
    assertEquals(
      latestResult.result?.includes("=== MODULE Billing SOURCE CODE ==="),
      true,
      "Expected most recent tool result to remain full fidelity",
    );
  },
);
