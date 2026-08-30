import { assertEquals, assertStringIncludes } from "@std/assert";
import { z } from "zod/v4";
import { runAgent } from "../mod.ts";
import {
  type HistoryEvent,
  injectCallModel,
  ownUtteranceTurn,
  participantUtteranceTurn,
  readScratchFileToolName,
  type Tool,
  type ToolOutputScratchPad,
} from "../src/agent.ts";
import { compactToolResultsInMemory } from "../src/continuousCompaction.ts";
import { agentDeps, noopRewriteHistory } from "../test_helpers.ts";
import { pipe } from "gamla";

Deno.test(
  "compactToolResultsInMemory never compacts read_scratch_file and preserves full scratchpad content for already-spilled results",
  async () => {
    const scratchStore = new Map<string, string>();
    const scratchPad: ToolOutputScratchPad = {
      get: (id: string) => Promise.resolve(scratchStore.get(id)),
      set: (id: string, content: string) => {
        scratchStore.set(id, content);
        return Promise.resolve();
      },
    };

    const initialFullContent = "Item details line\n".repeat(200); // ~3600 chars
    scratchStore.set("spilled-tool-call-1", initialFullContent);

    const history: HistoryEvent[] = [
      {
        id: "p-1",
        type: "participant_utterance",
        isOwn: false,
        name: "user",
        text: "Search items",
        timestamp: Date.now() - 6000,
      },
      // 1. A tool call that was already spilled on turn 0 (has preview and spill notice)
      {
        id: "spilled-tool-call-1",
        type: "tool_call",
        name: "search_items",
        parameters: { query: "all" },
        isOwn: true,
        timestamp: Date.now() - 5000,
      },
      {
        id: "spilled-result-1",
        type: "tool_result",
        toolCallId: "spilled-tool-call-1",
        result: "Item details line\n".repeat(100) +
          "\n\n[Tool output was truncated (3600 chars, 200 lines total). If you need more of the content or want to search through it, you can call read_scratch_file({id: \"spilled-tool-call-1\", startLine: 101}) or use its 'grep' parameter.]",
        isOwn: true,
        timestamp: Date.now() - 4500,
      },
      // 2. A read_scratch_file call retrieving lines (turnDistance will be 2)
      {
        id: "read-call-1",
        type: "tool_call",
        name: readScratchFileToolName,
        parameters: { id: "spilled-tool-call-1", startLine: 1, numLines: 100 },
        isOwn: true,
        timestamp: Date.now() - 4000,
      },
      {
        id: "read-result-1",
        type: "tool_result",
        toolCallId: "read-call-1",
        result:
          `[Scratch pad "spilled-tool-call-1": 200 lines, 3600 chars total.]\n` +
          "Item details line\n".repeat(100) +
          "\n[Showing lines 1-100 of 200. Call again with startLine=101 to continue.]",
        isOwn: true,
        timestamp: Date.now() - 3500,
      },
      // 3. A subsequent tool call
      {
        id: "call-3",
        type: "tool_call",
        name: "check_status",
        parameters: {},
        isOwn: true,
        timestamp: Date.now() - 3000,
      },
      {
        id: "res-3",
        type: "tool_result",
        toolCallId: "call-3",
        result: "Status: OK",
        isOwn: true,
        timestamp: Date.now() - 2500,
      },
      // 4. Another subsequent tool call (so read-call-1 has turnDistance = 4 - 1 - 1 = 2)
      {
        id: "call-4",
        type: "tool_call",
        name: "verify_item",
        parameters: { id: "item-1" },
        isOwn: true,
        timestamp: Date.now() - 2000,
      },
      {
        id: "res-4",
        type: "tool_result",
        toolCallId: "call-4",
        result: "Item verified: OK",
        isOwn: true,
        timestamp: Date.now() - 1500,
      },
    ];

    const projected = await compactToolResultsInMemory(history, {
      setScratch: (id, content) => scratchPad.set(id, content),
      generateTLDR: () => Promise.resolve("Inspected items."),
    });

    // 1. read_scratch_file output must NOT be compacted into a new Memory TLDR or new scratchpad pointer
    const projectedReadResult = projected.find((e) => e.id === "read-result-1");
    assertEquals(
      projectedReadResult && "result" in projectedReadResult &&
        typeof projectedReadResult.result === "string" &&
        projectedReadResult.result.includes(
          `[Scratch pad "spilled-tool-call-1": 200 lines, 3600 chars total.]`,
        ),
      true,
      "read_scratch_file result must remain uncompacted so the agent retains working memory",
    );

    // 2. Already-spilled tool result must NOT overwrite the original full scratchpad content with truncated preview
    assertEquals(
      scratchStore.get("spilled-tool-call-1"),
      initialFullContent,
      "Full original content in scratchpad must not be overwritten by truncated preview",
    );

    // 3. Compacted already-spilled tool result must point to the original scratchId (spilled-tool-call-1)
    const projectedSpilledResult = projected.find((e) =>
      e.id === "spilled-result-1"
    );
    assertStringIncludes(
      (projectedSpilledResult as Extract<HistoryEvent, { type: "tool_result" }>)
        .result,
      'read_scratch_file` with the ID: "spilled-tool-call-1"',
    );
  },
);

Deno.test(
  "runAgent - multi-step scratchpad reading preserves working memory without entering recursive scratchpad loops",
  async () => {
    const scratchStore = new Map<string, string>();
    const scratchPad: ToolOutputScratchPad = {
      get: (id: string) => Promise.resolve(scratchStore.get(id)),
      set: (id: string, content: string) => {
        scratchStore.set(id, content);
        return Promise.resolve();
      },
      threshold: 1000,
    };

    const bigOutput = Array.from(
      { length: 150 },
      (_, i) => `Event ${i + 1}: Concert at Hall ${i + 1}`,
    ).join("\n"); // ~4500 chars

    const searchTool: Tool<z.ZodObject<{ location: z.ZodString }>> = {
      name: "search_events",
      description: "Search events by location",
      parameters: z.object({ location: z.string() }),
      handler: () => Promise.resolve(bigOutput),
    };

    const dummyTool: Tool<z.ZodObject<{ msg: z.ZodString }>> = {
      name: "dummy_tool",
      description: "A dummy tool",
      parameters: z.object({ msg: z.string() }),
      handler: () => Promise.resolve("dummy ok"),
    };

    const history: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "What events are happening in City X?",
      }),
    ];

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
            id: "call-search",
            type: "tool_call",
            name: "search_events",
            parameters: { location: "City X" },
            isOwn: true,
            timestamp: Date.now(),
          },
        ]);
      }
      if (step === 2) {
        // Model inspects the scratch file
        return Promise.resolve([
          {
            id: "call-read-1",
            type: "tool_call",
            name: readScratchFileToolName,
            parameters: { id: "call-search", startLine: 1, numLines: 80 },
            isOwn: true,
            timestamp: Date.now(),
          },
        ]);
      }
      if (step === 3) {
        // Model inspects next chunk
        return Promise.resolve([
          {
            id: "call-read-2",
            type: "tool_call",
            name: readScratchFileToolName,
            parameters: { id: "call-search", startLine: 81, numLines: 80 },
            isOwn: true,
            timestamp: Date.now(),
          },
        ]);
      }
      if (step === 4) {
        // Model calls dummy tool (so call-read-1 is now 2 turns old)
        return Promise.resolve([
          {
            id: "call-dummy",
            type: "tool_call",
            name: "dummy_tool",
            parameters: { msg: "test" },
            isOwn: true,
            timestamp: Date.now(),
          },
        ]);
      }
      // On step 5, the model formulates final reply based on preserved read results
      return Promise.resolve([
        ownUtteranceTurn("Found 150 concerts in City X."),
      ]);
    };

    await pipe(
      injectCallModel(fakeCallModel),
      agentDeps(history),
    )(async () => {
      await runAgent({
        provider: "moonshot",
        maxIterations: 8,
        tools: [searchTool, dummyTool],
        prompt: "You are an events assistant.",
        rewriteHistory: noopRewriteHistory,
        toolOutputScratchPad: scratchPad,
        timezoneIANA: "UTC",
      });
    })();

    // Verify step 5 history contains the actual search lines without nested scratch headers or compaction replacement
    const step5History = modelReceivedHistories[4];
    const read1Result = step5History.find((e) =>
      e.type === "tool_result" && e.toolCallId === "call-read-1"
    ) as Extract<HistoryEvent, { type: "tool_result" }> | undefined;

    assertEquals(
      read1Result !== undefined,
      true,
      "Expected call-read-1 result to be present in step 5 history",
    );
    assertStringIncludes(
      read1Result!.result,
      "Event 1: Concert at Hall 1",
      "call-read-1 output must retain actual event content, not a recursive Memory TLDR",
    );
    assertEquals(
      read1Result!.result.includes("Because time has passed"),
      false,
      "read_scratch_file output must not be compacted to Memory TLDR during the active run",
    );

    const finalUtterance = history.find((e) => e.type === "own_utterance");
    assertEquals(
      finalUtterance?.text,
      "Found 150 concerts in City X.",
    );
  },
);
