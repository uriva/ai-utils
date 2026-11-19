import { assert, assertEquals } from "@std/assert";
import { pipe } from "gamla";
import { z } from "zod/v4";
import {
  type HistoryEvent,
  injectCallModel,
  ownThoughtTurn,
  ownUtteranceTurn,
  participantUtteranceTurn,
  runAgent,
  tool,
} from "../mod.ts";
import { agentDeps } from "../test_helpers.ts";

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

Deno.test(
  "runAgent does not trigger compactHistory when static prompt and skills exceed threshold while history is small",
  async () => {
    const history: HistoryEvent[] = [
      participantUtteranceTurn({ name: "user", text: "Hi, quick question?" }),
    ];
    let compactCalls = 0;

    const largePrompt = "Detailed agent policy guideline sentence. ".repeat(
      4_500,
    );

    const fakeCallModel = (): Promise<HistoryEvent[]> =>
      Promise.resolve([ownUtteranceTurn("Sure, happy to help!")]);

    await pipe(
      injectCallModel(fakeCallModel),
      agentDeps(history),
    )(() =>
      runAgent({
        maxIterations: 5,
        tools: [],
        prompt: largePrompt,
        timezoneIANA: "UTC",
        historyCompactionTokenThreshold: 25_000,
        compactHistory: () => {
          compactCalls++;
          return Promise.resolve();
        },
      })
    )();

    await delay(50);
    assertEquals(
      compactCalls,
      0,
      `Expected 0 compactHistory calls for a 2-message history despite large static prompt, got ${compactCalls}`,
    );
  },
);

Deno.test(
  "runAgent does not thrash compactHistory mid-loop during multi-step tool execution and preserves append-only prefix",
  async () => {
    const chunkPayload = "unique detail alpha beta gamma delta epsilon zeta "
      .repeat(360);
    const history: HistoryEvent[] = [
      {
        id: "prior-u-1",
        type: "participant_utterance",
        isOwn: false,
        name: "user",
        text: `Prior context A: ${chunkPayload}`,
        timestamp: Date.now() - 120_000,
      },
      {
        id: "prior-a-1",
        type: "own_utterance",
        isOwn: true,
        text: `Prior response A: ${chunkPayload}`,
        timestamp: Date.now() - 110_000,
      },
      {
        id: "prior-u-2",
        type: "participant_utterance",
        isOwn: false,
        name: "user",
        text: `Prior context B: ${chunkPayload}`,
        timestamp: Date.now() - 100_000,
      },
      {
        id: "prior-a-2",
        type: "own_utterance",
        isOwn: true,
        text: `Prior response B: ${chunkPayload}`,
        timestamp: Date.now() - 90_000,
      },
      participantUtteranceTurn({
        name: "user",
        text: "Research all five topics and summarize them.",
      }),
    ];
    const searchTool = tool({
      name: "fetch_research",
      description: "Fetch research data for a step",
      parameters: z.object({ step: z.number() }),
      handler: ({ step }) =>
        Promise.resolve(`Step ${step} findings: ${chunkPayload}`),
    });

    const totalToolSteps = 5;
    let modelStep = 0;
    let midLoopCompactions = 0;
    let totalCompactions = 0;
    const seenHistoryIdsPerStep: string[][] = [];

    const fakeCallModel = async (
      passedHistory: HistoryEvent[],
    ): Promise<HistoryEvent[]> => {
      const currentStep = modelStep++;
      seenHistoryIdsPerStep.push(passedHistory.map((e) => e.id));
      await delay(15);
      if (currentStep < totalToolSteps) {
        return [
          {
            id: `call-step-${currentStep}`,
            type: "tool_call",
            isOwn: true,
            name: "fetch_research",
            parameters: { step: currentStep },
            timestamp: Date.now(),
          },
        ];
      }
      return [ownUtteranceTurn("All five topics have been summarized.")];
    };

    await pipe(
      injectCallModel(fakeCallModel),
      agentDeps(history),
    )(() =>
      runAgent({
        maxIterations: 10,
        tools: [searchTool],
        prompt: "You are a research assistant.",
        timezoneIANA: "UTC",
        historyCompactionTokenThreshold: 25_000,
        compactHistory: () => {
          totalCompactions++;
          if (modelStep <= totalToolSteps) {
            midLoopCompactions++;
            // Simulate DO rewrite replacing prefix with a summary
            const summary = ownThoughtTurn("Compacted mid-loop summary");
            summary.id = `summary-mid-${totalCompactions}`;
            history.splice(0, Math.max(0, history.length - 2), summary);
          }
          return Promise.resolve();
        },
      })
    )();

    await delay(60);

    assertEquals(
      midLoopCompactions,
      0,
      `Expected 0 mid-loop compactions during a 5-step tool run (~30k history), got ${midLoopCompactions}`,
    );
    assertEquals(
      totalCompactions,
      1,
      `Expected exactly 1 end-of-turn compaction after turn completed, got ${totalCompactions}`,
    );

    // Every consecutive step must preserve the exact prefix of the previous step (prefix cache intact)
    for (let i = 1; i < seenHistoryIdsPerStep.length; i++) {
      const prev = seenHistoryIdsPerStep[i - 1];
      const curr = seenHistoryIdsPerStep[i];
      const prefixPreserved = prev.every((id, idx) => curr[idx] === id);
      assert(
        prefixPreserved,
        `Step ${i} broke the append-only history prefix (prompt cache invalidated mid-loop)`,
      );
    }
  },
);

Deno.test(
  "runAgent triggers compactHistory when accumulated summaries exceed summaryRollupTokenThreshold even when raw history is small",
  async () => {
    const baseTime = Date.now() - 24 * 60 * 60 * 1000;
    const twoHours = 2 * 60 * 60 * 1000;
    const summaryBody =
      "Past conversation history was compacted into a structured summary.\n\n" +
      "Detailed historical note from an earlier session. ".repeat(150);

    const history: HistoryEvent[] = [
      ...Array.from({ length: 8 }, (_, i) => {
        const start = baseTime + i * twoHours;
        return {
          id: `summary-${start}-${start + 60_000}`,
          type: "own_thought" as const,
          isOwn: true as const,
          text: `[This summary covers the period from Jan ${i + 1} to Jan ${
            i + 1
          }]\n\n${summaryBody}`,
          timestamp: start,
        };
      }),
      participantUtteranceTurn({ name: "user", text: "Hello!" }),
    ];

    let compactCalls = 0;

    await pipe(
      injectCallModel(() => Promise.resolve([ownUtteranceTurn("Hi!")])),
      agentDeps(history),
    )(() =>
      runAgent({
        maxIterations: 5,
        tools: [],
        prompt: "You are a helpful assistant.",
        timezoneIANA: "UTC",
        historyCompactionTokenThreshold: 25_000,
        summaryRollupTokenThreshold: 4_000,
        compactHistory: () => {
          compactCalls++;
          return Promise.resolve();
        },
      })
    )();

    await delay(50);
    assertEquals(
      compactCalls,
      1,
      `Expected 1 compactHistory call to roll up accumulated summaries, got ${compactCalls}`,
    );
  },
);

Deno.test(
  "runAgent triggers emergency mid-run compaction once when history exceeds emergency ceiling without thrashing",
  async () => {
    const hugeChunk = "emergency runaway payload token block ".repeat(4_500);
    const history: HistoryEvent[] = [
      ...Array.from({ length: 5 }, (_, i) => ({
        id: `huge-prior-${i}`,
        type: "own_utterance" as const,
        isOwn: true as const,
        text: `Block ${i}: ${hugeChunk}`,
        timestamp: Date.now() - (10 - i) * 10_000,
      })),
      participantUtteranceTurn({ name: "user", text: "Continue task" }),
    ];

    const stepTool = tool({
      name: "small_step",
      description: "Run a small step",
      parameters: z.object({ n: z.number() }),
      handler: ({ n }) => Promise.resolve(`small step ${n} ok`),
    });

    let step = 0;
    let midLoopCompactions = 0;

    await pipe(
      injectCallModel(async () => {
        const curr = step++;
        await delay(15);
        if (curr < 3) {
          return [
            {
              id: `small-call-${curr}`,
              type: "tool_call" as const,
              isOwn: true as const,
              name: "small_step",
              parameters: { n: curr },
              timestamp: Date.now(),
            },
          ];
        }
        return [ownUtteranceTurn("Done.")];
      }),
      agentDeps(history),
    )(() =>
      runAgent({
        maxIterations: 10,
        tools: [stepTool],
        prompt: "You are an assistant.",
        timezoneIANA: "UTC",
        historyCompactionTokenThreshold: 25_000,
        compactHistory: () => {
          if (step <= 3) midLoopCompactions++;
          return Promise.resolve();
        },
      })
    )();

    await delay(60);
    assertEquals(
      midLoopCompactions,
      1,
      `Expected exactly 1 emergency mid-run compaction (not thrashing on all 3 steps), got ${midLoopCompactions}`,
    );
  },
);
