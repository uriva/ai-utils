import { assert } from "@std/assert";
import { pipe } from "gamla";
import {
  type HistoryEvent,
  injectAccessHistory,
  injectCallModel,
  injectOutputEvent,
  ownUtteranceTurn,
  participantUtteranceTurn,
  type Skill,
  toolResultTurn,
  toolUseTurn,
} from "../src/agent.ts";
import { injectDecisionModel } from "../src/decisionModel.ts";
import { runAgent } from "../mod.ts";

const inMemoryDeps = (inMemoryHistory: HistoryEvent[]) =>
  pipe(
    injectAccessHistory(() => Promise.resolve(inMemoryHistory)),
    injectOutputEvent((event) => {
      inMemoryHistory.push(event);
      return Promise.resolve();
    }),
  );

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

const sampleSkill: Skill = {
  name: "data_exporter",
  description: "Export datasets to external files",
  instructions: "When asked to export data, use data_exporter.",
  tools: [],
};

Deno.test(
  "pre-LLM decision model checks (skills and memory cleanup) run in parallel rather than serially",
  async () => {
    const baseTime = 1700000000000;
    const filler = "long log entry with detailed diagnostic context ".repeat(
      500,
    ); // ~24,000 chars, exceeds 5000 tokens

    // 13 events with concluded tool episodes (qualifies for memory cleanup)
    const history: HistoryEvent[] = [
      {
        ...participantUtteranceTurn({ name: "user", text: "Step 1" }),
        timestamp: baseTime + 100,
      },
      {
        ...toolUseTurn({ name: "tool1", args: {} }),
        timestamp: baseTime + 200,
        id: "c1",
      },
      {
        ...toolResultTurn({ toolCallId: "c1", result: filler }),
        timestamp: baseTime + 300,
        id: "r1",
      },
      {
        ...ownUtteranceTurn("Step 1 done."),
        timestamp: baseTime + 400,
      },

      {
        ...participantUtteranceTurn({ name: "user", text: "Step 2" }),
        timestamp: baseTime + 1000,
      },
      {
        ...toolUseTurn({ name: "tool2", args: {} }),
        timestamp: baseTime + 1100,
        id: "c2",
      },
      {
        ...toolResultTurn({ toolCallId: "c2", result: filler }),
        timestamp: baseTime + 1200,
        id: "r2",
      },
      {
        ...ownUtteranceTurn("Step 2 done."),
        timestamp: baseTime + 1300,
      },

      {
        ...participantUtteranceTurn({ name: "user", text: "Step 3" }),
        timestamp: baseTime + 2000,
      },
      {
        ...toolUseTurn({ name: "tool3", args: {} }),
        timestamp: baseTime + 2100,
        id: "c3",
      },
      {
        ...toolResultTurn({ toolCallId: "c3", result: "ok" }),
        timestamp: baseTime + 2200,
        id: "r3",
      },
      {
        ...ownUtteranceTurn("Step 3 done."),
        timestamp: baseTime + 2300,
      },

      {
        ...participantUtteranceTurn({
          name: "user",
          text: "Export the results.",
        }),
        timestamp: baseTime + 3000,
      },
    ];

    const SIMULATED_DECISION_LATENCY_MS = 200;
    let preModelDecisionCallsCount = 0;
    let firstDecisionStartTime = 0;
    let lastDecisionEndTime = 0;

    const mockDecisionCaller = async (
      _state: unknown,
      // deno-lint-ignore no-explicit-any
      questions: Record<string, any>,
      // deno-lint-ignore no-explicit-any
    ): Promise<Record<string, any>> => {
      const isHallucinationAudit = "is_hallucination" in questions;
      if (!isHallucinationAudit) {
        preModelDecisionCallsCount++;
        const now = performance.now();
        if (firstDecisionStartTime === 0) {
          firstDecisionStartTime = now;
        }
        await delay(SIMULATED_DECISION_LATENCY_MS);
        lastDecisionEndTime = performance.now();
      }
      // deno-lint-ignore no-explicit-any
      const answers: Record<string, any> = {};
      for (const key of Object.keys(questions)) {
        if (key === "is_hallucination") {
          answers[key] = { type: "noul", noul: 0.05 };
        } else {
          answers[key] = { type: "noul", noul: 0.9 };
        }
      }
      return answers;
    };

    const fakeCallModel = (
      _received: HistoryEvent[],
    ): Promise<HistoryEvent[]> => {
      return Promise.resolve([
        ownUtteranceTurn("Here are the exported results."),
      ]);
    };

    await pipe(
      injectDecisionModel(mockDecisionCaller),
      injectCallModel(fakeCallModel),
      inMemoryDeps(history),
    )(async () => {
      await runAgent({
        provider: "anthropic",
        maxIterations: 5,
        tools: [],
        skills: [sampleSkill],
        prompt: "You are an assistant.",
        timezoneIANA: "UTC",
      });
    })();

    assert(
      preModelDecisionCallsCount >= 2,
      `Expected at least 2 pre-model decision model calls (skills + cleanup), got ${preModelDecisionCallsCount}`,
    );

    const decisionSpanMs = lastDecisionEndTime - firstDecisionStartTime;

    // If serial: 2 * 200ms = 400ms minimum.
    // If parallel: max(200ms, 200ms) = ~200ms.
    // A threshold of 320ms strictly fails serial execution while allowing plenty of headroom for parallel.
    assert(
      decisionSpanMs < 320,
      `Expected parallel execution under 320ms, but took ${
        Math.round(decisionSpanMs)
      }ms (serial execution detected)`,
    );
  },
);
