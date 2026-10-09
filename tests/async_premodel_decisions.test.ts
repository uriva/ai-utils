import { assert, assertEquals } from "@std/assert";
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
import { cleanActiveMemoryToolName } from "../src/utils.ts";
import { runAgent } from "../mod.ts";
import { z } from "zod/v4";

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
  "pre-model decisions are non-blocking: turn 1 model call starts without waiting for decision model latency, and decisions settle additively",
  async () => {
    const baseTime = 1700000000000;
    const filler = "long log entry with detailed diagnostic context ".repeat(
      500,
    );

    // Concluded tool episodes qualifying for memory cleanup
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
        timestamp: baseTime + 1500,
      },
      {
        ...toolUseTurn({ name: "tool3", args: {} }),
        timestamp: baseTime + 1600,
        id: "c3",
      },
      {
        ...toolResultTurn({ toolCallId: "c3", result: filler }),
        timestamp: baseTime + 1700,
        id: "r3",
      },
      {
        ...ownUtteranceTurn("Step 3 done."),
        timestamp: baseTime + 1800,
      },
      {
        ...participantUtteranceTurn({
          name: "user",
          text: "Export the results now.",
        }),
        timestamp: baseTime + 2000,
      },
    ];

    const SIMULATED_DECISION_LATENCY_MS = 500;
    let decisionStartTime = 0;
    let decisionEndTime = 0;
    let callModelStartTime = 0;
    let callModelInvokedWhileDecisionInFlight = false;

    const mockDecisionCaller = async (
      _state: unknown,
      // deno-lint-ignore no-explicit-any
      questions: Record<string, any>,
      // deno-lint-ignore no-explicit-any
    ): Promise<Record<string, any>> => {
      const isPreModelDecision = "data_exporter" in questions ||
        Object.keys(questions).some((k) => k.startsWith("compact_turn_"));
      if (isPreModelDecision) {
        if (decisionStartTime === 0) {
          decisionStartTime = performance.now();
        }
        await delay(SIMULATED_DECISION_LATENCY_MS);
        decisionEndTime = performance.now();
      }
      // deno-lint-ignore no-explicit-any
      const answers: Record<string, any> = {};
      for (const key of Object.keys(questions)) {
        if (key === "is_hallucination") {
          answers[key] = { type: "noul", noul: 0.05 };
        } else if (key === "requires_flash") {
          answers[key] = { type: "choice", choice: "lite" };
        } else {
          answers[key] = { type: "noul", noul: 0.95 };
        }
      }
      return answers;
    };

    const fakeCallModel = (
      _received: HistoryEvent[],
    ): Promise<HistoryEvent[]> => {
      callModelStartTime = performance.now();
      if (decisionStartTime > 0 && decisionEndTime === 0) {
        callModelInvokedWhileDecisionInFlight = true;
      }
      return Promise.resolve([
        ownUtteranceTurn("Export initiated immediately."),
      ]);
    };

    let agentRunStart = 0;

    await pipe(
      injectDecisionModel(mockDecisionCaller),
      injectCallModel(fakeCallModel),
      inMemoryDeps(history),
    )(async () => {
      agentRunStart = performance.now();
      await runAgent({
        provider: "anthropic",
        maxIterations: 5,
        tools: [],
        skills: [sampleSkill],
        prompt: "You are an assistant.",
        timezoneIANA: "UTC",
      });
    })();

    const timeUntilCallModelMs = callModelStartTime - agentRunStart;

    // 1. LATENCY ASSERTION: CallModel must start without waiting for the 500ms decision model
    assert(
      timeUntilCallModelMs < 300,
      `Expected callModel to start without waiting for decision model (< 300ms), but it was blocked for ${
        Math.round(timeUntilCallModelMs)
      }ms by pre-model decisions`,
    );

    // 2. CONCURRENCY ASSERTION: CallModel must be dispatched concurrently while pre-model decisions are still running
    assert(
      callModelInvokedWhileDecisionInFlight,
      "Expected callModel to be invoked while pre-model decisions were in-flight, but callModel waited until after decisions completed",
    );

    // 3. ADDITIVE ASSERTION: Decisions must complete in the background and additively record skills and memory cleanup to history
    const autoLearnEvents = history.filter(
      (e) => e.type === "tool_call" && e.id.startsWith("auto-learn-"),
    );
    assert(
      autoLearnEvents.length > 0,
      "Expected skill auto-learn event to be additively recorded in history by background task",
    );

    const autoCleanEvents = history.filter(
      (e) => e.type === "tool_call" && e.name === cleanActiveMemoryToolName,
    );
    assert(
      autoCleanEvents.length > 0,
      "Expected memory cleanup event to be additively recorded in history by background task",
    );
  },
);

Deno.test(
  "additive background decisions: turn 1 initiates model call without delay and subsequent iteration picks up background decisions",
  async () => {
    const baseTime = 1700000000000;
    const filler = "long log entry with detailed diagnostic context ".repeat(
      500,
    );

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
        ...participantUtteranceTurn({
          name: "user",
          text: "Check status then export.",
        }),
        timestamp: baseTime + 2000,
      },
    ];

    const SIMULATED_DECISION_LATENCY_MS = 200;
    let turn1ModelCallTime = 0;
    let turn1DecisionStartTime = 0;
    let iterationsCount = 0;

    const mockDecisionCaller = async (
      _state: unknown,
      // deno-lint-ignore no-explicit-any
      questions: Record<string, any>,
      // deno-lint-ignore no-explicit-any
    ): Promise<Record<string, any>> => {
      const isPreModelDecision = "data_exporter" in questions ||
        Object.keys(questions).some((k) => k.startsWith("compact_turn_"));
      if (isPreModelDecision) {
        if (turn1DecisionStartTime === 0) {
          turn1DecisionStartTime = performance.now();
        }
        await delay(SIMULATED_DECISION_LATENCY_MS);
      }
      // deno-lint-ignore no-explicit-any
      const answers: Record<string, any> = {};
      for (const key of Object.keys(questions)) {
        if (key === "is_hallucination") {
          answers[key] = { type: "noul", noul: 0.05 };
        } else if (key === "requires_flash") {
          answers[key] = { type: "choice", choice: "lite" };
        } else {
          answers[key] = { type: "noul", noul: 0.95 };
        }
      }
      return answers;
    };

    const fakeCallModel = (
      _received: HistoryEvent[],
    ): Promise<HistoryEvent[]> => {
      iterationsCount++;
      if (iterationsCount === 1) {
        turn1ModelCallTime = performance.now();
        // Turn 1 calls a quick status tool
        return Promise.resolve([
          toolUseTurn({ name: "status_check", args: {} }),
        ]);
      }
      // Turn 2 finishes
      return Promise.resolve([
        ownUtteranceTurn("Status checked and exported."),
      ]);
    };

    let agentRunStart = 0;

    await pipe(
      injectDecisionModel(mockDecisionCaller),
      injectCallModel(fakeCallModel),
      inMemoryDeps(history),
    )(async () => {
      agentRunStart = performance.now();
      await runAgent({
        provider: "anthropic",
        maxIterations: 5,
        tools: [
          {
            name: "status_check",
            description: "Quick status check",
            parameters: z.object({}),
            handler: () => Promise.resolve("All good"),
          },
        ],
        skills: [sampleSkill],
        prompt: "You are an assistant.",
        timezoneIANA: "UTC",
      });
    })();

    const turn1DelayMs = turn1ModelCallTime - agentRunStart;

    // Turn 1 must NOT be blocked by the 200ms decision model
    assert(
      turn1DelayMs < 100,
      `Expected Turn 1 to start immediately (< 100ms), but took ${
        Math.round(turn1DelayMs)
      }ms`,
    );

    // Verify background decisions were additive across iterations
    const autoLearnEvents = history.filter(
      (e) => e.type === "tool_call" && e.id.startsWith("auto-learn-"),
    );
    assert(
      autoLearnEvents.length > 0,
      "Expected skill auto-learn event to be additively recorded in history",
    );
  },
);

Deno.test(
  "race condition guard: if LLM explicitly learns a skill before background decision completes, duplicate auto-learn is not emitted",
  async () => {
    const history: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Please export the report.",
      }),
    ];

    const SIMULATED_DECISION_LATENCY_MS = 200;
    let iterationsCount = 0;

    const mockDecisionCaller = async (
      _state: unknown,
      // deno-lint-ignore no-explicit-any
      questions: Record<string, any>,
      // deno-lint-ignore no-explicit-any
    ): Promise<Record<string, any>> => {
      if ("data_exporter" in questions) {
        await delay(SIMULATED_DECISION_LATENCY_MS);
      }
      // deno-lint-ignore no-explicit-any
      const answers: Record<string, any> = {};
      for (const key of Object.keys(questions)) {
        if (key === "is_hallucination") {
          answers[key] = { type: "noul", noul: 0.05 };
        } else if (key === "requires_flash") {
          answers[key] = { type: "choice", choice: "lite" };
        } else {
          answers[key] = { type: "noul", noul: 0.95 };
        }
      }
      return answers;
    };

    const fakeCallModel = (
      _received: HistoryEvent[],
    ): Promise<HistoryEvent[]> => {
      iterationsCount++;
      if (iterationsCount === 1) {
        // LLM explicitly learned the skill before background decision finished
        return Promise.resolve([
          toolUseTurn({
            name: "learn_skill",
            args: { skillName: "data_exporter" },
          }),
        ]);
      }
      return Promise.resolve([
        ownUtteranceTurn("Report exported."),
      ]);
    };

    await pipe(
      injectDecisionModel(mockDecisionCaller),
      injectCallModel(fakeCallModel),
      inMemoryDeps(history),
    )(async () => {
      await runAgent({
        provider: "anthropic",
        maxIterations: 3,
        tools: [],
        skills: [sampleSkill],
        prompt: "You are an assistant.",
        timezoneIANA: "UTC",
      });
    })();

    // History should have exactly ONE learn_skill call (the explicit LLM one)
    const learnCalls = history.filter(
      (e) => e.type === "tool_call" && e.name === "learn_skill",
    );
    assert(
      !history.some((e) =>
        e.type === "tool_call" && e.id.startsWith("auto-learn-")
      ),
      "Should not emit duplicate auto-learn event when LLM already learned the skill",
    );
    assert(
      learnCalls.length === 1,
      `Expected exactly 1 learn_skill call, found ${learnCalls.length}`,
    );
  },
);

Deno.test(
  "race condition guard: if LLM invokes skill tool via run_command before background decision completes, auto-learn is still emitted",
  async () => {
    const history: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Please export the report.",
      }),
    ];

    const SIMULATED_DECISION_LATENCY_MS = 200;
    let iterationsCount = 0;

    const mockDecisionCaller = async (
      _state: unknown,
      // deno-lint-ignore no-explicit-any
      questions: Record<string, any>,
      // deno-lint-ignore no-explicit-any
    ): Promise<Record<string, any>> => {
      if ("data_exporter" in questions) {
        await delay(SIMULATED_DECISION_LATENCY_MS);
      }
      // deno-lint-ignore no-explicit-any
      const answers: Record<string, any> = {};
      for (const key of Object.keys(questions)) {
        if (key === "is_hallucination") {
          answers[key] = { type: "noul", noul: 0.05 };
        } else if (key === "requires_flash") {
          answers[key] = { type: "choice", choice: "lite" };
        } else {
          answers[key] = { type: "noul", noul: 0.95 };
        }
      }
      return answers;
    };

    const fakeCallModel = (
      _received: HistoryEvent[],
    ): Promise<HistoryEvent[]> => {
      iterationsCount++;
      if (iterationsCount === 1) {
        return Promise.resolve([
          toolUseTurn({
            name: "run_command",
            args: { command: "data_exporter/export", params: {} },
          }),
        ]);
      }
      return Promise.resolve([
        ownUtteranceTurn("Report exported."),
      ]);
    };

    await pipe(
      injectDecisionModel(mockDecisionCaller),
      injectCallModel(fakeCallModel),
      inMemoryDeps(history),
    )(async () => {
      await runAgent({
        provider: "anthropic",
        maxIterations: 3,
        tools: [],
        skills: [sampleSkill],
        prompt: "You are an assistant.",
        timezoneIANA: "UTC",
      });
    })();

    const autoLearnCalls = history.filter(
      (e) => e.type === "tool_call" && e.id.startsWith("auto-learn-"),
    );
    assert(
      autoLearnCalls.length === 1,
      `Expected auto-learn event to be emitted even when LLM executed skill tool via run_command before decision settled. Found: ${autoLearnCalls.length}`,
    );
  },
);

Deno.test(
  "race condition guard: if LLM invokes skill tool, background decision must NOT emit auto-unlearn for that skill",
  async () => {
    // The skill is already active from earlier
    const history: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Generate report.",
      }),
      toolUseTurn({
        name: "learn_skill",
        args: { skillName: "data_exporter" },
      }),
      toolResultTurn({
        toolCallId: "init-learn",
        result: 'Skill "data_exporter" learned successfully.',
      }),
      ownUtteranceTurn("Ready."),
      participantUtteranceTurn({
        name: "user",
        text: "Please run the second export.",
      }),
    ];

    const SIMULATED_DECISION_LATENCY_MS = 200;
    let iterationsCount = 0;

    // Decision caller scores data_exporter low (< 0.20), so without guard it would emit auto-unlearn
    const mockDecisionCaller = async (
      _state: unknown,
      // deno-lint-ignore no-explicit-any
      questions: Record<string, any>,
      // deno-lint-ignore no-explicit-any
    ): Promise<Record<string, any>> => {
      await delay(SIMULATED_DECISION_LATENCY_MS);
      // deno-lint-ignore no-explicit-any
      const answers: Record<string, any> = {};
      for (const key of Object.keys(questions)) {
        if (key === "is_hallucination") {
          answers[key] = { type: "noul", noul: 0.05 };
        } else if (key === "requires_flash") {
          answers[key] = { type: "choice", choice: "lite" };
        } else {
          // Score 0.05 for skills, which triggers toUnlearn
          answers[key] = { type: "noul", noul: 0.05 };
        }
      }
      return answers;
    };

    const fakeCallModel = (
      _received: HistoryEvent[],
    ): Promise<HistoryEvent[]> => {
      iterationsCount++;
      if (iterationsCount === 1) {
        // LLM immediately executes tool from data_exporter
        return Promise.resolve([
          toolUseTurn({
            name: "run_command",
            args: { command: "data_exporter/export", params: {} },
          }),
        ]);
      }
      return Promise.resolve([
        ownUtteranceTurn("Export complete."),
      ]);
    };

    await pipe(
      injectDecisionModel(mockDecisionCaller),
      injectCallModel(fakeCallModel),
      inMemoryDeps(history),
    )(async () => {
      await runAgent({
        provider: "anthropic",
        maxIterations: 3,
        tools: [],
        skills: [sampleSkill],
        prompt: "You are an assistant.",
        timezoneIANA: "UTC",
      });
    })();

    // Check if auto-unlearn was emitted for data_exporter
    const autoUnlearnCalls = history.filter(
      (e) =>
        e.type === "tool_call" &&
        e.id.startsWith("auto-unlearn-") &&
        // deno-lint-ignore no-explicit-any
        (e.parameters as any)?.skillName === "data_exporter",
    );
    assertEquals(
      autoUnlearnCalls.length,
      0,
      `Should not emit auto-unlearn for a skill that was invoked in the current request. Found: ${
        JSON.stringify(autoUnlearnCalls)
      }`,
    );
  },
);

Deno.test(
  "background decisions must run at most once per agent run and not repeatedly trigger across tool iterations",
  async () => {
    const history: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Run three steps.",
      }),
    ];

    let skillDecisionCallsCount = 0;
    let iterationsCount = 0;

    const mockDecisionCaller = (
      _state: unknown,
      // deno-lint-ignore no-explicit-any
      questions: Record<string, any>,
      // deno-lint-ignore no-explicit-any
    ): Promise<Record<string, any>> => {
      if ("data_exporter" in questions) {
        skillDecisionCallsCount++;
      }
      // deno-lint-ignore no-explicit-any
      const answers: Record<string, any> = {};
      for (const key of Object.keys(questions)) {
        if (key === "is_hallucination") {
          answers[key] = { type: "noul", noul: 0.05 };
        } else if (key === "requires_flash") {
          answers[key] = { type: "choice", choice: "lite" };
        } else {
          answers[key] = { type: "noul", noul: 0.95 };
        }
      }
      return Promise.resolve(answers);
    };

    const fakeCallModel = (
      _received: HistoryEvent[],
    ): Promise<HistoryEvent[]> => {
      iterationsCount++;
      if (iterationsCount === 1) {
        return Promise.resolve([
          toolUseTurn({
            name: "run_command",
            args: { command: "data_exporter/export", params: {} },
          }),
        ]);
      }
      if (iterationsCount === 2) {
        return Promise.resolve([
          toolUseTurn({
            name: "run_command",
            args: { command: "data_exporter/export", params: {} },
          }),
        ]);
      }
      return Promise.resolve([
        ownUtteranceTurn("All three steps done."),
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

    assertEquals(
      skillDecisionCallsCount,
      1,
      `Pre-model skill decision should only run once per agent run, but was called ${skillDecisionCallsCount} times`,
    );
  },
);

Deno.test(
  "waitForBackgroundDecisions: false allows agent run to complete without waiting for slow background decisions",
  async () => {
    const history: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Tell me a joke.",
      }),
    ];

    const SLOW_DECISION_LATENCY_MS = 600;
    let decisionCompleted = false;

    const mockDecisionCaller = async (
      _state: unknown,
      // deno-lint-ignore no-explicit-any
      questions: Record<string, any>,
      // deno-lint-ignore no-explicit-any
    ): Promise<Record<string, any>> => {
      if (
        "data_exporter" in questions ||
        Object.keys(questions).some((k) => k.startsWith("compact_turn_"))
      ) {
        await delay(SLOW_DECISION_LATENCY_MS);
        decisionCompleted = true;
      }
      // deno-lint-ignore no-explicit-any
      const answers: Record<string, any> = {};
      for (const key of Object.keys(questions)) {
        if (key === "is_hallucination") {
          answers[key] = { type: "noul", noul: 0.05 };
        } else if (key === "requires_flash") {
          answers[key] = { type: "choice", choice: "lite" };
        } else {
          answers[key] = { type: "noul", noul: 0.95 };
        }
      }
      return answers;
    };

    const fakeCallModel = (
      _received: HistoryEvent[],
    ): Promise<HistoryEvent[]> => {
      return Promise.resolve([
        ownUtteranceTurn("Why did the chicken cross the road?"),
      ]);
    };

    const startTime = performance.now();
    await pipe(
      injectDecisionModel(mockDecisionCaller),
      injectCallModel(fakeCallModel),
      inMemoryDeps(history),
    )(async () => {
      await runAgent({
        provider: "anthropic",
        maxIterations: 3,
        tools: [],
        skills: [sampleSkill],
        prompt: "You are an assistant.",
        timezoneIANA: "UTC",
        waitForBackgroundDecisions: false,
      });
    })();
    const elapsedMs = performance.now() - startTime;

    assert(
      elapsedMs < 250,
      `Expected runAgent with waitForBackgroundDecisions: false to return in < 250ms, but took ${
        Math.round(elapsedMs)
      }ms`,
    );
    assertEquals(
      decisionCompleted,
      false,
      "Background decision should still be in-flight when runAgent returns",
    );

    // Wait for the background task to settle cleanly before ending test
    await delay(SLOW_DECISION_LATENCY_MS);
    assertEquals(decisionCompleted, true);
  },
);
