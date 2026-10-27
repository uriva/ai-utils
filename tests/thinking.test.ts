import { assert, assertEquals } from "@std/assert";
import {
  type GenerateContentParameters,
  type GenerateContentResponse,
  ThinkingLevel,
} from "@google/genai";
import { injectCallModelWrapper, runAgent } from "../mod.ts";
import {
  getStreamThinkingChunk,
  type HistoryEvent,
  injectCallModel,
  ownThoughtTurn,
  ownUtteranceTurn,
  participantUtteranceTurn,
  toolUseTurn,
} from "../src/agent.ts";
import {
  type DecisionAnswer,
  injectDecisionModel,
} from "../src/decisionModel.ts";
import {
  geminiGenJsonFromConvo,
  geminiThinkingConfig,
  injectGeminiGenerateContent,
} from "../src/gemini.ts";
import { pipe } from "gamla";
import { z } from "zod/v4";
import { agentDeps, runForAllProviders } from "../test_helpers.ts";

runForAllProviders(
  "agent returns own_thought events when thinking is enabled",
  async (runAgent) => {
    if (
      Deno.env.get("TEST_PROVIDER") === "google" ||
      Deno.env.get("TEST_PROVIDER") === "gemini"
    ) return;
    const mockHistory = [
      participantUtteranceTurn({
        name: "user",
        text: "What is 137 * 248? Think step by step.",
      }),
    ];

    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 1,
      tools: [],
      prompt: "You are a helpful assistant. Think carefully before answering.",
      timezoneIANA: "UTC",
    });

    const thoughts = mockHistory.filter((e) => e.type === "own_thought");
    assert(
      thoughts.length > 0,
      `Expected at least one own_thought event from thinking, but got none. Events: ${
        mockHistory.map((e) => e.type).join(", ")
      }`,
    );
  },
);

// Streaming is a contract between runAgent and the injected callModel,
// independent of provider SDK. Injecting a fake callModel that fires chunks
// during the call lets us test the contract deterministically and provider-
// agnostically.
Deno.test(
  "onStreamThinkingChunk receives thinking chunks fired during callModel",
  async () => {
    let thinkingText = "";
    let thinkingChunkCount = 0;

    const fakeCallModel = async () => {
      const emit = getStreamThinkingChunk();
      await emit("The answer is ");
      await emit("42 because ");
      await emit("math says so.");
      return [
        ownThoughtTurn("The answer is 42 because math says so."),
        ownUtteranceTurn("42"),
      ];
    };

    await injectCallModel(fakeCallModel)(async () => {
      await agentDeps([
        participantUtteranceTurn({ name: "user", text: "what is 6*7?" }),
      ])(runAgent)({
        maxIterations: 1,
        tools: [],
        prompt: "unused in fake",
        onStreamThinkingChunk: (chunk) => {
          thinkingText += chunk;
          thinkingChunkCount++;
        },
        timezoneIANA: "UTC",
      });
    })();

    assert(
      thinkingChunkCount === 3,
      `expected 3 thinking chunks, got ${thinkingChunkCount}`,
    );
    assert(
      thinkingText === "The answer is 42 because math says so.",
      `expected assembled thinking text, got: ${thinkingText}`,
    );
  },
);

Deno.test(
  "geminiThinkingConfig uses ThinkingLevel constants and never thinkingBudget numbers",
  () => {
    const highConfig = geminiThinkingConfig(ThinkingLevel.HIGH);
    assert(highConfig.includeThoughts === true);
    assertEquals(highConfig.thinkingLevel, ThinkingLevel.HIGH);
    assert(!("thinkingBudget" in highConfig));

    const lowConfig = geminiThinkingConfig(ThinkingLevel.LOW);
    assert(lowConfig.includeThoughts === true);
    assertEquals(lowConfig.thinkingLevel, ThinkingLevel.LOW);
    assert(!("thinkingBudget" in lowConfig));

    const disabledConfig = geminiThinkingConfig(ThinkingLevel.LOW, false);
    assertEquals(disabledConfig.includeThoughts, false);
    assertEquals(disabledConfig.thinkingLevel, ThinkingLevel.LOW);
    assert(!("thinkingBudget" in disabledConfig));
  },
);

Deno.test(
  "geminiGenJsonFromConvo sets includeThoughts false when disableThinking is true",
  async () => {
    let capturedReq: GenerateContentParameters | undefined;
    const fakeGenerateContent = (req: GenerateContentParameters) => {
      capturedReq = req;
      return Promise.resolve({
        candidates: [{ finishReason: "STOP" }],
        text: '{"ok": true}',
      } as unknown as GenerateContentResponse);
    };

    const result = await pipe(
      injectGeminiGenerateContent(fakeGenerateContent),
    )(() =>
      geminiGenJsonFromConvo(
        { tier: "flash", disableThinking: true },
        [{ role: "user", content: "hello" }],
        z.object({ ok: z.boolean() }),
      )
    )();

    assertEquals(result, { ok: true });
    assertEquals(capturedReq?.config?.thinkingConfig?.includeThoughts, false);
    assertEquals(
      capturedReq?.config?.thinkingConfig?.thinkingLevel,
      ThinkingLevel.LOW,
    );
  },
);

Deno.test(
  "geminiGenJsonFromConvo sets includeThoughts false when tier is lite",
  async () => {
    let capturedReq: GenerateContentParameters | undefined;
    const fakeGenerateContent = (req: GenerateContentParameters) => {
      capturedReq = req;
      return Promise.resolve({
        candidates: [{ finishReason: "STOP" }],
        text: '{"ok": true}',
      } as unknown as GenerateContentResponse);
    };

    const result = await pipe(
      injectGeminiGenerateContent(fakeGenerateContent),
    )(() =>
      geminiGenJsonFromConvo(
        { tier: "lite" },
        [{ role: "user", content: "hello" }],
        z.object({ ok: z.boolean() }),
      )
    )();

    assertEquals(result, { ok: true });
    assertEquals(capturedReq?.config?.thinkingConfig?.includeThoughts, false);
    assertEquals(
      capturedReq?.config?.thinkingConfig?.thinkingLevel,
      ThinkingLevel.LOW,
    );
  },
);

Deno.test(
  "dynamicCallModel routes standard queries to ThinkingLevel.LOW via decision model without caller config",
  async () => {
    let capturedThinkingLevel: ThinkingLevel | undefined;
    const history: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "What parties are there tonight?",
      }),
    ];

    await injectCallModel((_events) =>
      Promise.resolve([ownUtteranceTurn("Here are some parties.")])
    )(() =>
      injectCallModelWrapper(({ thinkingLevel, inner }) => {
        capturedThinkingLevel = thinkingLevel;
        return inner;
      })(() =>
        agentDeps(history)(runAgent)({
          maxIterations: 1,
          prompt: "You are an events guide.",
          tools: [],
          timezoneIANA: "UTC",
        })
      )()
    )();

    assertEquals(capturedThinkingLevel, ThinkingLevel.LOW);
  },
);

Deno.test(
  "thinkingLevel is conversation state: starts at ThinkingLevel.LOW without blocking on decision model, updates in background to HIGH, and transitions back to LOW",
  async () => {
    const delay = (ms: number) =>
      new Promise((resolve) => setTimeout(resolve, ms));
    const capturedLevels: (ThinkingLevel | undefined)[] = [];
    const capturedHistories: HistoryEvent[][] = [];
    let call1StartMs = 0;
    let nextDecisionChoice: "low" | "high" = "high";
    const DECISION_DELAY_MS = 250;

    const mockDecisionCaller = async (
      _state: unknown,
      // deno-lint-ignore no-explicit-any
      questions: Record<string, any>,
    ): Promise<Record<string, DecisionAnswer>> => {
      if ("thinking_level" in questions) {
        await delay(DECISION_DELAY_MS);
        return {
          thinking_level: {
            type: "choice",
            choice: nextDecisionChoice,
          },
        };
      }
      return {};
    };

    let callCount = 0;
    const fakeCallModel = (events: HistoryEvent[]): Promise<HistoryEvent[]> => {
      callCount++;
      capturedHistories.push(events);
      if (callCount === 1) {
        call1StartMs = performance.now();
        return Promise.resolve([
          toolUseTurn({ name: "WaitTool", args: {} }),
        ]);
      }
      return Promise.resolve([ownUtteranceTurn(`Reply ${callCount}`)]);
    };

    const history: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Build a distributed consensus engine.",
      }),
    ];

    const runWithDeps = pipe(
      injectDecisionModel(mockDecisionCaller),
      injectCallModel(fakeCallModel),
      injectCallModelWrapper(({ thinkingLevel, inner }) => {
        capturedLevels.push(thinkingLevel);
        return inner;
      }),
      agentDeps(history),
    );

    const t0 = performance.now();
    await runWithDeps(() =>
      runAgent({
        maxIterations: 3,
        prompt: "You are a coding assistant.",
        tools: [
          {
            name: "WaitTool",
            description: "Waits for background work",
            parameters: z.object({}),
            handler: async () => {
              await delay(DECISION_DELAY_MS + 80);
              return "done";
            },
          },
        ],
        timezoneIANA: "UTC",
      })
    )();

    const call1LatencyMs = call1StartMs - t0;
    assert(
      call1LatencyMs < 100,
      `Expected iteration 1 not to block on 250ms decision model (< 100ms), but took ${
        Math.round(call1LatencyMs)
      }ms`,
    );

    // Iteration 1 starts at default ThinkingLevel.LOW; Iteration 2 picks up background transition to HIGH
    assertEquals(capturedLevels, [ThinkingLevel.LOW, ThinkingLevel.HIGH]);

    // Synthetic thinking level state events must not leak into model-facing preparedHistory
    for (const modelEvents of capturedHistories) {
      assert(
        !modelEvents.some((e) => e.id.startsWith("auto-thinking-")),
        "Synthetic thinking state events must be stripped before calling model",
      );
    }

    // Turn 2: user asks a simple greeting -> Turn 2 starts at current state (HIGH), background updates to LOW for Turn 3
    nextDecisionChoice = "low";
    history.push(
      participantUtteranceTurn({
        name: "user",
        text: "Thanks, what time is it?",
      }),
    );

    await runWithDeps(() =>
      runAgent({
        maxIterations: 1,
        prompt: "You are a coding assistant.",
        tools: [],
        timezoneIANA: "UTC",
      })
    )();
    await delay(DECISION_DELAY_MS + 50);

    // Turn 3: now uses the updated ThinkingLevel.LOW state
    history.push(
      participantUtteranceTurn({
        name: "user",
        text: "Cool, bye!",
      }),
    );
    await runWithDeps(() =>
      runAgent({
        maxIterations: 1,
        prompt: "You are a coding assistant.",
        tools: [],
        timezoneIANA: "UTC",
      })
    )();
    await delay(DECISION_DELAY_MS + 50);

    assertEquals(capturedLevels, [
      ThinkingLevel.LOW,
      ThinkingLevel.HIGH,
      ThinkingLevel.HIGH,
      ThinkingLevel.LOW,
    ]);
  },
);
