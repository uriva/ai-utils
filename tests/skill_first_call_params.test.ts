import { assert } from "@std/assert";
import { z } from "zod/v4";
import { runAgent } from "../mod.ts";
import {
  type AgentSpec,
  type CallModel,
  type CallModelWrapper,
  type HistoryEvent,
  injectCallModel,
  injectCallModelWrapper,
  ownUtteranceTurn,
  participantUtteranceTurn,
  runCommandToolName,
  type Skill,
  skillAutoLoadMarker,
} from "../src/agent.ts";
import {
  agentDeps,
  noopRewriteHistory,
  runForAllProviders,
} from "../test_helpers.ts";

// Effect contract for skill tool usage (strategy-agnostic):
// 1. No parameter validation error ever surfaces to the model, even when its
//    first attempt at an unlearned skill tool uses blindly guessed params.
// 2. The tool handler never receives params failing its own schema.
// 3. The base context stays compact: the inactive-skills listing carries each
//    tool's parameter signature (names, types, optionality — no parameter
//    descriptions), while full schemas and instructions stay deferred to
//    skill activation.
// 4. A schema-valid first-touch guess executes immediately: no auto-load
//    round trip is spent when the model already guessed correctly.
// 5. On the rare residual gate hit, the auto-load payload is sanitized out of
//    model-visible history once the skill is active (its instructions and
//    schemas are already re-sent via the system prompt).
const exactLineParam = "ingredientLine";
const pantryCodeParam = "pantryCode";
const toolName = "find_recipe";
const recipeResult = "grandma's pancakes";
const invalidParamsMarker = "Invalid parameters";
const recipesInstructions = "Use the tools to search the recipe database.";
const exactLineParamDescription =
  "The exact ingredient line to search for, verbatim";
const guessedWrongParams = { query: "2 cups of flour" };
const validParams = {
  [pantryCodeParam]: "PX-41",
  [exactLineParam]: "2 cups of flour",
};

const findRecipeParams = z.object({
  [pantryCodeParam]: z.string().describe("A pantry code from the pantry skill"),
  [exactLineParam]: z.string().describe(exactLineParamDescription),
  maxResults: z.number().default(5),
});

const recipesSkill = (record: (args: unknown) => void): Skill => ({
  name: "recipes",
  description: "Search a recipe database",
  instructions: recipesInstructions,
  tools: [{
    name: toolName,
    description: "Find recipes containing an exact ingredient line.",
    parameters: findRecipeParams,
    handler: (args: unknown) => {
      record(args);
      return Promise.resolve(recipeResult);
    },
  }],
});

const pantrySkill: Skill = {
  name: "pantry",
  description: "Inspect pantry contents",
  instructions: "Use the tools to inspect pantry contents.",
  tools: [{
    name: "check_stock",
    description: "List the current stock of a pantry.",
    parameters: z.object({ [pantryCodeParam]: z.string() }),
    handler: () => Promise.resolve("flour: 2kg, sugar: 1kg, eggs: 12"),
  }],
};

const cookingPrompt =
  "You are a cooking assistant. Serve the user's request immediately using your tools — be direct, do not waste steps.";
// Mirrors production bots whose prompts push direct invocation: the model
// calls run_command straight away instead of learn_skill first, so a listing
// without signatures forces a blind parameter guess.
const directInvocationPrompt =
  "You are a TV assistant. Serve the user's request immediately — invoke skill tools directly with run_command, do not waste steps loading skills.";

const scenarioSpec = (prompt: string) => (skills: Skill[]): AgentSpec => ({
  maxIterations: 8,
  tools: [],
  skills,
  prompt,
  rewriteHistory: noopRewriteHistory,
  timezoneIANA: "UTC",
});

const paramErrors = (history: HistoryEvent[]) =>
  history.filter((e): e is Extract<HistoryEvent, { type: "tool_result" }> =>
    e.type === "tool_result" && e.result.includes(invalidParamsMarker)
  );

const assertHealthySkillUsage = (
  history: HistoryEvent[],
  received: unknown[],
) => {
  assert(
    paramErrors(history).length === 0,
    `parameter errors surfaced: ${
      JSON.stringify(paramErrors(history).map((e) => e.result))
    }`,
  );
  assert(
    received.every((a) => findRecipeParams.safeParse(a).success),
    `tool handler received params failing its own schema: ${
      JSON.stringify(received)
    }`,
  );
};

const capturePrompts = (sink: string[]): CallModelWrapper =>
(
  { systemPrompt, inner },
) => {
  sink.push(systemPrompt);
  return inner;
};

const reportTokenCost = (prompts: string[]) =>
  console.log(
    `[token-cost] prompt tokens per model call: ${
      prompts.map((p) => Math.ceil(p.length / 4)).join(", ")
    }`,
  );

const promptTokens = (prompt: string) => Math.ceil(prompt.length / 4);

// Measured with compact signatures in the inactive listing: base 90, total
// 212. Budgets split that from a schema-dumping listing (full parameter
// descriptions); bump only with a conscious decision.
const basePromptTokenBudget = 100;
const correctGuessTotalTokenBudget = 250;

// Gemini's history prep strips tool_calls lacking a thoughtSignature from the
// model-visible history, so a reactive fake must carry one (queue-driven fakes
// that ignore their input don't care).
const correctGuessCall = (id: string, timestamp: number): HistoryEvent => ({
  type: "tool_call",
  isOwn: true,
  id,
  timestamp,
  name: runCommandToolName,
  parameters: {
    command: `recipes/${toolName}`,
    params: { [pantryCodeParam]: "PX-41", [exactLineParam]: "2 cups of flour" },
    spinnerText: "Finding recipes...",
  },
  modelMetadata: { type: "gemini", thoughtSignature: "sig", responseId: id },
});

// Model-visible history filters out events with stale timestamps, so fake
// calls must carry fresh ones (unlike queue-driven fakes that ignore input).
const retryUntilExecuted: CallModel = (events) => {
  const executed = events.some((e) =>
    e.type === "tool_result" && e.result.includes(recipeResult)
  );
  const calls =
    events.filter((e) =>
      e.type === "tool_call" && e.name === runCommandToolName
    ).length;
  return Promise.resolve([
    executed
      ? ownUtteranceTurn("done")
      : correctGuessCall(`guess-${calls + 1}`, Date.now() + calls),
  ]);
};

Deno.test(
  "correct first-touch guess executes immediately within token budget",
  async () => {
    const received: unknown[] = [];
    const prompts: string[] = [];
    const history: HistoryEvent[] = [participantUtteranceTurn({
      name: "user",
      text:
        "Find a recipe with the exact ingredient line '2 cups of flour' in pantry PX-41.",
    })];
    await injectCallModel(retryUntilExecuted)(() =>
      injectCallModelWrapper(capturePrompts(prompts))(() =>
        agentDeps(history)(runAgent)(
          scenarioSpec(cookingPrompt)([recipesSkill((a) => received.push(a))]),
        )
      )()
    )();
    reportTokenCost(prompts);
    assertHealthySkillUsage(history, received);
    assert(
      received.length === 1,
      `tool should execute exactly once, got ${received.length} executions`,
    );
    assert(
      promptTokens(prompts[0]) <= basePromptTokenBudget,
      `base prompt costs ${
        promptTokens(prompts[0])
      } tokens (budget ${basePromptTokenBudget}): unlearned tool schemas are creeping into the base context`,
    );
    const totalTokens = prompts.reduce((sum, p) => sum + promptTokens(p), 0);
    assert(
      totalTokens <= correctGuessTotalTokenBudget,
      `correct first-guess scenario costs ${totalTokens} prompt tokens (budget ${correctGuessTotalTokenBudget}, per call: ${
        prompts.map(promptTokens).join(", ")
      }): a schema-valid first-touch guess must execute immediately, without an auto-load round trip`,
    );
  },
);

Deno.test(
  "unlearned skill tool call surfaces no parameter error, handler gets only valid params, base context stays compact",
  async () => {
    const received: unknown[] = [];
    const prompts: string[] = [];
    const fakeCalls: HistoryEvent[] = [
      {
        type: "tool_call",
        isOwn: true,
        id: "call-1",
        timestamp: 1,
        name: runCommandToolName,
        parameters: {
          command: `recipes/${toolName}`,
          params: guessedWrongParams,
          spinnerText: "Finding recipes...",
        },
      },
      {
        type: "tool_call",
        isOwn: true,
        id: "call-2",
        timestamp: 2,
        name: runCommandToolName,
        parameters: {
          command: `recipes/${toolName}`,
          params: validParams,
          spinnerText: "Finding recipes...",
        },
      },
    ];
    const fakeModel: CallModel = () =>
      Promise.resolve(
        fakeCalls.length > 0
          ? [fakeCalls.shift() as HistoryEvent]
          : [ownUtteranceTurn("done")],
      );
    const history: HistoryEvent[] = [participantUtteranceTurn({
      name: "user",
      text:
        "Find a recipe with the exact ingredient line '2 cups of flour' in pantry PX-41.",
    })];
    await injectCallModel(fakeModel)(() =>
      injectCallModelWrapper(capturePrompts(prompts))(() =>
        agentDeps(history)(runAgent)(
          scenarioSpec(cookingPrompt)([recipesSkill((a) => received.push(a))]),
        )
      )()
    )();
    reportTokenCost(prompts);
    assertHealthySkillUsage(history, received);
    assert(
      prompts.length > 0 && prompts[0].includes(exactLineParam),
      "the inactive-skills listing must carry each tool's parameter signature so first-touch guesses are schema-valid",
    );
    assert(
      !prompts[0].includes(recipesInstructions) &&
        !prompts[0].includes(exactLineParamDescription),
      "base context must stay compact: full schemas (parameter descriptions) and instructions stay deferred to skill activation",
    );
  },
);

runForAllProviders(
  "agent uses skill tools in a multi-step flow without parameter errors",
  async (runAgentWithProvider) => {
    const received: unknown[] = [];
    const history: HistoryEvent[] = [participantUtteranceTurn({
      name: "user",
      text:
        "Check the stock of pantry PX-41. If it has flour, then find a recipe with the exact ingredient line '2 cups of flour'.",
    })];
    await agentDeps(history)(runAgentWithProvider)(
      scenarioSpec(cookingPrompt)([
        pantrySkill,
        recipesSkill((a) => received.push(a)),
      ]),
    );
    assertHealthySkillUsage(history, received);
    assert(
      received.length > 0,
      `recipe tool never executed. History: ${
        JSON.stringify(history, null, 2)
      }`,
    );
  },
);

const runCommandCall = (
  id: string,
  timestamp: number,
  params: Record<string, unknown>,
): HistoryEvent => ({
  type: "tool_call",
  isOwn: true,
  id,
  timestamp,
  name: runCommandToolName,
  parameters: {
    command: `recipes/${toolName}`,
    params,
    spinnerText: "Finding recipes...",
  },
  modelMetadata: { type: "gemini", thoughtSignature: "sig", responseId: id },
});

// Once the gate fires, the skill is active and its instructions + schemas are
// already re-sent via the system prompt on every subsequent model call — the
// full auto-load payload must not ALSO stay in model-visible history, where it
// would be re-sent per call for the rest of the conversation.
Deno.test(
  "auto-load gate payload does not persist in model-visible history once the skill is active",
  async () => {
    const seenByModel: HistoryEvent[][] = [];
    const reactiveFake: CallModel = (events) => {
      seenByModel.push(events);
      const n = seenByModel.length;
      if (n === 1) {
        return Promise.resolve([
          runCommandCall(`wrong-${n}`, Date.now(), guessedWrongParams),
        ]);
      }
      if (n === 2) {
        return Promise.resolve([
          runCommandCall(`right-${n}`, Date.now() + 1, validParams),
        ]);
      }
      return Promise.resolve([ownUtteranceTurn("done")]);
    };
    const history: HistoryEvent[] = [participantUtteranceTurn({
      name: "user",
      text:
        "Find a recipe with the exact ingredient line '2 cups of flour' in pantry PX-41.",
    })];
    await injectCallModel(reactiveFake)(() =>
      agentDeps(history)(runAgent)(
        scenarioSpec(cookingPrompt)([recipesSkill(() => {})]),
      )
    )();
    const seenAfterGate = seenByModel.slice(1);
    assert(
      seenAfterGate.length > 0,
      "model was never re-invoked after the gate",
    );
    const seenText = JSON.stringify(seenAfterGate);
    assert(
      !seenText.includes(recipesInstructions) &&
        !seenText.includes(skillAutoLoadMarker),
      "auto-load payload (instructions + schemas) persists in model-visible history even though the active skill already carries them in the system prompt",
    );
  },
);

// The inactive-skills listing must show each tool's parameter signature so the
// model's first touch is schema-valid and executes immediately (production
// friction: first-touch guesses repeatedly missed required param names, paying
// an auto-load round trip per skill). The param names here are deliberately
// not suggested by the tool description, mirroring the production failures.
const showsSkill = (record: (args: unknown) => void): Skill => ({
  name: "shows",
  description: "Search a TV episode archive",
  instructions: "Use the tools to search the TV episode archive.",
  tools: [{
    name: "find_episode",
    description: "Find TV episodes containing a spoken line.",
    parameters: z.object({
      title: z.string(),
      phrase: z.string(),
    }),
    handler: (args: unknown) => {
      record(args);
      return Promise.resolve("season 2, episode 7");
    },
  }],
});

runForAllProviders(
  "first touch of an unlearned skill tool executes immediately, without an auto-load round trip",
  async (runAgentWithProvider) => {
    const received: unknown[] = [];
    const history: HistoryEvent[] = [participantUtteranceTurn({
      name: "user",
      text:
        "In the show Autumn Alley, find the episode where someone says 'the gazebo is on fire'.",
    })];
    await agentDeps(history)(runAgentWithProvider)(
      scenarioSpec(directInvocationPrompt)([
        showsSkill((a) => received.push(a)),
      ]),
    );
    assert(
      paramErrors(history).length === 0,
      `parameter errors surfaced: ${
        JSON.stringify(paramErrors(history).map((e) => e.result))
      }`,
    );
    assert(
      received.length === 1,
      `tool should execute exactly once, got ${received.length} executions. History: ${
        JSON.stringify(history, null, 2)
      }`,
    );
    const gateHits = history.filter((e): e is Extract<HistoryEvent, {
      type: "tool_result";
    }> => e.type === "tool_result" && e.result.includes(skillAutoLoadMarker));
    assert(
      gateHits.length === 0,
      "first touch hit the auto-load gate: the model had to guess parameters blindly because the skills listing hides tool signatures",
    );
  },
);
