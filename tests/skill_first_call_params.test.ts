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
// 3. The base context stays minimal: unlearned tool schemas are not dumped
//    into the system prompt.
// 4. A schema-valid first-touch guess executes immediately: no auto-load
//    round trip is spent when the model already guessed correctly.
const exactLineParam = "ingredientLine";
const pantryCodeParam = "pantryCode";
const toolName = "find_recipe";
const recipeResult = "grandma's pancakes";
const invalidParamsMarker = "Invalid parameters";

const findRecipeParams = z.object({
  [pantryCodeParam]: z.string().describe("A pantry code from the pantry skill"),
  [exactLineParam]: z.string().describe(
    "The exact ingredient line to search for, verbatim",
  ),
  maxResults: z.number().default(5),
});

const recipesSkill = (record: (args: unknown) => void): Skill => ({
  name: "recipes",
  description: "Search a recipe database",
  instructions: "Use the tools to search the recipe database.",
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

const scenarioSpec = (skills: Skill[]): AgentSpec => ({
  maxIterations: 8,
  tools: [],
  skills,
  prompt:
    "You are a cooking assistant. Serve the user's request immediately using your tools — be direct, do not waste steps.",
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

// Measured with the immediate-execution behavior: base 70, total 192.
// Budgets split that from the always-gate behavior (total 314) and from
// schema-dumping listings; bump only with a conscious decision.
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
          scenarioSpec([recipesSkill((a) => received.push(a))]),
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
  "unlearned skill tool call surfaces no parameter error, handler gets only valid params, base context stays minimal",
  async () => {
    const received: unknown[] = [];
    const prompts: string[] = [];
    const guessedWrongParams = { query: "2 cups of flour" };
    const validParams = {
      [pantryCodeParam]: "PX-41",
      [exactLineParam]: "2 cups of flour",
    };
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
          scenarioSpec([recipesSkill((a) => received.push(a))]),
        )
      )()
    )();
    reportTokenCost(prompts);
    assertHealthySkillUsage(history, received);
    assert(
      prompts.length > 0 && !prompts[0].includes(exactLineParam),
      "base context must stay minimal: unlearned tool schemas must not be dumped into the prompt",
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
      scenarioSpec([pantrySkill, recipesSkill((a) => received.push(a))]),
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
