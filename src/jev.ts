import { context, type Injection, type Injector } from "@uri/inject";
import { pipe } from "gamla";
import { cache } from "rmmbr";
import { makeCache } from "./cacher.ts";
import {
  decideCleanupWithDecisionModel,
  decideSkillsWithDecisionModel,
  type DecisionAnswer,
  type DecisionQuestion,
  extractCandidateToolEpisodes,
  formatAgentStateForDecisionModel,
  type PastToolEpisode,
} from "./decisionModel.ts";
import {
  decisionModelSelectionCriteria,
  decisionModelSelectionInstructions,
  type ModelTier,
} from "./utils.ts";
import { injectRespanToken } from "./respan.ts";

export {
  decideCleanupWithDecisionModel as decideCleanupWithJev,
  decideSkillsWithDecisionModel as decideSkillsWithJev,
  extractCandidateToolEpisodes,
  formatAgentStateForDecisionModel as formatAgentStateForJev,
  type PastToolEpisode,
};

export const jevApiUrl = "https://api.typesafe.ai/v1/systemone";
const rmmbrUrl = "https://rmmbr.net";

const jevToken: Injection<() => string | undefined> = context(
  (): string | undefined => Deno.env.get("JEV_API_KEY"),
);

export const accessJevToken = jevToken.access;
export const injectJevToken = (token: string): Injector => {
  const injectJev = jevToken.inject(() => token);
  if (token === "") {
    return pipe(injectRespanToken(""), injectJev);
  }
  return injectJev;
};

export const jevModelSelectionInstructions = decisionModelSelectionInstructions;

export const jevModelSelectionCriteria = decisionModelSelectionCriteria;

const rawCallJev = async (
  token: string,
  state: string | Record<string, unknown> | unknown[],
): Promise<ModelTier> => {
  const response = await fetch(jevApiUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({
      model: "jev-latest",
      state,
      questions: {
        model_selection: {
          type: "choice",
          instructions: jevModelSelectionInstructions,
          criteria: jevModelSelectionCriteria,
        },
      },
    }),
    signal: AbortSignal.timeout(2500),
  });

  if (!response.ok) return "flash";
  const data = await response.json();
  const choice = data?.answers?.model_selection?.choice;
  return choice === "lite" ? "lite" : "flash";
};

const inMemoryJevCache = new Map<
  string,
  { tier: ModelTier; expiresAt: number }
>();
const memoryTtlMs = 60 * 60 * 1000;

const getRmmbrJevCacher = () => {
  const token = Deno.env.get("RMMBR_TOKEN");
  return token
    ? cache({
      cacheId: "jev-model-route-v5",
      ttl: 60 * 60 * 24 * 7,
      url: rmmbrUrl,
      token,
    })
    : undefined;
};

let rmmbrJevCaller:
  | ((token: string, state: string) => Promise<ModelTier>)
  | undefined;

export const routeTaskWithJev = async (
  state: string | Record<string, unknown> | unknown[],
): Promise<ModelTier> => {
  const token = accessJevToken();
  if (!token) return "flash";

  const serializedState = typeof state === "string"
    ? state
    : JSON.stringify(state);
  const memCached = inMemoryJevCache.get(serializedState);
  if (memCached && Date.now() < memCached.expiresAt) {
    return memCached.tier;
  }

  try {
    if (!rmmbrJevCaller) {
      const cacher = getRmmbrJevCacher();
      rmmbrJevCaller = cacher
        ? cacher((t: string, s: string) => rawCallJev(t, s))
        : (t: string, s: string) => rawCallJev(t, s);
    }
    const tier = await rmmbrJevCaller(token, serializedState);
    inMemoryJevCache.set(serializedState, {
      tier,
      expiresAt: Date.now() + memoryTtlMs,
    });
    return tier;
  } catch {
    return "flash";
  }
};

let rmmbrDecisionCaller:
  | ((
    token: string,
    payload: string,
  ) => Promise<Record<string, DecisionAnswer>>)
  | undefined;

const rawCallJevDecision = async (
  token: string,
  payload: string,
): Promise<Record<string, DecisionAnswer>> => {
  const response = await fetch(jevApiUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: payload,
    signal: AbortSignal.timeout(10000),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Jev API error (${response.status}): ${errorText}`);
  }
  const data = await response.json();
  return data?.answers ?? {};
};

const inMemoryDecisionCache = new Map<
  string,
  { answers: Record<string, DecisionAnswer>; expiresAt: number }
>();

const getRmmbrDecisionCacher = () => {
  const token = Deno.env.get("RMMBR_TOKEN");
  return token
    ? cache({
      cacheId: "jev-decision-v1",
      ttl: 60 * 60 * 24 * 30,
      url: rmmbrUrl,
      token,
      customKeyFn: (_token: string, payload: string) => payload,
    })
    : undefined;
};

export const callJevDecisionModel = async (
  state: unknown,
  questions: Record<string, DecisionQuestion>,
): Promise<Record<string, DecisionAnswer>> => {
  const token = accessJevToken();
  if (!token) throw new Error("No Jev token available");

  const payload = JSON.stringify({
    model: "jev-latest",
    state,
    questions,
  });

  const memCached = inMemoryDecisionCache.get(payload);
  if (memCached && Date.now() < memCached.expiresAt) {
    return memCached.answers;
  }

  if (!rmmbrDecisionCaller) {
    const rmmbrCacher = getRmmbrDecisionCacher();
    const injected = makeCache("jev-decision-v1");
    const base = rmmbrCacher
      ? rmmbrCacher(rawCallJevDecision)
      : rawCallJevDecision;
    rmmbrDecisionCaller = injected(base);
  }

  const answers = await rmmbrDecisionCaller(token, payload);
  inMemoryDecisionCache.set(payload, {
    answers,
    expiresAt: Date.now() + memoryTtlMs,
  });
  return answers;
};
