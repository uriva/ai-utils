import { context, type Injection, type Injector } from "@uri/inject";
import { conditionalRetry, empty } from "gamla";
import { cache } from "rmmbr";
import { ThinkingLevel } from "@google/genai";
import { isCustomCacherInjected, makeCache } from "./cacher.ts";
import {
  type ChoiceDecisionAnswer,
  decideCleanupWithDecisionModel,
  decideSkillsWithDecisionModel,
  type DecisionAnswer,
  type DecisionQuestion,
  extractCandidateToolEpisodes,
  formatAgentStateForDecisionModel,
  type NoulDecisionAnswer,
  type PastToolEpisode,
  routeThinkingLevel,
  type ScoreDecisionAnswer,
} from "./decisionModel.ts";
import { accessOptionalGeminiToken } from "./gemini.ts";
import {
  decisionThinkingLevelCriteria,
  decisionThinkingLevelInstructions,
  isServerError,
  isTransientFetchError,
  type ModelTier,
} from "./utils.ts";

export {
  decideCleanupWithDecisionModel as decideCleanupWithGemini,
  decideSkillsWithDecisionModel as decideSkillsWithGemini,
  extractCandidateToolEpisodes,
  formatAgentStateForDecisionModel as formatAgentStateForGemini,
  type PastToolEpisode,
  routeThinkingLevel,
};

export const geminiDecisionApiUrl =
  "https://generativelanguage.googleapis.com/v1beta/decisions";
export const defaultGeminiDecisionModel = "gemini-decision-1";
export const geminiDecisionCacheId = "gemini-decision-v1";
export const geminiThinkingLevelCacheId = "gemini-thinking-level-v1";
export const noGeminiDecisionTokenMessage =
  "No Gemini decision token available";
export const geminiDecisionApiErrorPrefix = "Gemini Decisions API error";

const rmmbrUrl = "https://rmmbr.net";
const memoryTtlMs = 60 * 60 * 1000;

const geminiDecisionUrlInjection: Injection<() => string> = context(
  (): string => geminiDecisionApiUrl,
);

export const accessGeminiDecisionUrl = geminiDecisionUrlInjection.access;
export const injectGeminiDecisionUrl = (url: string): Injector =>
  geminiDecisionUrlInjection.inject(() => url);

const geminiDecisionTokenOverride: Injection<() => string | undefined> =
  context((): string | undefined => undefined);

export const injectGeminiDecisionToken = (token: string): Injector =>
  geminiDecisionTokenOverride.inject(() => token);

export const accessGeminiDecisionToken = (): string | undefined => {
  const override = geminiDecisionTokenOverride.access();
  return override !== undefined
    ? override
    : (accessOptionalGeminiToken() ?? Deno.env.get("GEMINI_API_KEY"));
};

const requireGeminiDecisionToken = (): string => {
  const rawToken = accessGeminiDecisionToken();
  const token = rawToken ? String(rawToken).trim() : "";
  if (!token || token === "undefined") {
    throw new Error(noGeminiDecisionTokenMessage);
  }
  return token;
};

const resolveGeminiDecisionModel = (): string =>
  Deno.env.get("GEMINI_DECISION_MODEL") || defaultGeminiDecisionModel;

const stateToGeminiInput = (state: unknown): string => {
  if (typeof state === "string") {
    return (state.trim() ? state : " ").toWellFormed();
  }
  const serialized = JSON.stringify(state ?? null) || " ";
  return (serialized.trim() ? serialized : " ").toWellFormed();
};

const isRecord = (val: unknown): val is Record<string, unknown> =>
  typeof val === "object" && val !== null;

const isValidQuestionKey = (key: string): boolean =>
  /^[A-Za-z0-9_-]{1,64}$/.test(key);

const invalidOptionWhitespaceOrControl =
  // deno-lint-ignore no-control-regex
  /[\u0000-\u001f\u007f-\u009f\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]/;

const isValidOptionKey = (key: string): boolean =>
  key.length >= 1 &&
  key.length <= 128 &&
  key === key.trim() &&
  !invalidOptionWhitespaceOrControl.test(key) &&
  new TextEncoder().encode(key).length <= 512;

type WireNoulQuestion = {
  type: "noul";
  instructions: string;
  criteria?: {
    true_description?: string;
    false_description?: string;
  };
};

type WireChoiceQuestion = {
  type: "choice";
  instructions: string;
  criteria: Record<string, string>;
};

type WireScoreQuestion = {
  type: "score";
  instructions: string;
  criteria: string[];
};

type WireQuestion = WireNoulQuestion | WireChoiceQuestion | WireScoreQuestion;

type PreparedQuestion = {
  origKey: string;
  wireKey: string;
  wireQuestion: WireQuestion;
  optionMap?: Record<string, string>;
};

const makeSafeWireKey = (existingKeys: Set<string>, idx: number): string => {
  const candidate = `q_${idx}`;
  return existingKeys.has(candidate) ? `q_${idx}_gemini` : candidate;
};

const buildChoiceWireQuestion = (
  origKey: string,
  instructions: string | undefined,
  criteria: Record<string, string>,
): { wireQuestion: WireChoiceQuestion; optionMap?: Record<string, string> } => {
  const entries = Object.entries(criteria);
  const allValid = entries.every(([opt]) => isValidOptionKey(opt));
  if (allValid) {
    return {
      wireQuestion: {
        type: "choice",
        instructions: (instructions?.trim() || `Select ${origKey}`)
          .toWellFormed(),
        criteria: Object.fromEntries(
          entries.map(([opt, desc]) => [opt, desc.toWellFormed()]),
        ),
      },
    };
  }
  const mapped = entries.map(([opt, desc], idx) => ({
    wireOpt: `opt_${idx}`,
    origOpt: opt,
    wireDesc: (desc.trim() ? `${opt}: ${desc}` : opt).toWellFormed(),
  }));
  return {
    wireQuestion: {
      type: "choice",
      instructions: (instructions?.trim() || `Select ${origKey}`)
        .toWellFormed(),
      criteria: Object.fromEntries(
        mapped.map(({ wireOpt, wireDesc }) => [wireOpt, wireDesc]),
      ),
    },
    optionMap: Object.fromEntries(
      mapped.map(({ wireOpt, origOpt }) => [wireOpt, origOpt]),
    ),
  };
};

const prepareSingleQuestion = (existingKeys: Set<string>) =>
(
  [origKey, q]: [string, DecisionQuestion],
  idx: number,
):
  | { kind: "deterministic"; origKey: string; answer: DecisionAnswer }
  | { kind: "wire"; prepared: PreparedQuestion } => {
  if (q.type === "choice" && Object.keys(q.criteria).length <= 1) {
    const [onlyChoice = ""] = Object.keys(q.criteria);
    return {
      kind: "deterministic",
      origKey,
      answer: {
        type: "choice",
        choice: onlyChoice,
        confidence: 1,
        probabilities: onlyChoice ? { [onlyChoice]: 1 } : {},
      },
    };
  }

  const wireKey = isValidQuestionKey(origKey)
    ? origKey
    : makeSafeWireKey(existingKeys, idx);

  if (q.type === "noul") {
    const trueDesc = q.criteria?.true?.trim();
    const falseDesc = q.criteria?.false?.trim();
    return {
      kind: "wire",
      prepared: {
        origKey,
        wireKey,
        wireQuestion: {
          type: "noul",
          instructions: (q.instructions?.trim() || `Is ${origKey} true?`)
            .toWellFormed(),
          ...(trueDesc || falseDesc
            ? {
              criteria: {
                ...(trueDesc
                  ? { true_description: trueDesc.toWellFormed() }
                  : {}),
                ...(falseDesc
                  ? { false_description: falseDesc.toWellFormed() }
                  : {}),
              },
            }
            : {}),
        },
      },
    };
  }

  if (q.type === "choice") {
    const { wireQuestion, optionMap } = buildChoiceWireQuestion(
      origKey,
      q.instructions,
      q.criteria,
    );
    return {
      kind: "wire",
      prepared: { origKey, wireKey, wireQuestion, optionMap },
    };
  }

  if (q.type === "score") {
    return {
      kind: "wire",
      prepared: {
        origKey,
        wireKey,
        wireQuestion: {
          type: "score",
          instructions: (q.instructions?.trim() || `Rate ${origKey}`)
            .toWellFormed(),
          criteria: q.criteria.map((c) => (c.trim() || "-").toWellFormed()),
        },
      },
    };
  }

  throw new Error(
    `Gemini Decisions API does not support '${q.type}' questions`,
  );
};

const toNumberMap = (val: unknown): Record<string, number> | undefined =>
  isRecord(val)
    ? Object.fromEntries(
      Object.entries(val).filter((entry): entry is [string, number] =>
        typeof entry[1] === "number"
      ),
    )
    : undefined;

const toStringMap = (val: unknown): Record<string, string> | undefined =>
  isRecord(val)
    ? Object.fromEntries(
      Object.entries(val).filter((entry): entry is [string, string] =>
        typeof entry[1] === "string"
      ),
    )
    : undefined;

const remapOptionProbabilities = (
  probs: Record<string, number> | undefined,
  optionMap: Record<string, string> | undefined,
): Record<string, number> | undefined =>
  probs && optionMap
    ? Object.fromEntries(
      Object.entries(probs).map(([k, v]) => [optionMap[k] ?? k, v]),
    )
    : probs;

const parseWireAnswer = (
  prepared: PreparedQuestion,
  raw: unknown,
): DecisionAnswer | undefined => {
  if (!isRecord(raw)) return undefined;
  if (raw.type === "noul" && typeof raw.noul === "number") {
    const ans: NoulDecisionAnswer = { type: "noul", noul: raw.noul };
    return ans;
  }
  if (raw.type === "choice" && typeof raw.choice === "string") {
    const mappedChoice = prepared.optionMap?.[raw.choice] ?? raw.choice;
    const probabilities = remapOptionProbabilities(
      toNumberMap(raw.probabilities),
      prepared.optionMap,
    );
    const ans: ChoiceDecisionAnswer = {
      type: "choice",
      choice: mappedChoice,
      ...(typeof raw.confidence === "number"
        ? { confidence: raw.confidence }
        : {}),
      ...(probabilities ? { probabilities } : {}),
    };
    return ans;
  }
  if (raw.type === "score" && typeof raw.score === "number") {
    const probabilities = toNumberMap(raw.probabilities);
    const legend = toStringMap(raw.legend);
    const ans: ScoreDecisionAnswer = {
      type: "score",
      score: raw.score,
      ...(typeof raw.confidence === "number"
        ? { confidence: raw.confidence }
        : {}),
      ...(legend ? { legend } : {}),
      ...(probabilities ? { probabilities } : {}),
    };
    return ans;
  }
  return undefined;
};

const isTransientDecisionError = (error: unknown): boolean =>
  isServerError(error) || isTransientFetchError(error);

const postGeminiDecisionsOnce = (timeoutMs: number) =>
async (
  token: string,
  payload: string,
): Promise<Record<string, unknown>> => {
  const response = await fetch(accessGeminiDecisionUrl(), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-goog-api-key": token,
    },
    body: payload,
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!response.ok) {
    const errorText = await response.text();
    const err = Object.assign(
      new Error(
        `${geminiDecisionApiErrorPrefix} (${response.status}): ${errorText}`,
      ),
      { status: response.status },
    );
    throw err;
  }
  const parsed: unknown = await response.json();
  return isRecord(parsed) && isRecord(parsed.answers) ? parsed.answers : {};
};

const postGeminiDecisions = (timeoutMs: number) =>
  conditionalRetry(isTransientDecisionError)(
    300,
    2,
    postGeminiDecisionsOnce(timeoutMs),
  );

const inMemoryDecisionCache = new Map<
  string,
  { answers: Record<string, unknown>; expiresAt: number }
>();

const rmmbrDecisionCallers = new Map<
  string,
  (token: string, payload: string) => Promise<Record<string, unknown>>
>();

const getRmmbrDecisionCaller = (rmmbrToken: string) => {
  const existing = rmmbrDecisionCallers.get(rmmbrToken);
  if (existing) return existing;
  const wrapped = cache({
    cacheId: geminiDecisionCacheId,
    ttl: 60 * 60 * 24 * 30,
    url: rmmbrUrl,
    token: rmmbrToken,
    customKeyFn: (_token: string, payload: string) => payload,
  })(postGeminiDecisions(10000));
  rmmbrDecisionCallers.set(rmmbrToken, wrapped);
  return wrapped;
};

const callCachedWireDecisions = async (
  token: string,
  payload: string,
): Promise<Record<string, unknown>> => {
  const memKey = `${token}\n${payload}`;
  const memCached = inMemoryDecisionCache.get(memKey);
  if (memCached && Date.now() < memCached.expiresAt) {
    return memCached.answers;
  }
  const rmmbrToken = Deno.env.get("RMMBR_TOKEN");
  const rawAnswers = isCustomCacherInjected()
    ? await makeCache(geminiDecisionCacheId, 60 * 60 * 24 * 30)(
      (p: string) => postGeminiDecisions(10000)(token, p),
    )(payload)
    : rmmbrToken
    ? await getRmmbrDecisionCaller(rmmbrToken)(token, payload)
    : await postGeminiDecisions(10000)(token, payload);
  inMemoryDecisionCache.set(memKey, {
    answers: rawAnswers,
    expiresAt: Date.now() + memoryTtlMs,
  });
  return rawAnswers;
};

export const callGeminiDecisionModel = async (
  state: unknown,
  questions: Record<string, DecisionQuestion>,
): Promise<Record<string, DecisionAnswer>> => {
  const token = requireGeminiDecisionToken();
  const entries = Object.entries(questions);
  if (empty(entries)) return {};

  const existingKeys = new Set(entries.map(([k]) => k));
  const partitioned = entries.map(prepareSingleQuestion(existingKeys));
  const deterministicEntries = partitioned.flatMap((item) =>
    item.kind === "deterministic" ? [[item.origKey, item.answer] as const] : []
  );
  const wireItems = partitioned.flatMap((item) =>
    item.kind === "wire" ? [item.prepared] : []
  );

  if (empty(wireItems)) {
    return Object.fromEntries(deterministicEntries);
  }

  const payload = JSON.stringify({
    model: resolveGeminiDecisionModel(),
    input: stateToGeminiInput(state),
    questions: Object.fromEntries(
      wireItems.map(({ wireKey, wireQuestion }) => [wireKey, wireQuestion]),
    ),
  }).toWellFormed();

  const rawAnswers = await callCachedWireDecisions(token, payload);
  const wireAnswers = wireItems.flatMap((prepared) => {
    const parsed = parseWireAnswer(prepared, rawAnswers[prepared.wireKey]);
    return parsed ? [[prepared.origKey, parsed] as const] : [];
  });

  return Object.fromEntries([...deterministicEntries, ...wireAnswers]);
};

const rawCallGeminiRoute = async (
  token: string,
  payload: string,
): Promise<ThinkingLevel> => {
  const rawAnswers = await postGeminiDecisions(5000)(token, payload);
  const thinkingAns = rawAnswers.thinking_level;
  const modelAns = rawAnswers.model_selection;
  const choice = isRecord(thinkingAns) && typeof thinkingAns.choice === "string"
    ? thinkingAns.choice
    : isRecord(modelAns) && typeof modelAns.choice === "string"
    ? modelAns.choice
    : undefined;
  return choice === "low" || choice === "lite"
    ? ThinkingLevel.LOW
    : ThinkingLevel.HIGH;
};

const inMemoryRouteCache = new Map<
  string,
  { level: ThinkingLevel; expiresAt: number }
>();

const rmmbrRouteCallers = new Map<
  string,
  (token: string, payload: string) => Promise<ThinkingLevel>
>();

const getRmmbrRouteCaller = (rmmbrToken: string) => {
  const existing = rmmbrRouteCallers.get(rmmbrToken);
  if (existing) return existing;
  const wrapped = cache({
    cacheId: geminiThinkingLevelCacheId,
    ttl: 60 * 60 * 24 * 7,
    url: rmmbrUrl,
    token: rmmbrToken,
    customKeyFn: (_token: string, payload: string) => payload,
  })(rawCallGeminiRoute);
  rmmbrRouteCallers.set(rmmbrToken, wrapped);
  return wrapped;
};

export const routeThinkingLevelWithGemini = async (
  state: string | Record<string, unknown> | unknown[],
): Promise<ThinkingLevel> => {
  const token = requireGeminiDecisionToken();
  const payload = JSON.stringify({
    model: resolveGeminiDecisionModel(),
    input: stateToGeminiInput(state),
    questions: {
      thinking_level: {
        type: "choice",
        instructions: decisionThinkingLevelInstructions,
        criteria: decisionThinkingLevelCriteria,
      },
    },
  }).toWellFormed();

  const memKey = `${token}\n${payload}`;
  const memCached = inMemoryRouteCache.get(memKey);
  if (memCached && Date.now() < memCached.expiresAt) {
    return memCached.level;
  }

  const rmmbrToken = Deno.env.get("RMMBR_TOKEN");
  const level = isCustomCacherInjected()
    ? await makeCache(geminiThinkingLevelCacheId, 60 * 60 * 24 * 7)(
      (p: string) => rawCallGeminiRoute(token, p),
    )(payload)
    : rmmbrToken
    ? await getRmmbrRouteCaller(rmmbrToken)(token, payload)
    : await rawCallGeminiRoute(token, payload);

  inMemoryRouteCache.set(memKey, {
    level,
    expiresAt: Date.now() + memoryTtlMs,
  });
  return level;
};

export const routeTaskWithGemini = async (
  state: string | Record<string, unknown> | unknown[],
): Promise<ModelTier> => {
  const level = await routeThinkingLevelWithGemini(state);
  return level === ThinkingLevel.LOW ? "lite" : "flash";
};
