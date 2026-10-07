import { context, type Injection, type Injector } from "@uri/inject";
import { empty } from "gamla";
import type { z, ZodType } from "zod/v4";
import { ThinkingLevel } from "@google/genai";
import type { HistoryEvent, ParticipantUtterance, Skill } from "./agent.ts";
import {
  accessJevToken,
  callJevDecisionModel,
  routeThinkingLevelWithJev,
} from "./jev.ts";
import {
  accessRespanToken,
  callRespanDecisionModel,
  routeThinkingLevelWithRespan,
} from "./respan.ts";
import {
  cleanActiveMemoryToolName,
  decisionModelSelectionCriteria,
  decisionModelSelectionInstructions,
  decisionThinkingLevelCriteria,
  decisionThinkingLevelInstructions,
  type ModelOpts,
  type ModelTier,
  validateAgainstSchema,
} from "./utils.ts";

export {
  decisionModelSelectionCriteria,
  decisionModelSelectionInstructions,
  decisionThinkingLevelCriteria,
  decisionThinkingLevelInstructions,
};

export type ChoiceDecisionQuestion = {
  type: "choice";
  instructions?: string;
  criteria: Record<string, string>;
};

export type NoulDecisionQuestion = {
  type: "noul";
  instructions?: string;
  criteria?: { true?: string; false?: string };
};

export type ScoreDecisionQuestion = {
  type: "score";
  instructions?: string;
  criteria: string[];
};

export type DecisionQuestion =
  | ChoiceDecisionQuestion
  | NoulDecisionQuestion
  | ScoreDecisionQuestion;

export type ChoiceDecisionAnswer = {
  type: "choice";
  choice: string;
  confidence?: number;
  probabilities?: Record<string, number>;
};

export type NoulDecisionAnswer = {
  type: "noul";
  noul: number;
};

export type ScoreDecisionAnswer = {
  type: "score";
  score: number;
  confidence?: number;
  legend?: Record<string, string>;
  probabilities?: Record<string, number>;
};

export type DecisionAnswer =
  | ChoiceDecisionAnswer
  | NoulDecisionAnswer
  | ScoreDecisionAnswer;

export type DecisionModelCaller = (
  state: unknown,
  questions: Record<string, DecisionQuestion>,
) => Promise<Record<string, DecisionAnswer>>;

export type DecisionProvider = "respan" | "jev";

const decisionProviderInjection: Injection<() => DecisionProvider | undefined> =
  context((): DecisionProvider | undefined => {
    const env = Deno.env.get("DECISION_PROVIDER");
    if (env === "jev" || env === "respan") return env;
    return undefined;
  });

export const accessDecisionProvider = decisionProviderInjection.access;
export const injectDecisionProvider = (provider: DecisionProvider): Injector =>
  decisionProviderInjection.inject(() => provider);

export const resolveDecisionProvider = (): DecisionProvider => {
  const configured = accessDecisionProvider();
  if (configured) return configured;
  if (accessRespanToken()) return "respan";
  if (accessJevToken()) return "jev";
  return "respan";
};

const decisionModelOverrideInjection: Injection<
  () => DecisionModelCaller | null
> = context((): DecisionModelCaller | null => null);

export const injectDecisionModel = (
  caller: DecisionModelCaller,
): Injector => decisionModelOverrideInjection.inject(() => caller);

export const isDecisionModelInjected = (): boolean =>
  Boolean(decisionModelOverrideInjection.access());

export const isDecisionModelAvailable = (): boolean => {
  if (decisionModelOverrideInjection.access()) return true;
  const provider = resolveDecisionProvider();
  if (provider === "jev") return Boolean(accessJevToken());
  return Boolean(accessRespanToken());
};

export const callDecisionModel = async (
  state: unknown,
  questions: Record<string, DecisionQuestion>,
): Promise<Record<string, DecisionAnswer>> => {
  const override = decisionModelOverrideInjection.access();
  if (override) {
    return override(state, questions);
  }
  const provider = resolveDecisionProvider();
  if (provider === "jev") {
    try {
      return await callJevDecisionModel(state, questions);
    } catch (err) {
      if (accessRespanToken()) {
        console.warn(
          "[decision-model] Jev failed, falling back to Respan:",
          err,
        );
        return await callRespanDecisionModel(state, questions);
      }
      throw err;
    }
  }
  try {
    return await callRespanDecisionModel(state, questions);
  } catch (err) {
    if (accessJevToken()) {
      console.warn(
        "[decision-model] Respan failed, falling back to Jev:",
        err,
      );
      return await callJevDecisionModel(state, questions);
    }
    throw err;
  }
};

const isRecord = (val: unknown): val is Record<string, unknown> =>
  typeof val === "object" && val !== null;

export const unwrapZod = (schema: unknown): unknown => {
  if (!isRecord(schema)) return schema;
  const def = isRecord(schema._def)
    ? schema._def
    : isRecord(schema.def)
    ? schema.def
    : undefined;
  if (!def) return schema;
  const inner = def.innerType ?? def.schema;
  if (inner) return unwrapZod(inner);
  return schema;
};

export const getZodTypeName = (schema: unknown): string => {
  const unwrapped = unwrapZod(schema);
  if (!isRecord(unwrapped)) return "";
  const def = isRecord(unwrapped._def)
    ? unwrapped._def
    : isRecord(unwrapped.def)
    ? unwrapped.def
    : undefined;
  const ctor = unwrapped.constructor;
  return typeof def?.typeName === "string"
    ? def.typeName
    : typeof def?.type === "string"
    ? def.type
    : isRecord(ctor) && typeof ctor.name === "string"
    ? ctor.name
    : "";
};

export const isLiteralField = (schema: unknown): boolean => {
  const t = getZodTypeName(schema);
  return t === "literal" || t === "ZodLiteral";
};

export const isStringField = (schema: unknown): boolean => {
  const t = getZodTypeName(schema);
  return t === "string" || t === "ZodString";
};

export const isDecisionField = (schema: unknown): boolean => {
  const t = getZodTypeName(schema);
  if (t === "boolean" || t === "ZodBoolean") return true;
  if (t === "enum" || t === "ZodEnum") return true;
  if (t === "literal" || t === "ZodLiteral") return true;
  if (t === "nativeEnum" || t === "ZodNativeEnum") return true;
  if (t === "union" || t === "ZodUnion") {
    const unwrapped = unwrapZod(schema);
    if (!isRecord(unwrapped)) return false;
    const def = isRecord(unwrapped.def)
      ? unwrapped.def
      : isRecord(unwrapped._def)
      ? unwrapped._def
      : undefined;
    const options = Array.isArray(unwrapped.options)
      ? unwrapped.options
      : def && Array.isArray(def.options)
      ? def.options
      : [];
    if (!empty(options) && options.every(isLiteralField)) {
      return true;
    }
  }
  return false;
};

export const fieldToDecisionQuestion = (
  name: string,
  rawField: unknown,
): DecisionQuestion | undefined => {
  if (!isRecord(rawField)) return undefined;
  const unwrapped = unwrapZod(rawField);
  if (!isRecord(unwrapped)) return undefined;

  const t = getZodTypeName(rawField);
  const description = typeof rawField.description === "string"
    ? rawField.description
    : typeof unwrapped.description === "string"
    ? unwrapped.description
    : undefined;

  if (t === "boolean" || t === "ZodBoolean") {
    return {
      type: "noul",
      instructions: description || `Is ${name} true?`,
    };
  }

  if (t === "enum" || t === "ZodEnum") {
    const options = Array.isArray(unwrapped.options)
      ? unwrapped.options
      : isRecord(unwrapped.def) && isRecord(unwrapped.def.entries)
      ? Object.keys(unwrapped.def.entries)
      : [];
    return {
      type: "choice",
      instructions: description || `Select ${name}`,
      criteria: Object.fromEntries(
        options.map((opt) => [String(opt), String(opt)]),
      ),
    };
  }

  if (t === "literal" || t === "ZodLiteral") {
    const def = isRecord(unwrapped.def) ? unwrapped.def : undefined;
    const values = def && Array.isArray(def.values) ? def.values : undefined;
    const val = String(unwrapped.value ?? values?.[0] ?? "");
    return {
      type: "choice",
      instructions: description || `Value of ${name}`,
      criteria: { [val]: val },
    };
  }

  if (t === "union" || t === "ZodUnion") {
    const def = isRecord(unwrapped.def)
      ? unwrapped.def
      : isRecord(unwrapped._def)
      ? unwrapped._def
      : undefined;
    const rawOptions = Array.isArray(unwrapped.options)
      ? unwrapped.options
      : def && Array.isArray(def.options)
      ? def.options
      : [];
    const entries = rawOptions.map((opt) => {
      if (!isRecord(opt)) return ["", ""];
      const optDef = isRecord(opt.def) ? opt.def : undefined;
      const optValues = optDef && Array.isArray(optDef.values)
        ? optDef.values
        : undefined;
      const val = String(opt.value ?? optValues?.[0] ?? "");
      const desc = typeof opt.description === "string" ? opt.description : val;
      return [val, desc];
    });
    return {
      type: "choice",
      instructions: description || `Select ${name}`,
      criteria: Object.fromEntries(entries),
    };
  }

  return undefined;
};

export const parseDecisionAnswer = (
  rawField: unknown,
  answer: DecisionAnswer | undefined,
): unknown => {
  if (!answer) return undefined;
  const unwrapped = unwrapZod(rawField);
  const t = getZodTypeName(rawField);

  if (t === "boolean" || t === "ZodBoolean") {
    if (answer.type === "noul") return answer.noul >= 0.5;
    if (answer.type === "choice") {
      return answer.choice === "true" || answer.choice === "yes";
    }
  }

  if (t === "enum" || t === "ZodEnum") {
    if (answer.type === "choice") return answer.choice;
  }

  if (t === "literal" || t === "ZodLiteral") {
    if (!isRecord(unwrapped)) return undefined;
    const def = isRecord(unwrapped.def) ? unwrapped.def : undefined;
    const values = def && Array.isArray(def.values) ? def.values : undefined;
    return unwrapped.value ?? values?.[0];
  }

  if (t === "union" || t === "ZodUnion") {
    if (answer.type === "choice" && isRecord(unwrapped)) {
      const def = isRecord(unwrapped.def)
        ? unwrapped.def
        : isRecord(unwrapped._def)
        ? unwrapped._def
        : undefined;
      const rawOptions = Array.isArray(unwrapped.options)
        ? unwrapped.options
        : def && Array.isArray(def.options)
        ? def.options
        : [];
      const matched = rawOptions.find((opt) => {
        if (!isRecord(opt)) return false;
        const optDef = isRecord(opt.def) ? opt.def : undefined;
        const optValues = optDef && Array.isArray(optDef.values)
          ? optDef.values
          : undefined;
        return String(opt.value ?? optValues?.[0]) === answer.choice;
      });
      if (isRecord(matched)) {
        const matchedDef = isRecord(matched.def) ? matched.def : undefined;
        const matchedValues = matchedDef && Array.isArray(matchedDef.values)
          ? matchedDef.values
          : undefined;
        return matched.value ?? matchedValues?.[0];
      }
      return answer.choice;
    }
  }

  if (answer.type === "choice") return answer.choice;
  if (answer.type === "noul") return answer.noul >= 0.5;
  if (answer.type === "score") return answer.score;
  return undefined;
};

const formatStateForDecision = (
  systemMsg: string,
  userMsg: unknown,
): unknown => {
  if (typeof userMsg === "string") {
    return systemMsg ? { instructions: systemMsg, content: userMsg } : userMsg;
  }
  if (isRecord(userMsg)) {
    return systemMsg && !("instructions" in userMsg) && !("prompt" in userMsg)
      ? { instructions: systemMsg, ...userMsg }
      : userMsg;
  }
  return userMsg;
};

export const decideSingleField = async <T extends ZodType>(
  systemMsg: string,
  zodType: T,
  userMsg: unknown,
): Promise<z.infer<T>> => {
  const q = fieldToDecisionQuestion("decision", zodType);
  if (!q) {
    throw new Error(
      `Cannot convert schema of type '${
        getZodTypeName(zodType)
      }' to a decision question`,
    );
  }
  const state = formatStateForDecision(systemMsg, userMsg);
  const answers = await callDecisionModel(state, {
    decision: {
      ...q,
      instructions: systemMsg || q.instructions,
    },
  });
  const parsed = parseDecisionAnswer(zodType, answers.decision);
  return validateAgainstSchema(zodType, parsed);
};

export const decideObject = async <T extends ZodType>(
  systemMsg: string,
  zodType: T,
  userMsg: unknown,
): Promise<z.infer<T>> => {
  const shape = isRecord(zodType) && isRecord(zodType.shape)
    ? (zodType.shape as Record<string, unknown>)
    : undefined;
  if (!shape) {
    throw new Error("Schema must be an object with shape for decideObject");
  }

  const questions: Record<string, DecisionQuestion> = {};
  for (const [k, field] of Object.entries(shape)) {
    const q = fieldToDecisionQuestion(k, field);
    if (!q) {
      throw new Error(`Cannot convert field '${k}' to a decision question`);
    }
    questions[k] = q;
  }

  const state = formatStateForDecision(systemMsg, userMsg);
  const answers = await callDecisionModel(state, questions);
  const result: Record<string, unknown> = {};
  for (const [k, field] of Object.entries(shape)) {
    result[k] = parseDecisionAnswer(field, answers[k]);
  }

  return validateAgainstSchema(zodType, result);
};

export function decide<T extends ZodType>(
  systemMsg: string,
  zodType: T,
): (userMsg: unknown) => Promise<z.infer<T>>;
export function decide<T extends ZodType>(
  opts: ModelOpts,
  systemMsg: string,
  zodType: T,
): (userMsg: unknown) => Promise<z.infer<T>>;
export function decide<T extends ZodType>(
  optsOrSystemMsg: ModelOpts | string,
  systemMsgOrZodType: string | T,
  maybeZodType?: T,
): (userMsg: unknown) => Promise<z.infer<T>> {
  const systemMsg = typeof optsOrSystemMsg === "string"
    ? optsOrSystemMsg
    : (systemMsgOrZodType as string);
  const zodType = typeof optsOrSystemMsg === "string"
    ? (systemMsgOrZodType as T)
    : (maybeZodType as T);

  return (userMsg: unknown): Promise<z.infer<T>> => {
    const isObject = isRecord(zodType) && isRecord(zodType.shape);
    if (isObject) {
      return decideObject(systemMsg, zodType, userMsg);
    }
    return decideSingleField(systemMsg, zodType, userMsg);
  };
}

export const genDecision = decide;

export const lastParticipantUtterance = (
  history: HistoryEvent[],
): ParticipantUtterance | undefined =>
  [...history].reverse().find((e): e is ParticipantUtterance =>
    e.type === "participant_utterance"
  );

export const verifiedToolFacts = (history: HistoryEvent[]): string => {
  const toolResults = history
    .flatMap((e) => e.type === "tool_result" ? [e.result] : [])
    .join("\n\n");
  const otherFacts = history
    .flatMap((e) => {
      if (e.type === "external_event") return [e.text];
      if (e.type === "own_thought" && typeof e.text === "string") {
        return [e.text];
      }
      return [];
    })
    .join("\n\n");
  return `${toolResults}\n\n${otherFacts}`.slice(-10000).trim();
};

export const cleanHistoryEventForDecision = (e: HistoryEvent) => {
  if (e.type === "participant_utterance") {
    return { type: e.type, ...(e.name ? { name: e.name } : {}), text: e.text };
  }
  if (e.type === "own_utterance") {
    return { type: e.type, text: e.text };
  }
  if (e.type === "tool_call") {
    return { type: e.type, name: e.name, parameters: e.parameters };
  }
  if (e.type === "tool_result") {
    return {
      type: e.type,
      ...(e.toolCallId ? { toolCallId: e.toolCallId } : {}),
      result: e.result,
    };
  }
  if (e.type === "own_thought" && typeof e.text === "string") {
    return { type: e.type, text: e.text };
  }
  if ("text" in e && typeof e.text === "string") {
    return { type: e.type, text: e.text };
  }
  return undefined;
};

export const historyEventsToDecisionJson = (
  prompt: string,
  history: HistoryEvent[],
): string =>
  JSON.stringify({
    system_prompt: prompt,
    history: history.map(cleanHistoryEventForDecision).filter(Boolean),
  });

export const eventContent = (event: HistoryEvent): string | undefined => {
  if ("text" in event && typeof event.text === "string") return event.text;
  if ("result" in event && typeof event.result === "string") {
    return event.result;
  }
  if (event.type === "tool_call") {
    return `${event.name}(${JSON.stringify(event.parameters ?? {})})`;
  }
  return undefined;
};

export const formatAgentStateForDecisionModel = (
  prompt: string,
  history: HistoryEvent[],
  tools?: { name: string }[],
): Record<string, unknown> => {
  const lastEvent = history[history.length - 1];
  const lastUserMsg = [...history]
    .reverse()
    .find((e) => e.type === "participant_utterance");
  const lastEventText = lastEvent ? eventContent(lastEvent) : undefined;
  const triggerContent = lastEventText
    ? lastEventText.slice(0, 1000)
    : (lastUserMsg && "text" in lastUserMsg &&
        typeof lastUserMsg.text === "string"
      ? lastUserMsg.text.slice(0, 1000)
      : "Empty user request");
  const triggerType = lastEvent ? lastEvent.type : "conversation_start";
  const recentEvents = history.slice(-4).map((e) => ({
    type: e.type,
    text: eventContent(e)?.slice(0, 300),
  }));
  return {
    trigger_type: triggerType,
    trigger_content: triggerContent,
    agent_role: prompt.slice(0, 500),
    tools_available: (tools ?? []).map((t) => t.name).slice(0, 15),
    recent_turns: recentEvents,
  };
};

export const formatAgentStateForRespan = formatAgentStateForDecisionModel;
export const formatAgentStateForJev = formatAgentStateForDecisionModel;

export const routeThinkingLevel = async (
  state: string | Record<string, unknown> | unknown[],
): Promise<ThinkingLevel> => {
  const override = decisionModelOverrideInjection.access();
  if (override) {
    const answers = await override(state, {
      thinking_level: {
        type: "choice",
        criteria: decisionThinkingLevelCriteria,
        instructions: decisionThinkingLevelInstructions,
      },
    });
    const ans = answers.thinking_level ?? answers.requires_flash;
    if (
      ans && ans.type === "choice" &&
      (ans.choice === "low" || ans.choice === "lite")
    ) {
      return ThinkingLevel.LOW;
    }
    return ThinkingLevel.HIGH;
  }
  if (!accessRespanToken() && !accessJevToken()) {
    return ThinkingLevel.HIGH;
  }
  const provider = resolveDecisionProvider();
  if (provider === "jev") {
    try {
      return await routeThinkingLevelWithJev(state);
    } catch (err) {
      if (accessRespanToken()) {
        console.warn(
          "[decision-model] Jev routeThinkingLevel failed, falling back to Respan:",
          err,
        );
        try {
          return await routeThinkingLevelWithRespan(state);
        } catch (fallbackErr) {
          console.warn(
            "[decision-model] Respan fallback routeThinkingLevel failed:",
            fallbackErr,
          );
          return ThinkingLevel.HIGH;
        }
      }
      return ThinkingLevel.HIGH;
    }
  }
  try {
    return await routeThinkingLevelWithRespan(state);
  } catch (err) {
    if (accessJevToken()) {
      console.warn(
        "[decision-model] Respan routeThinkingLevel failed, falling back to Jev:",
        err,
      );
      try {
        return await routeThinkingLevelWithJev(state);
      } catch (fallbackErr) {
        console.warn(
          "[decision-model] Jev fallback routeThinkingLevel failed:",
          fallbackErr,
        );
        return ThinkingLevel.HIGH;
      }
    }
    return ThinkingLevel.HIGH;
  }
};

export const routeTask = async (
  state: string | Record<string, unknown> | unknown[],
): Promise<ModelTier> => {
  const level = await routeThinkingLevel(state);
  return level === ThinkingLevel.LOW ? "lite" : "flash";
};

export type PastToolEpisode = {
  turnIndex: number;
  startTime: number;
  endTime: number;
  startTimeStr: string;
  endTimeStr: string;
  userText: string;
  botText: string;
  toolCalls: { name: string; params: unknown }[];
  toolResultsSummary: string[];
  eventsCount: number;
};

export const extractCandidateToolEpisodes = (
  history: HistoryEvent[],
): PastToolEpisode[] => {
  const episodes: PastToolEpisode[] = [];
  let currentEpisode: Partial<PastToolEpisode> | null = null;
  let turnIdx = 0;

  for (let i = 0; i < history.length; i++) {
    const e = history[i];
    if (e.type === "participant_utterance") {
      if (
        currentEpisode &&
        currentEpisode.toolCalls &&
        currentEpisode.toolCalls.length > 0 &&
        currentEpisode.botText
      ) {
        episodes.push(currentEpisode as PastToolEpisode);
      }
      turnIdx++;
      currentEpisode = {
        turnIndex: turnIdx,
        startTime: e.timestamp,
        endTime: e.timestamp,
        startTimeStr: new Date(e.timestamp).toISOString(),
        endTimeStr: new Date(e.timestamp).toISOString(),
        userText: e.text ?? "",
        botText: "",
        toolCalls: [],
        toolResultsSummary: [],
        eventsCount: 1,
      };
    } else if (currentEpisode) {
      currentEpisode.endTime = e.timestamp;
      currentEpisode.endTimeStr = new Date(e.timestamp).toISOString();
      currentEpisode.eventsCount = (currentEpisode.eventsCount ?? 0) + 1;

      if (e.type === "tool_call") {
        currentEpisode.toolCalls?.push({ name: e.name, params: e.parameters });
      } else if (e.type === "tool_result") {
        const preview = typeof e.result === "string"
          ? e.result.slice(0, 150).replace(/\n/g, " ")
          : "";
        currentEpisode.toolResultsSummary?.push(preview);
      } else if (e.type === "own_utterance") {
        currentEpisode.botText = e.text ?? "";
      }
    }
  }

  if (episodes.length < 2) return [];

  const candidatePool = episodes.slice(0, -1);

  return candidatePool.filter((ep) => {
    const alreadyCompacted = history.some((e) =>
      e.type === "tool_call" &&
      e.name === cleanActiveMemoryToolName &&
      // deno-lint-ignore no-explicit-any
      (e.parameters as any)?.start_time === ep.startTimeStr
    );
    return !alreadyCompacted;
  });
};

export const decideSkillsWithDecisionModel = async (
  prompt: string,
  history: HistoryEvent[],
  candidateSkills: Skill[],
  currentlyActiveSkills: Set<string>,
): Promise<{ toLearn: Skill[]; toUnlearn: Skill[] }> => {
  if (!isDecisionModelAvailable()) {
    return { toLearn: [], toUnlearn: [] };
  }
  if (empty(candidateSkills)) {
    return { toLearn: [], toUnlearn: [] };
  }
  const lastUser = lastParticipantUtterance(history);
  if (!lastUser || typeof lastUser.text !== "string" || !lastUser.text.trim()) {
    return { toLearn: [], toUnlearn: [] };
  }

  const recentDialogue = history
    .filter((e) =>
      e.type === "participant_utterance" || e.type === "own_utterance"
    )
    .slice(-4)
    .map((e) =>
      `${e.type === "participant_utterance" ? "User" : "Assistant"}: ${
        "text" in e && typeof e.text === "string" ? e.text : ""
      }`
    )
    .join("\n");

  const facts = verifiedToolFacts(history);

  const state = {
    agent_role: prompt.slice(0, 1000),
    verified_conversation_facts_and_history:
      (recentDialogue + (facts ? `\n\nFacts:\n${facts}` : "")).slice(-10000),
    current_user_request: lastUser.text,
  };

  const questions = Object.fromEntries(
    candidateSkills.map((s) => [
      s.name,
      {
        type: "noul" as const,
        instructions:
          `Does answering this turn or executing the user request directly require the '${s.name}' skill (${s.description})?`,
      },
    ]),
  );

  try {
    const answers = await callDecisionModel(state, questions);
    const scores = Object.fromEntries(
      candidateSkills.map((s) => {
        const ans = answers[s.name];
        return [s.name, ans && ans.type === "noul" ? ans.noul : 0];
      }),
    );

    const candidateToLearn = candidateSkills
      .filter((s) => !currentlyActiveSkills.has(s.name.toLowerCase()))
      .filter((s) => (scores[s.name] ?? 0) >= 0.60)
      .sort((a, b) => (scores[b.name] ?? 0) - (scores[a.name] ?? 0));

    const toLearn = candidateToLearn.length > 0
      ? [
        candidateToLearn[0],
        ...candidateToLearn.slice(1).filter((s) =>
          (scores[s.name] ?? 0) >= 0.80
        ),
      ]
      : [];

    const toUnlearn = candidateSkills
      .filter((s) => currentlyActiveSkills.has(s.name.toLowerCase()))
      .filter((s) => (scores[s.name] ?? 1) < 0.20);

    return { toLearn, toUnlearn };
  } catch (err) {
    console.warn(
      "[decision-model-skills] skill decision check failed, failing open:",
      err,
    );
    return { toLearn: [], toUnlearn: [] };
  }
};

export const decideSkillsWithRespan = decideSkillsWithDecisionModel;
export const decideSkillsWithJev = decideSkillsWithDecisionModel;

export const decideCleanupWithDecisionModel = async (
  prompt: string,
  history: HistoryEvent[],
): Promise<PastToolEpisode[]> => {
  if (!isDecisionModelAvailable()) {
    return [];
  }

  const candidateEpisodes = extractCandidateToolEpisodes(history);
  if (empty(candidateEpisodes)) return [];

  const lastUser = lastParticipantUtterance(history);
  const currentUserRequest = lastUser && typeof lastUser.text === "string"
    ? lastUser.text
    : "No prompt";

  const state = {
    agent_role: prompt.slice(0, 1000),
    current_user_request: currentUserRequest,
    past_tool_episodes: candidateEpisodes.map((ep) => ({
      turn_id: `turn_${ep.turnIndex}`,
      user_request: ep.userText.slice(0, 150),
      tools_invoked: ep.toolCalls.map((t) => t.name).join(", "),
      tool_results_preview: ep.toolResultsSummary.slice(0, 2).join(" | ")
        .slice(0, 200),
      bot_response: ep.botText.slice(0, 150),
    })),
  };

  const questions = Object.fromEntries(
    candidateEpisodes.map((ep) => [
      `compact_turn_${ep.turnIndex}`,
      {
        type: "noul" as const,
        instructions:
          `Should the intermediate tool execution logs of Turn ${ep.turnIndex} ("${
            ep.userText.slice(0, 60)
          }") be compacted into a summary?`,
        criteria: {
          true:
            `The action or subtask in Turn ${ep.turnIndex} has been completed and responded to. The current user request ("${
              currentUserRequest.slice(0, 60)
            }") does not require re-inspecting the exact raw tool output, error stack traces, or intermediate line numbers from Turn ${ep.turnIndex}.`,
          false:
            `The current user request is directly asking about, debugging, or relying on specific raw details from Turn ${ep.turnIndex} that would be lost if summarized.`,
        },
      },
    ]),
  );

  try {
    const answers = await callDecisionModel(state, questions);
    return candidateEpisodes.filter((ep) => {
      const ans = answers[`compact_turn_${ep.turnIndex}`];
      const score = ans && ans.type === "noul" ? ans.noul : 0;
      return score >= 0.60;
    });
  } catch (err) {
    console.warn(
      "[decision-model-cleanup] active memory cleanup check failed, failing open:",
      err,
    );
    return [];
  }
};

export const decideCleanupWithRespan = decideCleanupWithDecisionModel;
export const decideCleanupWithJev = decideCleanupWithDecisionModel;
