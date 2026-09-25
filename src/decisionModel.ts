import { context, type Injection, type Injector } from "@uri/inject";
import { empty } from "gamla";
import type { z, ZodType } from "zod/v4";
import { callJevDecisionModel } from "./jev.ts";
import { type ModelOpts, validateAgainstSchema } from "./utils.ts";

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

const decisionModelOverrideInjection: Injection<
  () => DecisionModelCaller | null
> = context((): DecisionModelCaller | null => null);

export const injectDecisionModel = (
  caller: DecisionModelCaller,
): Injector => decisionModelOverrideInjection.inject(() => caller);

export const callDecisionModel = (
  state: unknown,
  questions: Record<string, DecisionQuestion>,
): Promise<Record<string, DecisionAnswer>> => {
  const override = decisionModelOverrideInjection.access();
  if (override) {
    return override(state, questions);
  }
  return callJevDecisionModel(state, questions);
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
