import { context, type Injection, type Injector } from "@uri/inject";
import { empty } from "gamla";
import { cache } from "rmmbr";
import { makeCache } from "./cacher.ts";
import {
  type ChoiceDecisionAnswer,
  decideCleanupWithDecisionModel,
  decideSkillsWithDecisionModel,
  type DecisionAnswer,
  type DecisionQuestion,
  formatAgentStateForDecisionModel,
  type NoulDecisionAnswer,
  type NoulDecisionQuestion,
  routeTask,
  type ScoreDecisionAnswer,
} from "./decisionModel.ts";
import type { ModelTier } from "./utils.ts";

export {
  decideCleanupWithDecisionModel as decideCleanupWithRespan,
  decideSkillsWithDecisionModel as decideSkillsWithRespan,
  formatAgentStateForDecisionModel as formatAgentStateForRespan,
  routeTask,
};

export const respanApiUrl = "https://api.respan.ai/api/v1/scores";
export const defaultRespanModel = "span-01-free";
export const proRespanModel = "span-01-pro";
const rmmbrUrl = "https://rmmbr.net";

const respanToken: Injection<() => string | undefined> = context(
  (): string | undefined => Deno.env.get("RESPAN_API_KEY"),
);

export const accessRespanToken = respanToken.access;
export const injectRespanToken = (token: string): Injector =>
  respanToken.inject(() => token);

export type RespanMessage = {
  role: "user" | "assistant" | "system";
  content: string;
};

export type RespanSpan = {
  input: RespanMessage[];
  output: RespanMessage;
};

export type RespanBehavior = {
  id: string;
  definition: string;
};

export type RespanBehaviorResult = {
  id: string;
  p_present: number;
  p_absent: number;
  p_not_observable: number;
};

export type RespanScoreResponse = {
  model: string;
  results: RespanBehaviorResult[];
  usage?: {
    input_tokens: number;
  };
};

const toRespanMessage = (
  event: unknown,
): RespanMessage | undefined => {
  if (typeof event !== "object" || event === null) return undefined;
  if ("role" in event && typeof event.role === "string") {
    const role: RespanMessage["role"] =
      event.role === "user" || event.role === "system"
        ? event.role
        : "assistant";
    const content = "content" in event && typeof event.content === "string"
      ? event.content
      : JSON.stringify(event);
    return { role, content };
  }
  if ("type" in event && typeof event.type === "string") {
    if (event.type === "participant_utterance") {
      return {
        role: "user",
        content: "text" in event && typeof event.text === "string"
          ? event.text
          : "",
      };
    }
    if (event.type === "own_utterance") {
      return {
        role: "assistant",
        content: "text" in event && typeof event.text === "string"
          ? event.text
          : "",
      };
    }
    if (event.type === "tool_call") {
      const name = "name" in event && typeof event.name === "string"
        ? event.name
        : "tool";
      const params = "parameters" in event ? event.parameters : {};
      return {
        role: "assistant",
        content: `${name}(${JSON.stringify(params)})`,
      };
    }
    if (event.type === "tool_result") {
      return {
        role: "system",
        content: "result" in event && typeof event.result === "string"
          ? event.result
          : "",
      };
    }
    if (
      event.type === "system_notification" ||
      event.type === "system_prompt"
    ) {
      return {
        role: "system",
        content: "text" in event && typeof event.text === "string"
          ? event.text
          : "",
      };
    }
  }
  return undefined;
};

const isRespanSpan = (val: unknown): val is RespanSpan =>
  typeof val === "object" &&
  val !== null &&
  "input" in val &&
  Array.isArray((val as RespanSpan).input) &&
  "output" in val &&
  typeof (val as RespanSpan).output === "object" &&
  (val as RespanSpan).output !== null &&
  (val as RespanSpan).output.role === "assistant";

export const stateToSpan = (state: unknown): RespanSpan => {
  if (isRespanSpan(state)) return state;

  if (typeof state === "string") {
    try {
      const parsed = JSON.parse(state);
      if (typeof parsed === "object" && parsed !== null) {
        return stateToSpan(parsed);
      }
    } catch {
      // plain text string, fall through
    }
    return {
      input: [{ role: "user", content: state }],
      output: { role: "assistant", content: "" },
    };
  }

  if (Array.isArray(state)) {
    const rawMessages = state
      .map(toRespanMessage)
      .filter((m): m is RespanMessage => Boolean(m));
    const messages = empty(rawMessages)
      ? [{ role: "user" as const, content: "Empty request" }]
      : rawMessages;
    const lastMsg = messages[messages.length - 1];
    if (lastMsg && lastMsg.role === "assistant") {
      const input = messages.slice(0, -1);
      return {
        input: empty(input)
          ? [{ role: "user" as const, content: "Context" }]
          : input,
        output: lastMsg,
      };
    }
    return {
      input: messages,
      output: { role: "assistant", content: "" },
    };
  }

  if (typeof state === "object" && state !== null) {
    const record = state as Record<string, unknown>;

    const historyList = ("conversation_history_events" in record &&
        Array.isArray(record.conversation_history_events)
      ? record.conversation_history_events
      : undefined) ??
      ("history" in record && Array.isArray(record.history)
        ? record.history
        : undefined);

    if (historyList) {
      const rawMessages = historyList
        .map(toRespanMessage)
        .filter((m): m is RespanMessage => Boolean(m));
      const sysPrompt = "system_prompt" in record &&
          typeof record.system_prompt === "string"
        ? record.system_prompt
        : "";
      const allMessages = sysPrompt
        ? [{ role: "system" as const, content: sysPrompt }, ...rawMessages]
        : rawMessages;
      const lastMsg = allMessages[allMessages.length - 1];
      if (lastMsg && lastMsg.role === "assistant") {
        const input = allMessages.slice(0, -1);
        return {
          input: empty(input)
            ? [{ role: "user" as const, content: "Context" }]
            : input,
          output: lastMsg,
        };
      }
      return {
        input: empty(allMessages)
          ? [{ role: "user" as const, content: "Empty request" }]
          : allMessages,
        output: { role: "assistant", content: "" },
      };
    }

    if (
      "assistant_response" in record &&
      typeof record.assistant_response === "string"
    ) {
      const userQuery =
        "user_query" in record && typeof record.user_query === "string"
          ? record.user_query
          : "";
      const facts = ("verified_facts_and_tool_outputs" in record &&
          typeof record.verified_facts_and_tool_outputs === "string"
        ? record.verified_facts_and_tool_outputs
        : undefined) ??
        ("verified_facts_from_tools" in record &&
            typeof record.verified_facts_from_tools === "string"
          ? record.verified_facts_from_tools
          : "");
      const sysParts = [
        "instructions" in record && typeof record.instructions === "string"
          ? record.instructions
          : "",
        facts ? `Facts:\n${facts}` : "",
      ].filter(Boolean);
      return {
        input: [
          ...(sysParts.length > 0
            ? [{ role: "system" as const, content: sysParts.join("\n\n") }]
            : []),
          { role: "user" as const, content: userQuery },
        ],
        output: {
          role: "assistant",
          content: record.assistant_response,
        },
      };
    }

    if ("instructions" in record && "content" in record) {
      const instr = typeof record.instructions === "string"
        ? record.instructions
        : "";
      const cont = typeof record.content === "string"
        ? record.content
        : JSON.stringify(record.content);
      const isAssistant = /\bassistant\b/i.test(instr);
      return {
        input: [
          ...(instr ? [{ role: "system" as const, content: instr }] : []),
          ...(!isAssistant ? [{ role: "user" as const, content: cont }] : []),
        ],
        output: {
          role: "assistant",
          content: isAssistant ? cont : "",
        },
      };
    }

    if (
      "current_user_request" in record &&
      typeof record.current_user_request === "string"
    ) {
      const sysParts = [
        "agent_role" in record && typeof record.agent_role === "string"
          ? `Agent Role: ${record.agent_role}`
          : "",
        "verified_conversation_facts_and_history" in record &&
          typeof record.verified_conversation_facts_and_history === "string"
          ? `Conversation context:\n${record.verified_conversation_facts_and_history}`
          : "",
        "past_tool_episodes" in record
          ? `Past tool episodes:\n${JSON.stringify(record.past_tool_episodes)}`
          : "",
      ].filter(Boolean);
      return {
        input: [
          ...(sysParts.length > 0
            ? [{ role: "system" as const, content: sysParts.join("\n\n") }]
            : []),
          { role: "user" as const, content: record.current_user_request },
        ],
        output: { role: "assistant", content: "" },
      };
    }

    if (
      "trigger_content" in record &&
      typeof record.trigger_content === "string"
    ) {
      const recentTurns = "recent_turns" in record &&
          Array.isArray(record.recent_turns)
        ? record.recent_turns
          .map((t) => `${t?.type ?? "turn"}: ${t?.text ?? ""}`)
          .join("\n")
        : "";
      const sysParts = [
        "agent_role" in record && typeof record.agent_role === "string"
          ? `Agent Role: ${record.agent_role}`
          : "",
        "tools_available" in record && Array.isArray(record.tools_available)
          ? `Tools available: ${record.tools_available.join(", ")}`
          : "",
        recentTurns ? `Recent turns:\n${recentTurns}` : "",
      ].filter(Boolean);
      return {
        input: [
          ...(sysParts.length > 0
            ? [{ role: "system" as const, content: sysParts.join("\n\n") }]
            : []),
          { role: "user" as const, content: record.trigger_content },
        ],
        output: { role: "assistant", content: "" },
      };
    }

    return {
      input: [{ role: "user", content: JSON.stringify(state) }],
      output: { role: "assistant", content: "" },
    };
  }

  return {
    input: [{ role: "user", content: "Empty request" }],
    output: { role: "assistant", content: "" },
  };
};

const formatNoulDefinition = (
  q: NoulDecisionQuestion,
  name: string,
): string => {
  const parts = [
    q.instructions || name,
    q.criteria?.true ? `Present if: ${q.criteria.true}.` : "",
    q.criteria?.false ? `Absent if: ${q.criteria.false}.` : "",
  ].filter(Boolean);
  const def = parts.join(" ").trim();
  return def.length < 3 ? `${def} (check)` : def;
};

const formatChoiceOptionDefinition = (
  instructions: string | undefined,
  qName: string,
  optionKey: string,
  criteriaText: string,
): string => {
  const parts = [
    instructions ? `${instructions}.` : `Choice for '${qName}':`,
    criteriaText && criteriaText !== optionKey
      ? `The choice is '${optionKey}': ${criteriaText}.`
      : `The choice is '${optionKey}'.`,
  ];
  const def = parts.join(" ").trim();
  return def.length < 3 ? `${def} (option)` : def;
};

const formatScoreLevelDefinition = (
  instructions: string | undefined,
  qName: string,
  level: number,
  crit: string,
): string => {
  const prefix = instructions ? `${instructions}. ` : "";
  const def = `${prefix}Score level ${level} for '${qName}': ${crit}`.trim();
  return def.length < 3 ? `${def} (score)` : def;
};

const buildBehaviorsForQuestions = (
  questions: Record<string, DecisionQuestion>,
): RespanBehavior[] =>
  Object.entries(questions).flatMap(([qName, q]) => {
    if (q.type === "noul") {
      return [{ id: qName, definition: formatNoulDefinition(q, qName) }];
    }
    if (q.type === "choice") {
      return Object.entries(q.criteria).map(([opt, crit]) => ({
        id: `${qName}__choice__${opt}`,
        definition: formatChoiceOptionDefinition(
          q.instructions,
          qName,
          opt,
          crit,
        ),
      }));
    }
    if (q.type === "score") {
      return q.criteria.map((crit, i) => ({
        id: `${qName}__score__${i}`,
        definition: formatScoreLevelDefinition(q.instructions, qName, i, crit),
      }));
    }
    return [];
  });

const parseScoresToAnswers = (
  questions: Record<string, DecisionQuestion>,
  results: RespanBehaviorResult[],
): Record<string, DecisionAnswer> => {
  const resultMap = new Map(results.map((r) => [r.id, r]));

  return Object.fromEntries(
    Object.entries(questions).map(([qName, q]) => {
      if (q.type === "noul") {
        const res = resultMap.get(qName);
        const pPresent = res?.p_present ?? 0;
        const pAbsent = res?.p_absent ?? 0;
        const total = pPresent + pAbsent;
        const noul = total > 0 ? pPresent / total : pPresent;
        return [qName, { type: "noul", noul } satisfies NoulDecisionAnswer];
      }
      if (q.type === "choice") {
        const prefix = `${qName}__choice__`;
        const optionEntries = Object.keys(q.criteria).map((opt) => {
          const res = resultMap.get(`${prefix}${opt}`);
          return [opt, res?.p_present ?? 0] as const;
        });
        const totalP = optionEntries.reduce((sum, [, p]) => sum + p, 0);
        const probabilities = Object.fromEntries(
          optionEntries.map(([opt, p]) => [opt, totalP > 0 ? p / totalP : 0]),
        );
        const best = optionEntries.reduce(
          (curr, prev) => (curr[1] > prev[1] ? curr : prev),
          optionEntries[0] ?? ["", 0],
        );
        return [
          qName,
          {
            type: "choice",
            choice: best[0],
            confidence: best[1],
            probabilities,
          } satisfies ChoiceDecisionAnswer,
        ];
      }
      if (q.type === "score") {
        const prefix = `${qName}__score__`;
        const scoreEntries = q.criteria.map((_, i) => {
          const res = resultMap.get(`${prefix}${i}`);
          return [i, res?.p_present ?? 0] as const;
        });
        const totalP = scoreEntries.reduce((sum, [, p]) => sum + p, 0);
        const probabilities = Object.fromEntries(
          scoreEntries.map(([i, p]) => [
            String(i),
            totalP > 0 ? p / totalP : 0,
          ]),
        );
        const best = scoreEntries.reduce(
          (curr, prev) => (curr[1] > prev[1] ? curr : prev),
          scoreEntries[0] ?? [0, 0],
        );
        return [
          qName,
          {
            type: "score",
            score: best[0],
            confidence: best[1],
            probabilities,
          } satisfies ScoreDecisionAnswer,
        ];
      }
      return [qName, { type: "noul", noul: 0 }];
    }),
  );
};

const rawCallRespanScores = async (
  token: string,
  payload: string,
): Promise<RespanScoreResponse> => {
  let response = await fetch(respanApiUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: payload,
    signal: AbortSignal.timeout(10000),
  });

  if (response.status === 402 && payload.includes(proRespanModel)) {
    await response.body?.cancel();
    const freePayload = payload.replaceAll(proRespanModel, defaultRespanModel);
    response = await fetch(respanApiUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: freePayload,
      signal: AbortSignal.timeout(10000),
    });
  }

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Respan API error (${response.status}): ${errorText}`);
  }
  return await response.json();
};

const inMemoryDecisionCache = new Map<
  string,
  { answers: Record<string, DecisionAnswer>; expiresAt: number }
>();
const memoryTtlMs = 60 * 60 * 1000;

const getRmmbrDecisionCacher = () => {
  const token = Deno.env.get("RMMBR_TOKEN");
  return token
    ? cache({
      cacheId: "respan-decision-v1",
      ttl: 60 * 60 * 24 * 30,
      url: rmmbrUrl,
      token,
      customKeyFn: (_token: string, payload: string) => payload,
    })
    : undefined;
};

let rmmbrDecisionCaller:
  | ((token: string, payload: string) => Promise<RespanScoreResponse>)
  | undefined;

export const callRespanDecisionModel = async (
  state: unknown,
  questions: Record<string, DecisionQuestion>,
): Promise<Record<string, DecisionAnswer>> => {
  const token = accessRespanToken();
  if (!token) throw new Error("No Respan token available");
  if (empty(Object.keys(questions))) return {};

  const span = stateToSpan(state);
  const behaviors = buildBehaviorsForQuestions(questions);

  const payload = JSON.stringify({
    model: Deno.env.get("RESPAN_MODEL") || defaultRespanModel,
    span,
    behaviors,
  });

  const memCached = inMemoryDecisionCache.get(payload);
  if (memCached && Date.now() < memCached.expiresAt) {
    return memCached.answers;
  }

  if (!rmmbrDecisionCaller) {
    const rmmbrCacher = getRmmbrDecisionCacher();
    const injected = makeCache("respan-decision-v1");
    const base = rmmbrCacher
      ? rmmbrCacher(rawCallRespanScores)
      : rawCallRespanScores;
    rmmbrDecisionCaller = injected(base);
  }

  const response = await rmmbrDecisionCaller(token, payload);
  const answers = parseScoresToAnswers(questions, response.results ?? []);
  inMemoryDecisionCache.set(payload, {
    answers,
    expiresAt: Date.now() + memoryTtlMs,
  });
  return answers;
};

export const respanModelSelectionInstructions =
  "The user request or turn requires complex multi-step reasoning, mathematical problem solving, writing code or complex scripts, or hard logic puzzles.";

export const respanModelRoutingBehavior: RespanBehavior = {
  id: "requires_flash",
  definition: respanModelSelectionInstructions,
};

const rawCallRespanRoute = async (
  token: string,
  payload: string,
): Promise<ModelTier> => {
  let response = await fetch(respanApiUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: payload,
    signal: AbortSignal.timeout(3500),
  });

  if (response.status === 402 && payload.includes(proRespanModel)) {
    await response.body?.cancel();
    const freePayload = payload.replaceAll(proRespanModel, defaultRespanModel);
    response = await fetch(respanApiUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: freePayload,
      signal: AbortSignal.timeout(3500),
    });
  }

  if (!response.ok) {
    await response.body?.cancel();
    return "flash";
  }
  const data = await response.json();
  const result = (data?.results as RespanBehaviorResult[] | undefined)?.find(
    (r) => r.id === "requires_flash",
  );
  if (!result) return "flash";
  return result.p_present >= 0.50 ? "flash" : "lite";
};

const inMemoryRouteCache = new Map<
  string,
  { tier: ModelTier; expiresAt: number }
>();

const getRmmbrRouteCacher = () => {
  const token = Deno.env.get("RMMBR_TOKEN");
  return token
    ? cache({
      cacheId: "respan-model-route-v1",
      ttl: 60 * 60 * 24 * 7,
      url: rmmbrUrl,
      token,
      customKeyFn: (_token: string, payload: string) => payload,
    })
    : undefined;
};

let rmmbrRouteCaller:
  | ((token: string, payload: string) => Promise<ModelTier>)
  | undefined;

export const routeTaskWithRespan = async (
  state: string | Record<string, unknown> | unknown[],
): Promise<ModelTier> => {
  const token = accessRespanToken();
  if (!token) return "flash";

  const span = stateToSpan(state);
  const payload = JSON.stringify({
    model: Deno.env.get("RESPAN_MODEL") || defaultRespanModel,
    span,
    behaviors: [respanModelRoutingBehavior],
  });

  const memCached = inMemoryRouteCache.get(payload);
  if (memCached && Date.now() < memCached.expiresAt) {
    return memCached.tier;
  }

  try {
    if (!rmmbrRouteCaller) {
      const cacher = getRmmbrRouteCacher();
      rmmbrRouteCaller = cacher
        ? cacher((t: string, p: string) => rawCallRespanRoute(t, p))
        : (t: string, p: string) => rawCallRespanRoute(t, p);
    }
    const tier = await rmmbrRouteCaller(token, payload);
    inMemoryRouteCache.set(payload, {
      tier,
      expiresAt: Date.now() + memoryTtlMs,
    });
    return tier;
  } catch {
    return "flash";
  }
};
