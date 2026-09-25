import { context, type Injection, type Injector } from "@uri/inject";
import { cache } from "rmmbr";
import type { HistoryEvent } from "./agent.ts";
import { makeCache } from "./cacher.ts";
import type { DecisionAnswer, DecisionQuestion } from "./decisionModel.ts";
import type { ModelTier } from "./utils.ts";

export const jevApiUrl = "https://api.typesafe.ai/v1/systemone";
const rmmbrUrl = "https://rmmbr.net";

const jevToken: Injection<() => string | undefined> = context(
  (): string | undefined => Deno.env.get("JEV_API_KEY"),
);

export const accessJevToken = jevToken.access;
export const injectJevToken = (token: string): Injector =>
  jevToken.inject(() => token);

const eventContent = (event: HistoryEvent): string | undefined => {
  if ("text" in event && typeof event.text === "string") return event.text;
  if ("result" in event && typeof event.result === "string") {
    return event.result;
  }
  if (event.type === "tool_call") {
    return `${event.name}(${JSON.stringify(event.parameters ?? {})})`;
  }
  return undefined;
};

export const formatAgentStateForJev = (
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

export const jevModelSelectionInstructions =
  "Which model tier should handle this task? Choose 'lite' for conversational turns, greetings, basic FAQ, or general information questions; choose 'flash' for tool execution, action requests (such as downloading, cutting, editing, or booking), recent tool activity, system notifications, negative constraints, coding, or multi-step logic.";

export const jevModelSelectionCriteria = {
  lite:
    "Conversational greeting, general FAQ, information question, or pleasantry without tool actions",
  flash:
    "Action request (downloading, cutting, media processing, external actions), recent tool activity, system notification, negative constraint adherence, coding, or complex reasoning",
};

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
      cacheId: "jev-model-route-v4",
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
