import { context, type Injection, type Injector } from "@uri/inject";
import { cache } from "rmmbr";
import type { HistoryEvent } from "./agent.ts";
import type { ModelTier } from "./utils.ts";

export const jevApiUrl = "https://api.typesafe.ai/v1/systemone";
const rmmbrUrl = "https://rmmbr.net";

const jevToken: Injection<() => string | undefined> = context(
  (): string | undefined => Deno.env.get("JEV_API_KEY"),
);

export const accessJevToken = jevToken.access;
export const injectJevToken = (token: string): Injector =>
  jevToken.inject(() => token);

export const formatAgentStateForJev = (
  prompt: string,
  history: HistoryEvent[],
  tools?: { name: string }[],
): Record<string, unknown> => {
  const lastEvent = history[history.length - 1];
  const lastUserMsg = [...history]
    .reverse()
    .find((e) => e.type === "participant_utterance");
  const triggerContent =
    lastEvent && "text" in lastEvent && typeof lastEvent.text === "string"
      ? lastEvent.text.slice(0, 1000)
      : (lastUserMsg && "text" in lastUserMsg
        ? String(lastUserMsg.text).slice(0, 1000)
        : "Empty user request");
  const triggerType = lastEvent ? lastEvent.type : "conversation_start";
  const recentEvents = history.slice(-4).map((e) => ({
    type: e.type,
    text: "text" in e && typeof e.text === "string"
      ? e.text.slice(0, 300)
      : undefined,
  }));
  return {
    trigger_type: triggerType,
    trigger_content: triggerContent,
    agent_role: prompt.slice(0, 500),
    tools_available: (tools ?? []).map((t) => t.name).slice(0, 15),
    recent_turns: recentEvents,
  };
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
          instructions:
            "Which model tier should handle this task? Choose 'lite' for simple user conversational turns, greetings, basic FAQ, or straightforward single-step data extraction; choose 'flash' for system notifications, behavioral instructions, multi-step reasoning, coding, mathematical logic, or intricate planning.",
          criteria: {
            lite:
              "Simple user conversational turn, basic information retrieval, straightforward query, or basic request",
            flash:
              "System notification, behavioral correction, complex multi-step task, nuanced reasoning, coding, or intricate planning",
          },
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
      cacheId: "jev-model-route-v3",
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
