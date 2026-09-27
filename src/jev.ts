import { context, type Injection, type Injector } from "@uri/inject";
import { cache } from "rmmbr";
import type { HistoryEvent, Skill } from "./agent.ts";
import { makeCache } from "./cacher.ts";
import {
  callDecisionModel,
  type DecisionAnswer,
  type DecisionQuestion,
  isDecisionModelInjected,
} from "./decisionModel.ts";
import {
  lastParticipantUtterance,
  verifiedToolFacts,
} from "./hallucinationGate.ts";
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

export const decideSkillsWithJev = async (
  prompt: string,
  history: HistoryEvent[],
  candidateSkills: Skill[],
  currentlyActiveSkills: Set<string>,
): Promise<{ toLearn: Skill[]; toUnlearn: Skill[] }> => {
  const token = accessJevToken();
  if (!token && !isDecisionModelInjected()) {
    return { toLearn: [], toUnlearn: [] };
  }
  if (candidateSkills.length === 0) {
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
      "[jev-skills] skill decision check failed, failing open:",
      err,
    );
    return { toLearn: [], toUnlearn: [] };
  }
};
