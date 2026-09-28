import { z } from "zod/v4";
import type { HistoryEvent, ParticipantUtterance } from "./agent.ts";
import { decide } from "./decisionModel.ts";
import { accessJevToken } from "./jev.ts";
import { accessRespanToken } from "./respan.ts";

export const maxHallucinationRetries = 2;

export const HallucinationDecisionSchema: z.ZodObject<{
  is_hallucination: z.ZodBoolean;
}> = z.object({
  is_hallucination: z.boolean().describe(
    "True if the assistant response is an off-topic hallucination, invents unprompted claims or user concerns, or completely fails to address what the user asked. False if the assistant legitimately answers, addresses the user request, clarifies, greets, or explains it could not find the information.",
  ),
});

export const hallucinationCorrectionText = (userQuery: string): string =>
  `SYSTEM AUDIT: Your proposed response was flagged as an off-topic hallucination or unprompted diversion. The user explicitly asked: "${
    userQuery.slice(0, 300)
  }". Do NOT address unprompted topics or invent concerns. Directly answer the user's explicit question using only facts verified by your tools or memory.`;

export const lastParticipantUtterance = (
  history: HistoryEvent[],
): ParticipantUtterance | undefined =>
  [...history].reverse().find((e): e is ParticipantUtterance =>
    e.type === "participant_utterance"
  );

export const recentUserQueriesText = (history: HistoryEvent[]): string =>
  history
    .filter((e): e is ParticipantUtterance =>
      e.type === "participant_utterance" && typeof e.text === "string"
    )
    .slice(-3)
    .map((e) => (e.name ? `${e.name}: ${e.text}` : e.text))
    .join("\n");

export const verifiedToolFacts = (history: HistoryEvent[]): string =>
  history
    .flatMap((e) => {
      if (e.type === "tool_result") return [e.result];
      if (e.type === "external_event") return [e.text];
      if (e.type === "own_thought" && typeof e.text === "string") {
        return [e.text];
      }
      return [];
    })
    .join("\n\n")
    .slice(0, 10000);

export const auditUtteranceForHallucination = async (
  userQuery: string,
  assistantResponse: string,
  verifiedFacts?: string,
): Promise<boolean> => {
  const token = accessRespanToken() || accessJevToken();
  if (!token) return false;
  try {
    const result = await decide(
      "Audit assistant response for hallucination or off-topic diversion from the user request.",
      HallucinationDecisionSchema,
    )({
      user_query: userQuery,
      assistant_response: assistantResponse,
      ...(verifiedFacts
        ? {
          verified_facts_and_tool_outputs: verifiedFacts,
          verified_facts_from_tools: verifiedFacts,
        }
        : {}),
    });
    return result.is_hallucination;
  } catch (err) {
    console.warn(
      "[hallucination-gate] decision check failed, failing open:",
      err,
    );
    return false;
  }
};
