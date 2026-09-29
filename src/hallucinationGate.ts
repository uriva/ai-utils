import { z } from "zod/v4";
import type { HistoryEvent, ParticipantUtterance } from "./agent.ts";
import { isCompactedSummaryText } from "./compaction.ts";
import {
  decide,
  isDecisionModelAvailable,
  lastParticipantUtterance,
  verifiedToolFacts,
} from "./decisionModel.ts";

export { lastParticipantUtterance, verifiedToolFacts };

export const maxHallucinationRetries = 1;

export const cleanUserQuery = (text: string): string =>
  text.replace(/^\[(?:replying to you|quoted reply):[^\]]*\]\s*/i, "").trim() ||
  text;

export const HallucinationDecisionSchema: z.ZodObject<{
  is_hallucination: z.ZodBoolean;
}> = z.object({
  is_hallucination: z.boolean().describe(
    "True if the assistant response is an off-topic hallucination, invents unprompted claims or user concerns, or completely fails to address what the user asked. False if the assistant legitimately answers, addresses the user request, provides relevant options or recommendations, clarifies, greets, or explains it could not find the information.",
  ),
});

export const hallucinationCorrectionText = (userQuery: string): string =>
  `SYSTEM AUDIT: Your proposed response was flagged as an off-topic hallucination or unprompted diversion. The user explicitly asked: "${
    userQuery.slice(0, 300)
  }". Do NOT address unprompted topics or invent concerns. Directly answer the user's explicit question using only facts verified by your tools or memory.`;

export const isUserPromptedTurn = (history: HistoryEvent[]): boolean => {
  const lastUserIndex = history.findLastIndex(
    (e) =>
      e.type === "participant_utterance" ||
      e.type === "participant_edit_message",
  );
  if (lastUserIndex === -1) return false;
  const subsequentEvents = history.slice(lastUserIndex + 1);
  const hadInterveningReply = subsequentEvents.some(
    (e) => e.type === "own_utterance" || e.type === "own_edit_message",
  );
  if (hadInterveningReply) return false;
  const hadInterveningPlatformEvent = subsequentEvents.some(
    (e) =>
      e.type === "external_event" ||
      (e.type === "own_thought" &&
        !("modelMetadata" in e && e.modelMetadata) &&
        typeof e.text === "string" &&
        !isCompactedSummaryText(e.text)),
  );
  if (hadInterveningPlatformEvent) return false;
  return true;
};

export const recentUserQueriesText = (history: HistoryEvent[]): string =>
  history
    .filter((e): e is ParticipantUtterance =>
      e.type === "participant_utterance" && typeof e.text === "string"
    )
    .slice(-3)
    .map((e) => (e.name ? `${e.name}: ${e.text}` : e.text))
    .join("\n");

export const auditUtteranceForHallucination = async (
  userQuery: string,
  assistantResponse: string,
  verifiedFacts?: string,
): Promise<boolean> => {
  if (!isDecisionModelAvailable()) return false;
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
