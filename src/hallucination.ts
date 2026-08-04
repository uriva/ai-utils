import { genJson } from "./genJson.ts";
import { accessGeminiToken } from "./gemini.ts";
import {
  type AgentInputs,
  getSpecForTurn,
  type HistoryEvent,
  ownUtteranceTurn,
} from "./agent.ts";
import { eventToPlainText } from "./compaction.ts";
import { formatInternalSentTimestamp } from "./internalMessageMetadata.ts";
import { zodToTypingString } from "./toolTyping.ts";
import { z } from "zod/v4";

export type HallucinationCheckResult = {
  isHallucinating: boolean;
  explanation: string;
  noteToBot?: string;
};

const hallucinationCheckSchema: z.ZodType<HallucinationCheckResult> = z.object({
  isHallucinating: z.boolean().describe(
    "Whether the response is hallucinating",
  ),
  explanation: z.string().describe(
    "Why the response is or isn't hallucinating",
  ),
  noteToBot: z.string().optional().describe(
    "A note to the bot acknowledging that the previous inaccurate message was already sent to the user, and advising how to correct it. Phrase it as a note to self. The style should conform to the prompt style and language.",
  ),
});

const editLastMessageToolName = "edit_last_message";

const correctionInstruction =
  `If hallucinations are detected, provide a note to the bot. Consider the nature of the issue:
- If the bot claimed it performed an action (e.g. calling a tool) but hasn't yet, or committed to a future action without performing or scheduling it, the note should simply advise the bot to go ahead and perform or schedule that action now with the appropriate tool — or, if the action is impossible, to rewrite the reply without the claim or commitment. No correction or apology needed.
- If the bot stated incorrect information, the note should advise the bot to use the ${editLastMessageToolName} tool to correct the message, or if that fails, to gently correct itself in a follow-up (e.g., "I sent an inaccurate message. I should use ${editLastMessageToolName} to fix it, or follow up with a correction like 'sorry, I meant...'").
Make sure to phrase this note as if the bot is writing it to itself.`;

// The checker judges claims against the serialized history, so that view must
// carry everything the model legitimately grounds claims in. Message
// timestamps are shown to the model (as " — sent ..." suffixes) and are the
// only source for time/date claims; omitting them here makes the checker flag
// legitimate time references as fabrications.
const eventToGroundTruthText =
  (timezoneIANA: string) => (e: HistoryEvent): string =>
    `[${formatInternalSentTimestamp(e.timestamp, timezoneIANA)}] ${
      eventToPlainText(e)
    }`;

export const createHallucinationCheckPrompt = (
  history: HistoryEvent[],
  spec: AgentInputs,
  timezoneIANA = "UTC",
): string => {
  const specForTurn = getSpecForTurn(spec, history);

  const toolsText = (specForTurn.tools || [])
    .map((t) =>
      `- ${t.name}(params: ${
        zodToTypingString(t.parameters)
      }): ${t.description}`
    )
    .join("\n");

  const lastEvent = history[history.length - 1];
  const modelOutput = lastEvent && lastEvent.type === "own_utterance"
    ? lastEvent.text
    : "";

  const contextHistory = history.slice(0, history.length - 1);
  const serializedHistory = contextHistory
    .map(eventToGroundTruthText(timezoneIANA))
    .join("\n\n");

  return `=== SYSTEM PROMPT & ACTIVE INSTRUCTIONS ===
${specForTurn.prompt}

=== AVAILABLE TOOLS ===
${toolsText || "None"}

=== CONVERSATION HISTORY ===
${serializedHistory}

=== BOT'S LAST RESPONSE ===
${modelOutput}

IMPORTANT: The system instructions, available tools, and conversation history sections above are absolute GROUND TRUTH. 
Any specific factual claim (names, prices, URLs, dates, addresses, etc.) in the bot's last response MUST appear verbatim or be directly traceable as a logical inference from this ground truth text.

Analyze the bot's response carefully. Flag a hallucination if ANY of the following patterns holds:

A. Unsupported facts: the response contains specific factual claims or third-party links/URLs that are NOT supported by the system instructions, available tools, or conversation history, and the fabrication would meaningfully derail the conversation.

B. Phantom actions: the response states that an action was performed — something was sent, saved, recorded, scheduled, booked, updated, deleted, or a person was notified — but no tool_call performing that action appears in the conversation history. Mere conversational acknowledgments (e.g. "noted", "got it") are NOT actions; do not flag those.

C. Empty commitments: the response commits the bot to a future consequential action — notifying, updating or contacting a person, sending or scheduling something, checking and reporting back — but no tool_call performing or scheduling that action appears in the conversation history. The bot's turn is over once this response is sent, so such a commitment will never be fulfilled. Offers and questions (e.g. "shall I update her?") are NOT commitments; do not flag those.

Do NOT flag a hallucination if:
- The information is reasonably correct common knowledge
- The information is supported by any tool_result, own_thought, or external_event in the history (even older ones)
- The bot is paraphrasing, summarizing, or making directly implied logical inferences from the ground truth
- An action claim or commitment is backed by a tool_call in the history that performs or schedules it

${correctionInstruction}`;
};

const callModel = (prompt: string) =>
  genJson(
    { provider: "google", mini: false },
    `You are a hallucination detection expert. Your job is to verify whether a bot's response contains fabricated or unverified information with NO basis in its instructions, prompt, or conversation history. The instructions, prompt, tools, and history are absolute ground truth. Only flag clear-cut fabrications, not paraphrasing, reasonable inferences, or common knowledge.`,
    hallucinationCheckSchema,
  )(prompt);

export const checkHallucination = async (
  history: HistoryEvent[],
  spec: AgentInputs,
  timezoneIANA?: string,
): Promise<HallucinationCheckResult> => {
  const checkPrompt = createHallucinationCheckPrompt(
    history,
    spec,
    timezoneIANA,
  );
  return await callModel(checkPrompt);
};

export const ungroundedReplyRetryAdvice =
  "Rewrite the reply so that every statement about what was done and its outcome matches the tool calls and tool results in the conversation. If an action failed or its outcome is unknown, say so plainly instead of claiming success. If something was not actually done, do not claim it was — either do it now using the appropriate tool, or tell the user it has not been done yet. Likewise, do not commit to a future action (updating or notifying someone, sending or scheduling something) unless you perform or schedule it now with a tool call; otherwise remove the commitment from the reply.";

export const blockedUngroundedReplyThought = (explanation: string): string =>
  `Your previous draft reply was not sent to the user: it contained claims that contradict the actual conversation record. ${explanation}. ${ungroundedReplyRetryAdvice}`;

export type GroundingVerdict =
  | { grounded: true }
  | { grounded: false; correctionThought: string };

// Pre-send gate: judges a reply the model is about to conclude its turn with
// against the conversation ground truth, so ungrounded claims (success over a
// failed tool result, actions never performed) are corrected before they reach
// the user rather than after. Fails open: a verifier error or a missing token
// must never silence the agent.
export const verifyConcludingUtterancesGrounded = async (
  spec: AgentInputs & { timezoneIANA?: string },
  history: HistoryEvent[],
  utteranceTexts: string[],
): Promise<GroundingVerdict> => {
  try {
    accessGeminiToken();
  } catch {
    return { grounded: true };
  }
  try {
    const verdict = await checkHallucination(
      [...history, ownUtteranceTurn(utteranceTexts.join("\n"))],
      spec,
      spec.timezoneIANA,
    );
    if (!verdict.isHallucinating) return { grounded: true };
    return {
      grounded: false,
      correctionThought: blockedUngroundedReplyThought(verdict.explanation),
    };
  } catch (e) {
    console.error("grounding verification failed, emitting reply anyway", e);
    return { grounded: true };
  }
};
