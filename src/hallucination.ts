import { genJson } from "./genJson.ts";
import {
  type AgentInputs,
  getSpecForTurn,
  type HistoryEvent,
} from "./agent.ts";
import { eventsToPlainText } from "./compaction.ts";
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
- If the bot claimed it performed an action (e.g. calling a tool) but hasn't yet, the note should simply advise the bot to go ahead and perform that action now — no correction or apology needed.
- If the bot stated incorrect information, the note should advise the bot to use the ${editLastMessageToolName} tool to correct the message, or if that fails, to gently correct itself in a follow-up (e.g., "I sent an inaccurate message. I should use ${editLastMessageToolName} to fix it, or follow up with a correction like 'sorry, I meant...'").
Make sure to phrase this note as if the bot is writing it to itself.`;

export const createHallucinationCheckPrompt = (
  history: HistoryEvent[],
  spec: AgentInputs,
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
  const serializedHistory = eventsToPlainText(contextHistory);

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

Analyze the bot's response carefully. Only flag a hallucination if ALL of the following are true:
1. The response contains specific factual claims or third-party links/URLs
2. Those claims are NOT supported by the system instructions, available tools, or conversation history
3. The fabrication would meaningfully derail the conversation

Do NOT flag a hallucination if:
- The information is reasonably correct common knowledge
- The information is supported by any tool_result, own_thought, or external_event in the history (even older ones)
- The bot is paraphrasing, summarizing, or making directly implied logical inferences from the ground truth

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
): Promise<HallucinationCheckResult> => {
  const checkPrompt = createHallucinationCheckPrompt(history, spec);
  return await callModel(checkPrompt);
};
