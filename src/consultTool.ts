import { z } from "zod/v4";
import {
  accessHistory,
  type CallModel,
  type HistoryEvent,
  participantUtteranceTurn,
  type Tool,
} from "./agent.ts";

export const consultToolName = "consult";

const consultParameters = z.object({
  question: z.string().describe(
    "The question or advice request to send to the stronger model. Include enough context for it to be actionable.",
  ),
});

const formatStrongModelReply = (events: HistoryEvent[]) =>
  events
    .filter((e): e is Extract<HistoryEvent, { type: "own_utterance" }> =>
      e.type === "own_utterance"
    )
    .map((e) => e.text)
    .filter((t) => t.length > 0)
    .join("\n\n");

export const createConsultTool = (
  strongCallModel: CallModel,
): Tool<typeof consultParameters> => ({
  name: consultToolName,
  description:
    "Ask the stronger model in your family for advice on the current conversation. The stronger model receives the full conversation history plus your question and returns guidance. Use this for hard reasoning, ambiguous decisions, or when you are unsure.",
  parameters: consultParameters,
  handler: async ({ question }) => {
    const history = await accessHistory();
    const withQuestion: HistoryEvent[] = [
      ...history,
      participantUtteranceTurn({
        name: "weaker_model",
        text:
          `I am the weaker model handling this conversation and I need advice. ${question}`,
      }),
    ];
    const reply = await strongCallModel(withQuestion);
    const text = formatStrongModelReply(reply);
    return text || "[stronger model returned no text]";
  },
});
