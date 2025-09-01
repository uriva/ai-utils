import {
  type Content,
  type FunctionCall,
  type GenerateContentParameters,
  type GenerateContentResponse,
  GoogleGenAI,
  type Part,
} from "@google/genai";
import { context } from "context-inject";
import { coerce, empty, groupBy, map, pipe, retry } from "gamla";
import {
  type AgentSpec,
  doNothingEvent,
  generateId,
  type HistoryEventWithMetadata,
  type MessageId,
  ownUtteranceTurn,
  ownUtteranceTurnWithMetadata,
  type Tool,
  toolUseTurn,
  toolUseTurnWithMetadata,
} from "./agent.ts";
import { makeCache } from "./cacher.ts";
import {
  accessGeminiToken,
  geminiFlashVersion,
  geminiProVersion,
  zodToGeminiParameters,
} from "./gemini.ts";

const geminiError = context((_1: Error, _2: GenerateContentParameters) => {});

export const injectGeminiErrorLogger = geminiError.inject;

type GeminiOutput = GeminiPartOfInterest[];

const callGemini = (req: GenerateContentParameters): Promise<GeminiOutput> =>
  new GoogleGenAI({ apiKey: accessGeminiToken() }).models.generateContent(req)
    .then((resp: GenerateContentResponse): GeminiOutput =>
      (resp.candidates?.[0]?.content?.parts ?? [])
        .flatMap((part: Part): GeminiOutput => {
          const { text, functionCall, thoughtSignature } = part;
          if (functionCall) {
            return [{ type: "function_call", functionCall, thoughtSignature }];
          }
          if (typeof text === "string") {
            return [{ type: "text", text, thoughtSignature }];
          }
          return [];
        })
    ).catch((err) => {
      geminiError.access(err, req);
      throw err;
    });

// deno-lint-ignore no-explicit-any
const actionToTool = ({ name, description, parameters }: Tool<any>) => ({
  name,
  description,
  parameters: zodToGeminiParameters(parameters),
});

const historyEventToContent =
  (eventById: (id: string) => GeminiHistoryEvent) =>
  (e: GeminiHistoryEvent): Content => {
    if (e.type === "participant_utterance") {
      return wrapUserContent([{ text: `${e.name}: ${e.text}` }]);
    }
    if (e.type === "own_utterance") {
      return wrapModelContent([{
        thoughtSignature: e.modelMetadata?.thoughtSignature,
        text: e.text,
      }]);
    }
    if (e.type === "tool_call") {
      return wrapModelContent([{
        thoughtSignature: e.modelMetadata?.thoughtSignature,
        functionCall: {
          name: e.name,
          args: e.parameters as Record<string, unknown>,
        },
      }]);
    }
    if (e.type === "tool_result") {
      return wrapUserContent(
        [{
          functionResponse: { name: e.name, response: { result: e.result } },
        }],
      );
    }
    if (e.type === "own_reaction") {
      const msg = eventById(e.onMessage);
      const text = typeof msg === "object" && "text" in msg ? msg.text : "";
      return wrapModelContent([{
        thoughtSignature: e.modelMetadata?.thoughtSignature,
        text: `You reacted ${e.reaction} to message: ${text.slice(0, 100)}`,
      }]);
    }
    if (e.type === "participant_reaction") {
      const msg = eventById(e.onMessage);
      const text = typeof msg === "object" && "text" in msg ? msg.text : "";
      return wrapUserContent([{
        text: `${e.name} reacted ${e.reaction} to message: ${
          text.slice(0, 100)
        }`,
      }]);
    }
    if (e.type === "do_nothing") {
      // Carry thoughtSignature if available (assume only one text part)
      return wrapModelContent([{
        text: "",
        thoughtSignature: e.modelMetadata?.thoughtSignature,
      }]);
    }
    throw new Error(
      `Unknown history event type: ${JSON.stringify(e, null, 2)}`,
    );
  };

const combineContent = (contents: Content[]): Content => ({
  role: contents.some((c) => c.role === "model") ? "model" : "user",
  parts: contents.flatMap((c) => c.parts ?? []),
});

const wrapRole = (role: "user" | "model") => (parts: Part[]): Content => ({
  role,
  parts,
});

const wrapModelContent = wrapRole("model");

const wrapUserContent = wrapRole("user");

const getOriginalId = (e: GeminiHistoryEvent): string =>
  "modelMetadata" in e ? e.modelMetadata?.responseId ?? e.id : e.id;

const indexById = (events: GeminiHistoryEvent[]) => {
  const eventIdToEvents = groupBy(({ id }: GeminiHistoryEvent) => id)(events);
  return (id: MessageId) => coerce(eventIdToEvents[id]?.[0]);
};

type GeminiFunctiontoolPart = {
  type: "function_call";
  functionCall: FunctionCall;
  thoughtSignature?: string;
};

type GeminiHistoryEvent = HistoryEventWithMetadata<{
  type: "gemini";
  thoughtSignature: string;
  responseId: string;
}>;

type GeminiPartOfInterest =
  | { type: "text"; text: string; thoughtSignature?: string }
  | GeminiFunctiontoolPart;

const sawFunction = (output: GeminiOutput) =>
  output.some(({ type }: GeminiPartOfInterest) => type === "function_call");

const didNothing = (output: GeminiOutput) =>
  !sawFunction(output) &&
  !output.some((p: GeminiPartOfInterest) => p.type === "text" && p.text);

export const geminiAgentCaller = ({ lightModel, prompt, tools }: AgentSpec) =>
  pipe(
    (historyOuter: GeminiHistoryEvent[]) => {
      const eventById = indexById(historyOuter);
      const grouped = Object.values(groupBy(getOriginalId)(historyOuter));
      const history = grouped.map(
        pipe(map(historyEventToContent(eventById)), combineContent),
      );
      if (empty(history) || history[0].role !== "user") {
        history.unshift({
          role: "user",
          parts: [{ text: "<conversation started>" }],
        });
      }
      return history;
    },
    (contents: Content[]): GenerateContentParameters => ({
      model: lightModel ? geminiFlashVersion : geminiProVersion,
      config: {
        systemInstruction: prompt,
        tools: [{ functionDeclarations: tools.map(actionToTool) }],
        // Only set allowedFunctionNames if mode is ANY. Since mode is not set, omit allowedFunctionNames.
        toolConfig: {
          functionCallingConfig: {
            // allowedFunctionNames: actions.map((a) => a.name),
          },
        },
      },
      contents,
    }),
    makeCache("gemini response with function calls v5")(
      // August 31st 2025, gemini returns frequent 500 errors.
      retry(1000, 2, callGemini),
    ),
    (geminiOutput: GeminiOutput): GeminiHistoryEvent[] => {
      const responseId = generateId();
      if (didNothing(geminiOutput)) {
        const textPart = geminiOutput.find((p) =>
          p.type === "text" && p.thoughtSignature
        );
        return [doNothingEvent(
          textPart?.thoughtSignature
            ? {
              type: "gemini",
              responseId,
              thoughtSignature: textPart.thoughtSignature,
            }
            : undefined,
        )];
      }
      return geminiOutput.map(geminiOutputPartToHistoryEvent(responseId));
    },
  );

const geminiOutputPartToHistoryEvent =
  (responseId: string) => (p: GeminiPartOfInterest): GeminiHistoryEvent => {
    if (p.type === "text" && p.text) {
      return p.thoughtSignature
        ? ownUtteranceTurnWithMetadata(p.text, {
          type: "gemini",
          responseId,
          thoughtSignature: p.thoughtSignature,
        })
        : ownUtteranceTurn(p.text);
    }
    if (p.type === "function_call") {
      return p.thoughtSignature
        ? toolUseTurnWithMetadata(p.functionCall, {
          type: "gemini",
          responseId,
          thoughtSignature: p.thoughtSignature,
        })
        : toolUseTurn(p.functionCall);
    }
    throw new Error(`Unknown part type: ${JSON.stringify(p)}`);
  };
