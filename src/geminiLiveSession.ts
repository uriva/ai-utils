import { zodToGeminiParameters } from "./gemini.ts";
import type { Tool } from "./agent.ts";

export type LiveAudioChunk = {
  mimeType: string;
  dataBase64: string;
};

export type AudioSessionEvent =
  | { type: "input_transcript"; text: string; finished: boolean }
  | { type: "output_transcript"; text: string; finished: boolean }
  | { type: "thought"; text: string }
  | { type: "audio"; chunk: LiveAudioChunk }
  | {
    type: "tool_call";
    id: string;
    name: string;
    args: Record<string, unknown>;
  }
  | { type: "turn_complete" }
  | { type: "interrupted" };

export type AudioSessionDebugEvent = {
  type: "debug";
  message: string;
};

export type AudioSessionConfig = {
  apiKey: string;
  prompt: string;
  voiceName: string;
  model?: string;
  turnTimeoutMs?: number;
  // deno-lint-ignore no-explicit-any
  tools?: Tool<any>[];
  onDebug?: (event: AudioSessionDebugEvent) => void;
  onSessionEvent?: (event: AudioSessionEvent) => void;
  onClose?: (code: number, reason: string) => void;
};

type LiveFunctionDeclaration = {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
};

type PendingTurn = {
  resolve: (events: AudioSessionEvent[]) => void;
  reject: (error: Error) => void;
  timeout: number;
};

const defaultModel = "models/gemini-2.5-flash-native-audio-preview-12-2025";
const defaultTurnTimeoutMs = 45_000;

const decodeWsData = async (data: string | Blob): Promise<string> =>
  data instanceof Blob ? await data.text() : data;

const mergeTranscript = (current: string, incoming: string) => {
  if (!incoming) return current;
  if (!current) return incoming;
  if (incoming.startsWith(current)) return incoming;
  if (current.startsWith(incoming)) return current;
  if (current.includes(incoming)) return current;
  return `${current}${incoming}`;
};

const toolsToDeclarations = (
  // deno-lint-ignore no-explicit-any
  tools: Tool<any>[] | undefined,
): Array<{ functionDeclarations: LiveFunctionDeclaration[] }> =>
  !tools || tools.length === 0 ? [] : [{
    functionDeclarations: tools.map((
      { name, description, parameters },
    ): LiveFunctionDeclaration => ({
      name,
      description,
      parameters: zodToGeminiParameters(parameters) as unknown as Record<
        string,
        unknown
      >,
    })),
  }];

const consumeTranscriptEvent = (
  events: AudioSessionEvent[],
  type: "input_transcript" | "output_transcript",
  text: string,
  finished: boolean,
  onSessionEvent?: (event: AudioSessionEvent) => void,
) => {
  const existing = [...events].reverse().find((event) => event.type === type);
  if (!existing || existing.type !== type) {
    const event: AudioSessionEvent = { type, text, finished };
    events.push(event);
    onSessionEvent?.(event);
    return;
  }
  existing.text = mergeTranscript(existing.text, text);
  existing.finished = finished;
  onSessionEvent?.({ type, text: existing.text, finished });
};

export type AudioSession = {
  sendText: (text: string) => Promise<AudioSessionEvent[]>;
  sendAudio: (chunk: LiveAudioChunk) => Promise<AudioSessionEvent[]>;
  sendAudioChunks: (chunks: LiveAudioChunk[]) => Promise<AudioSessionEvent[]>;
  streamAudioChunks: (chunks: LiveAudioChunk[]) => void;
  commitTurn: () => void;
  continueTurn: () => Promise<AudioSessionEvent[]>;
  respondToToolCall: (params: {
    id: string;
    name: string;
    response: Record<string, unknown>;
  }) => void;
  close: () => Promise<void>;
};

export const createAudioSession = async ({
  apiKey,
  prompt,
  voiceName,
  model = defaultModel,
  turnTimeoutMs = defaultTurnTimeoutMs,
  tools,
  onDebug,
  onSessionEvent,
  onClose,
}: AudioSessionConfig): Promise<AudioSession> => {
  const ws = new WebSocket(
    `wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key=${apiKey}`,
  );
  let pendingTurn: PendingTurn | undefined;
  let activeTurn = false;
  let bufferedEvents: AudioSessionEvent[] = [];
  let toolCallPending = false;
  let pendingToolCount = 0;
  const isNative = model.includes("native");
  const debug = (message: string) => {
    onDebug?.({ type: "debug", message });
  };

  const addEvent = (event: AudioSessionEvent) => {
    bufferedEvents.push(event);
    onSessionEvent?.(event);
  };

  await new Promise<void>((resolve, reject) => {
    const timeout = setTimeout(() => {
      reject(new Error("Gemini Live setup timed out"));
    }, turnTimeoutMs);
    ws.onopen = () => {
      debug("websocket open");
      const setupMsg = JSON.stringify({
        setup: {
          model,
          generationConfig: {
            responseModalities: ["AUDIO"],
            ...(isNative ? {} : {
              speechConfig: {
                voiceConfig: {
                  prebuiltVoiceConfig: { voiceName },
                },
              },
            }),
          },
          inputAudioTranscription: {},
          outputAudioTranscription: {},
          ...(prompt
            ? { systemInstruction: { parts: [{ text: prompt }] } }
            : {}),
          ...(tools && tools.length > 0
            ? { tools: toolsToDeclarations(tools) }
            : {}),
        },
      });
      ws.send(setupMsg);
      //
    };
    ws.onerror = (error) => {
      clearTimeout(timeout);
      reject(error instanceof Error ? error : new Error(String(error)));
    };
    ws.onclose = (e) => {
      debug(`close: code=${e.code} reason=${e.reason}`);
      clearTimeout(timeout);
      reject(new Error("Gemini Live closed before setupComplete"));
    };
    ws.onmessage = async (event) => {
      const msg = JSON.parse(await decodeWsData(event.data));
      if (msg.error) {
        debug(`setup error: ${JSON.stringify(msg.error)}`);
        clearTimeout(timeout);
        reject(new Error(JSON.stringify(msg.error)));
        return;
      }
      if (msg.setupComplete) {
        debug("setup complete");
        clearTimeout(timeout);
        resolve();
      }
    };
  });

  const flushEvents = () => {
    if (!pendingTurn || bufferedEvents.length === 0) return;
    clearTimeout(pendingTurn.timeout);
    const events = bufferedEvents;
    bufferedEvents = [];
    pendingTurn.resolve(events);
    pendingTurn = undefined;
  };

  const rejectPendingTurn = (error: Error) => {
    if (!pendingTurn) return;
    clearTimeout(pendingTurn.timeout);
    pendingTurn.reject(error);
    pendingTurn = undefined;
  };

  ws.onclose = (e) => {
    console.log(
      `[audio-tool] Gemini WS closed: code=${e.code} reason=${e.reason}`,
    );
    rejectPendingTurn(
      new Error(`Gemini WS closed: code=${e.code} reason=${e.reason}`),
    );
    onClose?.(e.code, e.reason);
  };

  ws.onerror = (e) => {
    console.error(
      `[audio-tool] Gemini WS error: ${
        e instanceof Error ? e.message : "unknown"
      }`,
    );
  };

  ws.onmessage = async (event) => {
    const msg = JSON.parse(await decodeWsData(event.data));
    if (msg.error) {
      debug(`message error: ${JSON.stringify(msg.error)}`);
      rejectPendingTurn(new Error(JSON.stringify(msg.error)));
      activeTurn = false;
      bufferedEvents = [];
      return;
    }
    const content = msg.serverContent;
    if (content?.inputTranscription?.text) {
      debug(`input transcription: ${content.inputTranscription.text}`);
      consumeTranscriptEvent(
        bufferedEvents,
        "input_transcript",
        content.inputTranscription.text,
        !!content.inputTranscription.finished,
        onSessionEvent,
      );
    }
    if (content?.outputTranscription?.text) {
      debug(`output transcription: ${content.outputTranscription.text}`);
      consumeTranscriptEvent(
        bufferedEvents,
        "output_transcript",
        content.outputTranscription.text,
        !!content.outputTranscription.finished,
        onSessionEvent,
      );
    }
    const parts = content?.modelTurn?.parts ?? [];
    for (const part of parts) {
      if (part.inlineData?.data) {
        debug(
          `audio chunk from model: ${part.inlineData.mimeType ?? "audio/pcm"}`,
        );
        addEvent({
          type: "audio",
          chunk: {
            mimeType: part.inlineData.mimeType ?? "audio/pcm;rate=24000",
            dataBase64: part.inlineData.data,
          },
        });
      }
      if (part.text) {
        if (part.thought) {
          addEvent({ type: "thought", text: part.text });
        } else {
          consumeTranscriptEvent(
            bufferedEvents,
            "output_transcript",
            part.text,
            false,
            onSessionEvent,
          );
        }
      }
    }
    const functionCalls = msg.toolCall?.functionCalls ?? [];
    for (const functionCall of functionCalls) {
      if (!functionCall.id || !functionCall.name) continue;
      debug(`tool call: ${functionCall.name}`);
      addEvent({
        type: "tool_call",
        id: functionCall.id,
        name: functionCall.name,
        args: functionCall.args ?? {},
      });
    }
    if (functionCalls.length > 0) {
      toolCallPending = true;
      pendingToolCount = functionCalls.length;
      flushEvents();
      return;
    }
    if (content?.interrupted) {
      debug("turn interrupted");
      addEvent({ type: "interrupted" });
      flushEvents();
      return;
    }
    if (content?.turnComplete) {
      debug("turn complete");
      activeTurn = false;
      addEvent({ type: "turn_complete" });
      flushEvents();
    }
  };

  const waitForEvents = () => {
    if (
      ws.readyState === WebSocket.CLOSED || ws.readyState === WebSocket.CLOSING
    ) {
      return Promise.reject(new Error("WebSocket is closed"));
    }
    return new Promise<AudioSessionEvent[]>((resolve, reject) => {
      if (pendingTurn) {
        clearTimeout(pendingTurn.timeout);
        pendingTurn.reject(new Error("Overwritten by new turn"));
      }
      const timeout = setTimeout(() => {
        debug("turn timed out");
        reject(new Error("Gemini Live turn timed out"));
      }, turnTimeoutMs);
      pendingTurn = { resolve, reject, timeout };
      flushEvents();
    });
  };

  return {
    sendText: async (text: string) => {
      debug(`sendText: ${text}`);
      activeTurn = true;
      const wait = waitForEvents();
      ws.send(JSON.stringify({
        clientContent: {
          turns: [{ role: "user", parts: [{ text }] }],
          turnComplete: true,
        },
      }));
      return await wait;
    },
    sendAudio: async ({ mimeType, dataBase64 }: LiveAudioChunk) => {
      debug(`sendAudio: ${mimeType} bytes=${dataBase64.length}`);
      activeTurn = true;
      const wait = waitForEvents();
      ws.send(JSON.stringify({
        realtimeInput: {
          mediaChunks: [{ mimeType, data: dataBase64 }],
        },
      }));
      return await wait;
    },
    sendAudioChunks: async (chunks: LiveAudioChunk[]) => {
      debug(
        `sendAudioChunks: count=${chunks.length} firstMime=${
          chunks[0]?.mimeType ?? "none"
        }`,
      );
      activeTurn = true;
      const wait = waitForEvents();
      for (const chunk of chunks) {
        ws.send(JSON.stringify({
          realtimeInput: {
            mediaChunks: [{ mimeType: chunk.mimeType, data: chunk.dataBase64 }],
          },
        }));
        await new Promise((resolve) => setTimeout(resolve, 40));
      }
      if (!isNative) {
        ws.send(JSON.stringify({
          clientContent: { turns: [], turnComplete: true },
        }));
      }
      return await wait;
    },
    streamAudioChunks: (chunks: LiveAudioChunk[]) => {
      if (toolCallPending) return;
      activeTurn = true;
      for (const chunk of chunks) {
        const payload = {
          realtimeInput: {
            mediaChunks: [{ mimeType: chunk.mimeType, data: chunk.dataBase64 }],
          },
        };
        ws.send(JSON.stringify(payload));
      }
    },
    commitTurn: () => {
      if (isNative) return;
      if (toolCallPending) return;
      debug("commitTurn manually triggered");
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
          clientContent: { turns: [], turnComplete: true },
        }));
      }
    },
    continueTurn: async () => {
      if (!activeTurn) return [];
      return await waitForEvents();
    },
    respondToToolCall: ({ id, name, response }: {
      id: string;
      name: string;
      response: Record<string, unknown>;
    }) => {
      if (ws.readyState !== WebSocket.OPEN) {
        console.error(
          `[audio-tool] respondToToolCall(${name}): WebSocket not open (readyState=${ws.readyState})`,
        );
        return;
      }
      ws.send(JSON.stringify({
        toolResponse: {
          functionResponses: [{ id, name, response }],
        },
      }));
      pendingToolCount = Math.max(0, pendingToolCount - 1);
      if (pendingToolCount === 0) toolCallPending = false;
    },
    close: async () => {
      if (ws.readyState === WebSocket.CLOSED) return;
      if (pendingTurn) {
        clearTimeout(pendingTurn.timeout);
        pendingTurn.resolve(bufferedEvents);
        pendingTurn = undefined;
      }
      await new Promise<void>((resolve) => {
        ws.onclose = () => resolve();
        ws.close();
      });
    },
  };
};
