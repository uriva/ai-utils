import {
  accessOutputEvent,
  type AgentSpec,
  handleFunctionCalls,
  type HistoryEvent,
  ownThoughtTurn,
  ownUtteranceTurn,
  participantEditMessageTurn,
  type Tool,
  toolUseTurnWithMetadata,
} from "./agent.ts";
import type { DuplexEndpoint } from "./duplex.ts";
import {
  type AudioSessionEvent,
  createAudioSession,
  type LiveAudioChunk,
} from "./geminiLiveSession.ts";
import { accessGeminiToken } from "./gemini.ts";

export const runAudioTransportAgent = async (
  spec: AgentSpec,
  customOutputEvent?: (event: HistoryEvent) => Promise<void>,
): Promise<void> => {
  if (!spec.transport || spec.transport.kind !== "audio") {
    throw new Error("audio transport required");
  }
  return await runAudioAgentLoop(
    spec,
    spec.transport.endpoint,
    customOutputEvent ?? accessOutputEvent,
  );
};

export const transcriptOf = (
  events: AudioSessionEvent[],
  type: "input_transcript" | "output_transcript",
) =>
  (events.findLast((event) => event.type === type) as
    | Extract<AudioSessionEvent, { type: typeof type }>
    | undefined)?.text?.trim() ?? "";

const stripReasoningPreamble = (text: string) => {
  const withoutMarkdownHeading = text.replace(/^\*\*[^*]+\*\*\s*/s, "");
  const paragraphs = withoutMarkdownHeading.split(/\n\n+/).map((part) =>
    part.trim()
  ).filter(Boolean);
  if (paragraphs.length <= 1) return withoutMarkdownHeading.trim();
  return paragraphs.at(-1) ?? withoutMarkdownHeading.trim();
};

const base64ToBytes = (dataBase64: string) =>
  Uint8Array.from(atob(dataBase64), (char) => char.charCodeAt(0));

const bytesToBase64 = (bytes: Uint8Array) => {
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
};

const concatBytes = (chunks: Uint8Array[]) => {
  const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const merged = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  return merged;
};

const resample24kTo16k = (pcm24k: Uint8Array) => {
  const input = new Int16Array(
    pcm24k.buffer,
    pcm24k.byteOffset,
    Math.floor(pcm24k.byteLength / 2),
  );
  const outLength = Math.floor(input.length * 16000 / 24000);
  const output = new Int16Array(outLength);
  for (let i = 0; i < outLength; i++) {
    output[i] = input[Math.floor(i * 24000 / 16000)] ?? 0;
  }
  return new Uint8Array(output.buffer.slice(0));
};

export const spokenReplyOnly = (text: string) => {
  const stripped = stripReasoningPreamble(text);
  if (
    stripped.includes("triggering the immediate use of") &&
    !stripped.includes("relay code is")
  ) {
    return "";
  }
  if (stripped.includes("Now, I will") && !stripped.includes("relay code is")) {
    return "";
  }
  return stripped;
};

type SessionEventCallbacks = {
  onAudio: (chunk: LiveAudioChunk) => void;
  onUtterance: (text: string) => void;
  onFlush: () => void;
  onTurnOutput: (
    sessionOutput: AudioSessionEvent[],
    wasInterrupted: boolean,
  ) => void;
};

export const makeSessionEventHandler = (
  callbacks: SessionEventCallbacks,
) => {
  let interrupted = false;
  let latestOutputTranscript = "";
  let turnEvents: AudioSessionEvent[] = [];
  return (event: AudioSessionEvent) => {
    if (event.type === "audio") {
      callbacks.onAudio(event.chunk);
    }
    turnEvents.push(event);
    if (event.type === "output_transcript") {
      latestOutputTranscript = event.text;
      if (event.finished && !interrupted) {
        callbacks.onUtterance(event.text);
        latestOutputTranscript = "";
      }
    }
    if (event.type === "interrupted") {
      interrupted = true;
      latestOutputTranscript = "";
      callbacks.onFlush();
    }
    if (
      event.type === "turn_complete" ||
      event.type === "interrupted" ||
      event.type === "tool_call"
    ) {
      if (
        event.type === "turn_complete" && !interrupted &&
        latestOutputTranscript
      ) {
        callbacks.onUtterance(latestOutputTranscript);
      }
      const sessionOutput = turnEvents;
      turnEvents = [];
      callbacks.onTurnOutput(sessionOutput, interrupted);
      if (event.type !== "tool_call") {
        interrupted = false;
        latestOutputTranscript = "";
      }
    }
  };
};

const audioToolTimeoutMs = 60_000;

const respondWithError = (
  session: Awaited<ReturnType<typeof createAudioSession>>,
  sessionOutput: AudioSessionEvent[],
  respondedGeminiIds: Set<string>,
  errorMsg: string,
) => {
  for (
    const tc of sessionOutput.filter(
      (ev): ev is Extract<AudioSessionEvent, { type: "tool_call" }> =>
        ev.type === "tool_call",
    )
  ) {
    if (!respondedGeminiIds.has(tc.id)) {
      session.respondToToolCall({
        id: tc.id,
        name: tc.name,
        response: { result: errorMsg },
      });
    }
  }
};

const resolveToolCalls = async (
  session: Awaited<ReturnType<typeof createAudioSession>>,
  // deno-lint-ignore no-explicit-any
  tools: Tool<any>[],
  sessionOutput: AudioSessionEvent[],
) => {
  const geminiIdByHistoryId = new Map<string, string>();
  const respondedGeminiIds = new Set<string>();
  const toolCallEvents = sessionOutput
    .filter((e): e is Extract<AudioSessionEvent, { type: "tool_call" }> =>
      e.type === "tool_call"
    )
    .map((e) => {
      const event = toolUseTurnWithMetadata(
        { id: e.id, name: e.name, args: e.args },
        undefined,
      );
      geminiIdByHistoryId.set(event.id, e.id);
      return event;
    });
  if (toolCallEvents.length === 0) return;
  const toolNames = sessionOutput
    .filter((e): e is Extract<AudioSessionEvent, { type: "tool_call" }> =>
      e.type === "tool_call"
    )
    .map((e) => e.name);
  console.log(
    `[audio-tool] resolveToolCalls start: ${toolCallEvents.length} calls: ${
      toolNames.join(", ")
    }`,
  );
  const startTime = Date.now();
  try {
    await Promise.race([
      handleFunctionCalls(tools, (event) => {
        if (event.type === "tool_result") {
          const geminiId = geminiIdByHistoryId.get(event.toolCallId!);
          const id = geminiId ?? event.toolCallId!;
          respondedGeminiIds.add(id);
          console.log(
            `[audio-tool] tool result for ${event.name} after ${
              Date.now() - startTime
            }ms, responding to Gemini`,
          );
          session.respondToToolCall({
            id,
            name: event.name,
            response: { result: event.result },
          });
        }
      })(toolCallEvents),
      new Promise<never>((_, reject) =>
        setTimeout(
          () => reject(new Error("Tool execution timed out")),
          audioToolTimeoutMs,
        )
      ),
    ]);
    console.log(
      `[audio-tool] resolveToolCalls completed in ${Date.now() - startTime}ms`,
    );
  } catch (e) {
    console.error(
      `[audio-tool] resolveToolCalls error after ${Date.now() - startTime}ms: ${
        e instanceof Error ? e.message : String(e)
      }`,
    );
    respondWithError(
      session,
      sessionOutput,
      respondedGeminiIds,
      `Error: ${e instanceof Error ? e.message : String(e)}`,
    );
  }
};
const emitNonUtteranceEvents = async (
  outputEvent: (event: HistoryEvent) => Promise<void>,
  sessionOutput: AudioSessionEvent[],
) => {
  for (const event of sessionOutput) {
    if (event.type === "thought") {
      await outputEvent(ownThoughtTurn(event.text));
    }
    if (event.type !== "tool_call") continue;
    await outputEvent(toolUseTurnWithMetadata({
      id: event.id,
      name: event.name,
      args: event.args,
    }, undefined));
  }
};

export const runAudioAgentLoop = async (
  spec: AgentSpec,
  endpoint: DuplexEndpoint,
  outputEvent: (event: HistoryEvent) => Promise<void>,
) => {
  if (!spec.transport || spec.transport.kind !== "audio") {
    throw new Error("audio transport required");
  }
  const transport = spec.transport;
  let isClosed = false;
  let audioInputBuffer = new Uint8Array(0);

  const emitUtterance = (text: string) => {
    const spoken = spokenReplyOnly(text);
    if (spoken.length === 0) return;
    void outputEvent(ownUtteranceTurn(spoken));
    void endpoint.sendData({
      type: "text" as const,
      text: spoken,
      from: transport.participantName,
    });
  };

  let loggedFirstAudioIn = false;
  let loggedFirstAudioOut = false;

  const session = await createAudioSession({
    apiKey: accessGeminiToken(),
    prompt: spec.prompt,
    voiceName: transport.voiceName,
    tools: spec.tools,
    onSessionEvent: makeSessionEventHandler({
      onAudio: (chunk) => {
        if (!loggedFirstAudioOut) {
          loggedFirstAudioOut = true;
          console.log("[audio] first output audio from Gemini");
        }
        void endpoint.sendData({
          type: "audio",
          chunks: [{ mimeType: chunk.mimeType, dataBase64: chunk.dataBase64 }],
          from: transport.participantName,
        });
      },
      onUtterance: emitUtterance,
      onFlush: () => {
        void endpoint.sendData({
          type: "flush",
          from: transport.participantName,
        });
      },
      onTurnOutput: (sessionOutput, wasInterrupted) => {
        processTurnOutput(sessionOutput, wasInterrupted).catch((e) =>
          console.error("processTurnOutput error:", e)
        );
      },
    }),
  });
  console.log("[audio] session created");

  const processTurnOutput = async (
    sessionOutput: AudioSessionEvent[],
    wasInterrupted: boolean,
  ) => {
    const heard = transcriptOf(sessionOutput, "input_transcript");
    if (heard.length > 0) {
      await outputEvent(participantEditMessageTurn({
        name: transport.participantName,
        text: heard,
        onMessage: crypto.randomUUID(),
      }));
    }

    if (!wasInterrupted) {
      await emitNonUtteranceEvents(
        outputEvent,
        sessionOutput,
      );
    }
    await resolveToolCalls(
      session,
      spec.tools,
      sessionOutput,
    );
  };

  let vadTimeout: number | undefined;

  await new Promise<void>((resolve) => {
    endpoint.onData(async (message) => {
      if (isClosed) return;
      if (message.type === "close") {
        console.log("[audio] session closed");
        isClosed = true;
        clearTimeout(vadTimeout);
        await session.close();
        resolve();
        return;
      }
      try {
        if (message.type === "text") {
          // sendText resolves with the output, but we already process it via onSessionEvent
          // so we just catch errors if any
          session.sendText(message.text).catch(() => {});
        } else if (message.type === "audio") {
          if (!loggedFirstAudioIn) {
            loggedFirstAudioIn = true;
            console.log("[audio] first input audio received");
          }
          // Buffer incoming audio to send in 3200-byte chunks
          const incomingBytes = concatBytes([
            ...message.chunks.map(({ mimeType, dataBase64 }) =>
              mimeType.includes("24000")
                ? resample24kTo16k(base64ToBytes(dataBase64))
                : base64ToBytes(dataBase64)
            ),
          ]);
          audioInputBuffer = concatBytes([audioInputBuffer, incomingBytes]);

          const chunksToStream: LiveAudioChunk[] = [];
          while (audioInputBuffer.length >= 3200) {
            const piece = audioInputBuffer.slice(0, 3200);
            audioInputBuffer = audioInputBuffer.slice(3200);
            chunksToStream.push({
              mimeType: "audio/pcm;rate=16000",
              dataBase64: bytesToBase64(piece),
            });
          }
          if (chunksToStream.length > 0) {
            // Check volume of first chunk to see if it's silence
            const testBuf = new Int16Array(
              base64ToBytes(chunksToStream[0].dataBase64).buffer,
            );
            let sumSq = 0;
            for (let i = 0; i < testBuf.length; i++) {
              sumSq += testBuf[i] * testBuf[i];
            }
            const rms = Math.sqrt(sumSq / testBuf.length);

            if (rms > 250) {
              // Reset VAD timeout only if there is ACTUAL audio (RMS > 250)
              clearTimeout(vadTimeout);
              vadTimeout = globalThis.setTimeout(() => {
                session.commitTurn();
              }, 1500) as unknown as number;
            } else if (!vadTimeout) {
              // If no timeout is running, start one just in case we never see loud audio again
              vadTimeout = globalThis.setTimeout(() => {
                session.commitTurn();
              }, 1500) as unknown as number;
            }

            session.streamAudioChunks(chunksToStream);
          }
        }
      } catch (error) {
        await outputEvent(ownThoughtTurn(
          `Audio transport error for ${transport.participantName}: ${
            error instanceof Error ? error.message : String(error)
          }`,
        ));
      }
    });
  });
};
