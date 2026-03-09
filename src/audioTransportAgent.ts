import { empty } from "gamla";
import {
  accessOutputEvent,
  type AgentSpec,
  type HistoryEvent,
  ownThoughtTurn,
  ownUtteranceTurn,
  participantEditMessageTurn,
  participantUtteranceTurn,
  type RegularTool,
  type Tool,
  toolResultTurn,
  type ToolReturn,
  toolUseTurnWithMetadata,
} from "./agent.ts";
import type { DuplexEndpoint, DuplexMessage } from "./duplex.ts";
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

const transcriptOf = (
  events: AudioSessionEvent[],
  type: "input_transcript" | "output_transcript",
) =>
  events.filter((
    event,
  ): event is Extract<AudioSessionEvent, { type: typeof type }> =>
    event.type === type
  ).map((event) => event.text).join(" ").trim();

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

const spokenReplyOnly = (text: string) => {
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

const regularTools = (
  // deno-lint-ignore no-explicit-any
  tools: Tool<any>[],
  // deno-lint-ignore no-explicit-any
): RegularTool<any>[] =>
  // deno-lint-ignore no-explicit-any
  tools.filter((tool): tool is RegularTool<any> => !tool.isDeferred);

const findRegularTool = (
  tools: ReturnType<typeof regularTools>,
  name: string,
) => tools.find((tool) => tool.name === name);

const emitModelEvents = async (
  outputEvent: (event: HistoryEvent) => Promise<void>,
  participantName: string,
  sessionOutput: AudioSessionEvent[],
) => {
  const heard = transcriptOf(sessionOutput, "input_transcript");
  if (heard.length > 0) {
    await outputEvent(
      participantUtteranceTurn({ name: participantName, text: heard }),
    );
  }
  const spoken = spokenReplyOnly(
    transcriptOf(sessionOutput, "output_transcript"),
  );
  const audioEvents = sessionOutput.filter((
    event,
  ): event is Extract<AudioSessionEvent, { type: "audio" }> =>
    event.type === "audio"
  );
  const attachments = audioEvents.map((event) => ({
    kind: "inline" as const,
    mimeType: event.chunk.mimeType,
    dataBase64: event.chunk.dataBase64,
  }));
  if (spoken.length > 0 || !empty(attachments)) {
    await outputEvent(ownUtteranceTurn(
      spoken,
      empty(attachments) ? undefined : attachments,
    ));
  }
  for (const event of sessionOutput) {
    if (event.type !== "tool_call") continue;
    await outputEvent(toolUseTurnWithMetadata({
      id: event.id,
      name: event.name,
      args: event.args,
    }, undefined));
  }
};

const resolveToolCalls = async (
  outputEvent: (event: HistoryEvent) => Promise<void>,
  session: Awaited<ReturnType<typeof createAudioSession>>,
  tools: ReturnType<typeof regularTools>,
  sessionOutput: AudioSessionEvent[],
) => {
  for (const event of sessionOutput) {
    if (event.type !== "tool_call") continue;
    const tool = findRegularTool(tools, event.name);
    if (!tool) continue;
    const result: string | ToolReturn = await tool.handler(event.args);
    const rendered = typeof result === "string"
      ? { result, attachments: undefined }
      : { result: result.result, attachments: result.attachments };
    await outputEvent(toolResultTurn({
      name: event.name,
      result: rendered.result,
      attachments: rendered.attachments,
      toolCallId: event.id,
    }));
    session.respondToToolCall({
      id: event.id,
      name: event.name,
      response: { result: rendered.result },
    });
  }
};

const sessionOutputToMessages = (
  from: string,
  sessionOutput: AudioSessionEvent[],
): DuplexMessage[] => {
  // We no longer send audio here because audio is streamed instantly via onSessionEvent.
  const textMessages = sessionOutput.filter((
    event,
  ): event is Extract<AudioSessionEvent, { type: "output_transcript" }> =>
    event.type === "output_transcript"
  ).map((event) => spokenReplyOnly(event.text)).filter((text) =>
    text.length > 0
  ).map((text) => ({
    type: "text" as const,
    text,
    from,
  }));
  return [
    ...textMessages,
  ];
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

  let turnEvents: AudioSessionEvent[] = [];
  let audioInputBuffer = new Uint8Array(0);

  const session = await createAudioSession({
    apiKey: accessGeminiToken(),
    prompt: spec.prompt,
    voiceName: transport.voiceName,
    tools: spec.tools,
    onDebug: ({ message }) => {
      void outputEvent(
        ownThoughtTurn(`audio-debug ${transport.participantName}: ${message}`),
      );
    },
    onSessionEvent: (event) => {
      if (event.type === "audio") {
        void endpoint.sendData({
          type: "audio",
          chunks: [
            {
              mimeType: event.chunk.mimeType,
              dataBase64: event.chunk.dataBase64,
            },
          ],
          from: transport.participantName,
        });
      }
      turnEvents.push(event);
      if (
        event.type === "turn_complete" ||
        event.type === "interrupted" ||
        event.type === "tool_call"
      ) {
        const sessionOutput = turnEvents;
        turnEvents = [];
        void processTurnOutput(sessionOutput);
      }
    },
  });
  const tools = regularTools(spec.tools);

  const processTurnOutput = async (sessionOutput: AudioSessionEvent[]) => {
    // We already sent audio instantly via onSessionEvent, so filter it out here
    // for emitModelEvents to avoid duplicate logs, or just let emitModelEvents
    // log it (emitModelEvents currently handles transcript and tool_call logging).
    // Actually emitModelEvents builds attachments from audio.
    const heard = transcriptOf(sessionOutput, "input_transcript");
    if (heard.length > 0) {
      await outputEvent(participantEditMessageTurn({
        name: transport.participantName,
        text: heard,
        onMessage: crypto.randomUUID(),
      }));
    }

    // For emitModelEvents, we still pass the output to get text / tool_calls / audio logged
    await emitModelEvents(
      outputEvent,
      transport.participantName,
      sessionOutput,
    );
    await resolveToolCalls(
      outputEvent,
      session,
      tools,
      sessionOutput,
    );

    const outgoingMessages = sessionOutputToMessages(
      transport.participantName,
      sessionOutput,
    );
    await Promise.all(outgoingMessages.map(endpoint.sendData));
  };

  let vadTimeout: number | undefined;

  await new Promise<void>((resolve) => {
    endpoint.onData(async (message) => {
      if (isClosed) return;
      if (message.type === "close") {
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
