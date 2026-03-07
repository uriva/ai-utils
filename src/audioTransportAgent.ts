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
} from "./geminiLiveSession.ts";
import { accessGeminiToken } from "./gemini.ts";

export const runAudioTransportAgent = async (spec: AgentSpec) => {
  if (!spec.transport || spec.transport.kind !== "audio") {
    throw new Error("audio transport required");
  }
  return await runAudioAgentLoop(
    spec,
    spec.transport.endpoint,
    accessOutputEvent,
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

const splitBytes = (bytes: Uint8Array, chunkSize: number) => {
  const chunks: Uint8Array[] = [];
  for (let i = 0; i < bytes.length; i += chunkSize) {
    chunks.push(bytes.slice(i, i + chunkSize));
  }
  return chunks;
};

const silence16k = (ms: number) =>
  new Uint8Array(Math.floor((16000 * 2 * ms) / 1000));

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

const normalizeAudioChunksForInput = (
  chunks: Array<{ mimeType: string; dataBase64: string }>,
) =>
  splitBytes(
    concatBytes([
      silence16k(20),
      ...chunks.map(({ mimeType, dataBase64 }) =>
        mimeType.includes("24000")
          ? resample24kTo16k(base64ToBytes(dataBase64))
          : base64ToBytes(dataBase64)
      ),
      silence16k(1000),
    ]),
    3200,
  ).map((piece) => ({
    mimeType: "audio/pcm;rate=16000",
    dataBase64: bytesToBase64(piece),
  }));

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
  from: string,
  endpoint: DuplexEndpoint,
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
    const continuation = await session.continueTurn();
    await emitModelEvents(outputEvent, "participant", continuation);
    await Promise.all(
      sessionOutputToMessages(from, continuation).map(endpoint.sendData),
    );
    if (continuation.some((next) => next.type === "tool_call")) {
      await resolveToolCalls(
        outputEvent,
        session,
        tools,
        from,
        endpoint,
        continuation,
      );
    }
  }
};

const sessionOutputToMessages = (
  from: string,
  sessionOutput: AudioSessionEvent[],
): DuplexMessage[] => {
  const audioChunks = sessionOutput.filter((
    event,
  ): event is Extract<AudioSessionEvent, { type: "audio" }> =>
    event.type === "audio"
  ).map((event) => ({
    mimeType: event.chunk.mimeType,
    dataBase64: event.chunk.dataBase64,
  }));
  if (!empty(audioChunks)) {
    return [{ type: "audio" as const, chunks: audioChunks, from }];
  }
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
  });
  const tools = regularTools(spec.tools);
  let isClosed = false;
  await new Promise<void>((resolve) => {
    endpoint.onData(async (message) => {
      if (isClosed) return;
      console.log(`${transport.participantName} endpoint got ${message.type}`);
      if (message.type === "close") {
        isClosed = true;
        await session.close();
        resolve();
        return;
      }
      try {
        console.log(`${transport.participantName} sending to Gemini Live`);
        const sessionOutput = message.type === "text"
          ? await session.sendText(message.text)
          : await session.sendAudioChunks(
            normalizeAudioChunksForInput(message.chunks),
          );
        console.log(
          `${transport.participantName} got ${sessionOutput.length} events from Gemini Live`,
        );
        if (message.type === "audio") {
          const heard = transcriptOf(sessionOutput, "input_transcript");
          if (heard.length > 0) {
            await outputEvent(participantEditMessageTurn({
              name: transport.participantName,
              text: heard,
              onMessage: crypto.randomUUID(),
            }));
          }
        }
        await emitModelEvents(
          outputEvent,
          transport.participantName,
          sessionOutput,
        );
        await resolveToolCalls(
          outputEvent,
          session,
          tools,
          transport.participantName,
          endpoint,
          sessionOutput,
        );
        const outgoingMessages = sessionOutputToMessages(
          transport.participantName,
          sessionOutput,
        );
        if (
          outgoingMessages.length === 0 &&
          !sessionOutput.some((e) => e.type === "tool_call")
        ) {
          console.log(
            `${transport.participantName} generated no output, sending fallback to prompt it`,
          );
          const retryOutput = await session.sendText(
            "Please continue your thought or respond to the previous message if you haven't.",
          );
          console.log(
            `${transport.participantName} got ${retryOutput.length} events from fallback`,
          );
          await emitModelEvents(
            outputEvent,
            transport.participantName,
            retryOutput,
          );
          await resolveToolCalls(
            outputEvent,
            session,
            tools,
            transport.participantName,
            endpoint,
            retryOutput,
          );
          await Promise.all(
            sessionOutputToMessages(transport.participantName, retryOutput).map(
              endpoint.sendData,
            ),
          );
        } else {
          await Promise.all(outgoingMessages.map(endpoint.sendData));
        }
      } catch (error) {
        console.log(`Error in transport agent: ${error}`);
        await outputEvent(ownThoughtTurn(
          `Audio transport error for ${transport.participantName}: ${
            error instanceof Error ? error.message : String(error)
          }`,
        ));
        console.log(
          `${transport.participantName} generated no output due to error, sending fallback to prompt it`,
        );
        try {
          const retryOutput = await session.sendText(
            "There was an error processing your audio. Please continue your thought or respond to the previous message.",
          );
          await emitModelEvents(
            outputEvent,
            transport.participantName,
            retryOutput,
          );
          await resolveToolCalls(
            outputEvent,
            session,
            tools,
            transport.participantName,
            endpoint,
            retryOutput,
          );
          await Promise.all(
            sessionOutputToMessages(transport.participantName, retryOutput).map(
              endpoint.sendData,
            ),
          );
        } catch (retryError) {
          console.log(`Retry also failed: ${retryError}`);
        }
      }
    });
  });
};
