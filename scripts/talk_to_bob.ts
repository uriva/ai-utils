#!/usr/bin/env -S deno run --allow-env --allow-run --allow-read --allow-write --allow-net

import {
  type HistoryEvent,
  injectAccessHistory,
  injectOutputEvent,
  participantUtteranceTurn,
  runAgent,
} from "../mod.ts";

const sampleRate = 16000;
const bytesPer100ms = sampleRate * 2 / 10;

let audioProcess: Deno.ChildProcess | null = null;
let playbackProcess: Deno.ChildProcess | null = null;
let playbackWriter: WritableStreamDefaultWriter<Uint8Array> | null = null;
let isTransmitting = false;
let isResponding = false;
const bufferedChunks: Array<{ mimeType: string; dataBase64: string }> = [];
let pendingMicBytes = new Uint8Array(0);
const history: HistoryEvent[] = [];

const bytesToBase64 = (bytes: Uint8Array) => {
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
};

const silence16k = (ms: number) =>
  new Uint8Array(Math.floor((sampleRate * 2 * ms) / 1000));

const latestOwnUtterance = (events: HistoryEvent[]) =>
  [...events].reverse().find((
    event,
  ): event is Extract<HistoryEvent, { type: "own_utterance" }> =>
    event.type === "own_utterance"
  );

const transcriptOf = (
  events: HistoryEvent[],
  type: "participant_utterance" | "participant_edit_message" | "own_utterance",
) =>
  events.filter((
    event,
  ): event is Extract<HistoryEvent, { type: typeof type }> =>
    event.type === type
  ).map((event) => event.text).join(" ").trim();

const playAudioChunk = async (dataBase64: string) => {
  const audioData = Uint8Array.from(
    atob(dataBase64),
    (char) => char.charCodeAt(0),
  );
  if (!playbackProcess || !playbackWriter) {
    playbackProcess = new Deno.Command("aplay", {
      args: ["-f", "S16_LE", "-r", "24000", "-c", "1", "-t", "raw"],
      stdin: "piped",
      stdout: "null",
      stderr: "null",
    }).spawn();
    if (!playbackProcess.stdin) {
      throw new Error("failed to open playback stdin");
    }
    playbackWriter = playbackProcess.stdin.getWriter();
  }
  await playbackWriter.write(audioData);
};

const captureAudioStream = async function* (): AsyncGenerator<string> {
  audioProcess = new Deno.Command("ffmpeg", {
    args: [
      "-f",
      "pulse",
      "-i",
      "default",
      "-af",
      "volume=0.5",
      "-ar",
      String(sampleRate),
      "-ac",
      "1",
      "-acodec",
      "pcm_s16le",
      "-f",
      "s16le",
      "-",
    ],
    stdout: "piped",
    stderr: "null",
  }).spawn();
  if (!audioProcess.stdout) {
    throw new Error("failed to capture audio stream");
  }
  const reader = audioProcess.stdout.getReader();
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const combined = new Uint8Array(pendingMicBytes.length + value.length);
      combined.set(pendingMicBytes, 0);
      combined.set(value, pendingMicBytes.length);
      let offset = 0;
      while (combined.length - offset >= bytesPer100ms) {
        const chunk = combined.slice(offset, offset + bytesPer100ms);
        yield btoa(String.fromCharCode(...chunk));
        offset += bytesPer100ms;
      }
      pendingMicBytes = combined.slice(offset);
    }
  } finally {
    reader.releaseLock();
  }
};

const ensureCommand = async (command: string, hint: string) => {
  const result = await new Deno.Command(command, {
    args: ["--version"],
    stdout: "null",
    stderr: "null",
  }).output().catch(() => null);
  if (result) return;
  throw new Error(hint);
};

const flushTurn = async () => {
  if (isResponding || isTransmitting || bufferedChunks.length === 0) return;
  isResponding = true;
  const chunks = bufferedChunks.splice(0);
  const paddedChunks = [
    {
      mimeType: "audio/pcm;rate=16000",
      dataBase64: bytesToBase64(silence16k(120)),
    },
    ...chunks,
    {
      mimeType: "audio/pcm;rate=16000",
      dataBase64: bytesToBase64(silence16k(220)),
    },
  ];
  console.log(`\nSending to Bob... (${chunks.length} chunks)`);
  let progress: number | undefined;
  try {
    progress = setInterval(() => {
      console.log("...waiting for Bob");
    }, 2000);
    const before = history.length;
    history.push(participantUtteranceTurn({
      name: "Uri",
      text: "",
      attachments: paddedChunks.map((chunk) => ({
        kind: "inline" as const,
        mimeType: chunk.mimeType,
        dataBase64: chunk.dataBase64,
      })),
    }));
    await runAgent({
      prompt:
        "You are Bob. You are talking live with Uri over voice. Sound warm, relaxed, and concise. Never speak your reasoning, preparation, or narration out loud. Only say the reply itself. Speak naturally like a person in a live call.",
      tools: [],
      maxIterations: 2,
      onMaxIterationsReached: () => {},
      rewriteHistory: async () => {},
      timezoneIANA: "UTC",
      transport: {
        kind: "audio",
        voiceName: "Orus",
        participantName: "Uri",
      },
    });
    const events = history.slice(before);
    const heard = transcriptOf(events, "participant_edit_message") ||
      transcriptOf(events, "participant_utterance");
    const reply = transcriptOf(events, "own_utterance");
    if (heard) console.log(`You: ${heard}`);
    if (reply) console.log(`Bob: ${reply}`);
    const ownUtterance = latestOwnUtterance(events);
    for (const attachment of ownUtterance?.attachments ?? []) {
      if (attachment.kind !== "inline") continue;
      await playAudioChunk(attachment.dataBase64);
    }
  } catch (error) {
    console.error(
      `Bob did not answer: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
    console.log(
      "Try a shorter utterance and wait a beat before pressing SPACE again.",
    );
  } finally {
    if (progress !== undefined) clearInterval(progress);
    isResponding = false;
  }
};

const main = async () => {
  const apiKey = Deno.env.get("GEMINI_API_KEY");
  if (!apiKey) {
    throw new Error("GEMINI_API_KEY environment variable not set");
  }
  await ensureCommand("ffmpeg", "ffmpeg not found. Install ffmpeg.");
  await ensureCommand("aplay", "aplay not found. Install alsa-utils.");
  await injectAccessHistory(() => Promise.resolve(history))(
    injectOutputEvent((event) => {
      history.push(event);
      return Promise.resolve();
    })(() => Promise.resolve()),
  );

  console.log("Connected to Bob.");
  console.log("Press SPACE to start/stop talking. Press Ctrl+C to exit.\n");
  Deno.stdin.setRaw(true);

  const keyListener = async () => {
    const buf = new Uint8Array(1);
    while (true) {
      const n = await Deno.stdin.read(buf);
      if (n === null) break;
      if (buf[0] === 32) {
        isTransmitting = !isTransmitting;
        console.log(isTransmitting ? "Listening..." : "Stopped listening.");
        if (!isTransmitting) {
          await flushTurn();
        }
        continue;
      }
      if (buf[0] === 3) {
        Deno.exit(0);
      }
    }
  };

  keyListener();

  for await (const dataBase64 of captureAudioStream()) {
    if (isTransmitting) {
      bufferedChunks.push({ mimeType: "audio/pcm;rate=16000", dataBase64 });
    }
  }
};

globalThis.addEventListener("unload", () => {
  audioProcess?.kill("SIGKILL");
  playbackProcess?.kill("SIGKILL");
});

if (import.meta.main) {
  await main();
}
