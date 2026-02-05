#!/usr/bin/env -S deno run --allow-env --allow-run --allow-read --allow-write --allow-net

import {
  createLiveSession,
  handleMessages,
  sendAudioChunk,
} from "./src/geminiVoiceStreaming.ts";

const SAMPLE_RATE = 16000;

let audioProcess: Deno.ChildProcess | null = null;
let playbackProcess: Deno.ChildProcess | null = null;
let playbackWriter: WritableStreamDefaultWriter<Uint8Array> | null = null;
let isTransmitting = false;

const captureAudioStream = async function* (): AsyncGenerator<string> {
  const command = new Deno.Command("ffmpeg", {
    args: [
      "-f",
      "pulse",
      "-i",
      "default",
      "-af",
      "volume=0.3",
      "-ar",
      SAMPLE_RATE.toString(),
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
  });

  audioProcess = command.spawn();

  if (!audioProcess.stdout) {
    throw new Error("Failed to capture audio stream");
  }

  const reader = audioProcess.stdout.getReader();

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const base64 = btoa(String.fromCharCode(...value));
      yield base64;
    }
  } finally {
    reader.releaseLock();
  }
};

const playAudioChunk = async (audioBase64: string): Promise<void> => {
  const audioData = Uint8Array.from(atob(audioBase64), (c) => c.charCodeAt(0));
  Deno.stdout.writeSync(new TextEncoder().encode(`\n🔊 Playing ${audioData.length} bytes\n`));

  if (!playbackProcess || !playbackWriter) {
    Deno.stdout.writeSync(new TextEncoder().encode("🎵 Starting aplay process...\n"));
    const command = new Deno.Command("aplay", {
      args: [
        "-f",
        "S16_LE",
        "-r",
        "24000",
        "-c",
        "1",
        "-t",
        "raw",
      ],
      stdin: "piped",
      stdout: "null",
      stderr: "piped",
    });

    playbackProcess = command.spawn();
    
    if (playbackProcess.stdin) {
      playbackWriter = playbackProcess.stdin.getWriter();
    }
    
    if (playbackProcess.stderr) {
      (async () => {
        const reader = playbackProcess!.stderr!.getReader();
        const decoder = new TextDecoder();
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            const text = decoder.decode(value);
            if (text.trim()) {
              Deno.stdout.writeSync(new TextEncoder().encode(`🎵 aplay: ${text.trim()}\n`));
            }
          }
        } finally {
          reader.releaseLock();
        }
      })();
    }
  }

  if (playbackWriter) {
    try {
      await playbackWriter.write(audioData);
      Deno.stdout.writeSync(new TextEncoder().encode("✅ Audio chunk written to aplay\n"));
    } catch (error) {
      if (error instanceof Deno.errors.BrokenPipe) {
        Deno.stdout.writeSync(new TextEncoder().encode("🔄 Restarting playback process...\n"));
        playbackWriter = null;
        playbackProcess = null;
        await playAudioChunk(audioBase64);
      } else {
        throw error;
      }
    }
  }
};

const main = async () => {
  const apiKey = Deno.env.get("GEMINI_API_KEY");
  if (!apiKey) {
    console.error("❌ Error: GEMINI_API_KEY environment variable not set");
    Deno.exit(1);
  }

  try {
    const checkFfmpeg = new Deno.Command("ffmpeg", {
      args: ["-version"],
      stdout: "null",
      stderr: "null",
    });
    await checkFfmpeg.output();
  } catch {
    console.error("❌ Error: ffmpeg not found. Install: sudo apt-get install ffmpeg");
    Deno.exit(1);
  }

  try {
    const checkAplay = new Deno.Command("aplay", {
      args: ["--version"],
      stdout: "null",
      stderr: "null",
    });
    await checkAplay.output();
  } catch {
    console.error("❌ Error: aplay not found. Install: sudo apt-get install alsa-utils");
    Deno.exit(1);
  }

  console.log("🎙️  Connecting to Gemini Live...");

  const session = await createLiveSession({ apiKey });

  console.log("✅ Session created, setting up message handler...");

  handleMessages(
    session,
    async (audioBase64) => {
      console.log("📥 Got audio response, playing...");
      await playAudioChunk(audioBase64);
    },
    (text) => {
      console.log(`🤖: ${text}`);
    },
    () => {
      console.log("✅ Turn complete");
    },
    (error) => {
      console.error("❌ Error:", error);
    },
  );

  console.log("✅ Message handler set up");
  await new Promise(resolve => setTimeout(resolve, 500));

  console.log("🎤 Voice chat started.");
  console.log("Press SPACE to toggle transmit on/off.");
  console.log("Press Ctrl+C to exit\n");

  Deno.stdin.setRaw(true);
  
  const keyListener = async () => {
    const buf = new Uint8Array(1);
    while (true) {
      try {
        const n = await Deno.stdin.read(buf);
        if (n === null) break;
        
        if (buf[0] === 32) {
          isTransmitting = !isTransmitting;
          const status = isTransmitting ? "🎙️  TRANSMITTING..." : "🔇 MUTED";
          Deno.stdout.writeSync(new TextEncoder().encode(`\n${status}\n`));
        } else if (buf[0] === 3) {
          Deno.exit(0);
        }
      } catch (e) {
        Deno.stdout.writeSync(new TextEncoder().encode(`\nKey listener error: ${e}\n`));
        break;
      }
    }
  };
  
  keyListener();

  let chunkCount = 0;
  try {
    for await (const audioChunk of captureAudioStream()) {
      if (isTransmitting) {
        chunkCount++;
        if (chunkCount % 5 === 0) {
          Deno.stdout.writeSync(new TextEncoder().encode(`📤 Sent ${chunkCount} chunks\n`));
        }
        sendAudioChunk(session, audioChunk);
      } else {
        if (chunkCount > 0) {
          Deno.stdout.writeSync(new TextEncoder().encode(`📤 Sent ${chunkCount} chunks total\n`));
          chunkCount = 0;
        }
      }
    }
  } catch (error) {
    Deno.stdout.writeSync(new TextEncoder().encode(`Error during audio capture: ${error}\n`));
  }
};

globalThis.addEventListener("unload", () => {
  if (audioProcess) {
    audioProcess.kill("SIGKILL");
  }
  if (playbackProcess) {
    playbackProcess.kill("SIGKILL");
  }
});

if (import.meta.main) {
  main();
}
