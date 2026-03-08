import { assert, assertEquals } from "@std/assert";
import { z } from "zod/v4";
import { createDuplexPair, type HistoryEvent, runAgent, tool } from "../mod.ts";
import {
  createAudioArtifactsWriter,
  createTransportAudioRecorder,
} from "../src/audioArtifacts.ts";
import { injectSecrets } from "../test_helpers.ts";

Deno.test({
  name: "two Gemini Live bots exchange and store a code over audio",
  ignore: !Deno.env.get("GEMINI_API_KEY"),
  fn: injectSecrets(async () => {
    const fetchedCode = "ALPHA TANGO";
    const storedCodes: string[] = [];

    const fetchCode = tool({
      name: "fetchCode",
      description: "Fetch the relay code Alice should pass on",
      parameters: z.object({}),
      handler: () => Promise.resolve(fetchedCode),
    });

    const storeCode = tool({
      name: "storeCode",
      description: "Store the relay code Bob heard",
      parameters: z.object({ code: z.string() }),
      handler: ({ code }) => {
        storedCodes.push(code);
        return Promise.resolve("stored");
      },
    });

    const hangups = new Set<string>();
    const delay = (ms: number) =>
      new Promise((resolve) => setTimeout(resolve, ms));
    const aliceHangUp = tool({
      name: "hangUp",
      description: "Hang up the live voice conversation when done",
      parameters: z.object({}),
      handler: async () => {
        await delay(100);
        hangups.add("alice");
        return Promise.resolve("hung up");
      },
    });
    const bobHangUp = tool({
      name: "hangUp",
      description: "Hang up the live voice conversation when done",
      parameters: z.object({}),
      handler: async () => {
        await delay(100);
        hangups.add("bob");
        return Promise.resolve("hung up");
      },
    });

    const { left: rawAliceEndpoint, right: rawBobEndpoint } =
      createDuplexPair();
    const runDir = new URL("../artifacts/latest/", import.meta.url);
    await Deno.mkdir(runDir, { recursive: true });
    console.log(`runArtifacts ${runDir.pathname}`);
    const writeArtifact = createAudioArtifactsWriter(runDir);
    const recordTransportAudio = createTransportAudioRecorder(runDir);
    const aliceEndpoint = {
      sendData: async (
        message: Parameters<typeof rawAliceEndpoint.sendData>[0],
      ) => {
        await recordTransportAudio(message);
        await rawAliceEndpoint.sendData(message);
      },
      onData: rawAliceEndpoint.onData,
    };
    const bobEndpoint = {
      sendData: async (
        message: Parameters<typeof rawBobEndpoint.sendData>[0],
      ) => {
        await recordTransportAudio(message);
        await rawBobEndpoint.sendData(message);
      },
      onData: rawBobEndpoint.onData,
    };

    const aliceSpec = {
      prompt:
        "You are Alice. You are in a voice conversation with Bob. Keep answers to less than 10 words. Never speak your reasoning. When Bob asks for the relay code, physically execute the fetchCode tool FIRST. Then say the fetched code ONCE over voice. After Bob confirms he stored it, say bye and physically execute the hangUp tool.",
      tools: [fetchCode, aliceHangUp],
      maxIterations: 2,
      onMaxIterationsReached: () => {},
      timezoneIANA: "UTC",
      transport: {
        kind: "audio" as const,
        endpoint: aliceEndpoint,
        voiceName: "Zephyr",
        participantName: "Bob",
      },
    };
    const bobSpec = {
      prompt:
        "You are Bob. You are in a voice conversation with Alice. Keep answers to less than 10 words. Never speak your reasoning. After greeting Alice, ask for the relay code. When Alice says a code, physically execute the storeCode tool with the exact words. Then say bye and physically execute the hangUp tool.",
      tools: [storeCode, bobHangUp],
      maxIterations: 2,
      onMaxIterationsReached: () => {},
      timezoneIANA: "UTC",
      transport: {
        kind: "audio" as const,
        endpoint: bobEndpoint,
        voiceName: "Orus",
        participantName: "Alice",
      },
    };

    const deadline = Date.now() + 180_000;
    const aliceHistory: HistoryEvent[] = [];
    const bobHistory: HistoryEvent[] = [];

    const captureOutput =
      (history: HistoryEvent[]) => (event: HistoryEvent) => {
        history.push(event);
        return writeArtifact(event);
      };

    const aliceRunner = () =>
      runAgent({
        ...aliceSpec,
        onOutputEvent: captureOutput(aliceHistory),
        rewriteHistory: async () => {},
      });
    const bobRunner = () =>
      runAgent({
        ...bobSpec,
        onOutputEvent: captureOutput(bobHistory),
        rewriteHistory: async () => {},
      });

    let deadlineTimer: number | undefined;
    let pollInterval: number | undefined;

    const timer = new Promise<void>((resolve) => {
      deadlineTimer = setTimeout(resolve, 180_000);
    });

    const aliceTask = aliceRunner();
    const bobTask = bobRunner();

    await bobEndpoint.sendData({
      type: "text",
      text: "Start the conversation with Bob with a short greeting.",
      from: "tester",
    });

    await Promise.race([
      timer,
      new Promise<void>((resolve) => {
        pollInterval = setInterval(() => {
          if (hangups.size === 2) {
            console.log("Both bots hung up! Resolving Promise.race!");
            resolve();
          }
        }, 100);
      }),
    ]);

    if (deadlineTimer) clearTimeout(deadlineTimer);
    if (pollInterval) clearInterval(pollInterval);
    await Promise.all([
      aliceEndpoint.sendData({ type: "close", from: "test" }),
      bobEndpoint.sendData({ type: "close", from: "test" }),
    ]);
    await Promise.all([aliceTask, bobTask]);

    assertEquals(storedCodes.at(-1)?.toUpperCase(), fetchedCode);
    assert(hangups.size === 2 || Date.now() >= deadline);
  }),
});
