import { assertEquals } from "@std/assert";
import { z } from "zod/v4";
import { createDuplexPair, runAgent, tool } from "../mod.ts";
import {
  createAudioArtifactsWriter,
  createTransportAudioRecorder,
} from "../src/audioArtifacts.ts";
import { injectSecrets, withRetries } from "../test_helpers.ts";

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

Deno.test({
  name: "single audio agent can fetch and speak a relay code",
  ignore: !Deno.env.get("GEMINI_API_KEY"),
  sanitizeOps: false,
  sanitizeResources: false,
  fn: withRetries(
    3,
    injectSecrets(async () => {
      const fetchedCode = "ALPHA TANGO";
      const { left: testEndpoint, right: agentEndpoint } = createDuplexPair();
      const runDir = new URL("../artifacts/latest/", import.meta.url);
      await Deno.mkdir(runDir, { recursive: true });
      console.log(`runArtifacts ${runDir.pathname}`);
      const writeArtifact = createAudioArtifactsWriter(runDir);
      const recordTransportAudio = createTransportAudioRecorder(runDir);
      const outputTexts: string[] = [];

      const fetchCode = tool({
        name: "fetchCode",
        description: "Fetch the relay code the assistant should say out loud",
        parameters: z.object({}),
        handler: () => Promise.resolve(fetchedCode),
      });

      const hangUp = tool({
        name: "hangUp",
        description: "Hang up the live voice conversation when done",
        parameters: z.object({}),
        handler: async () => {
          await delay(100);
          return Promise.resolve("hung up");
        },
      });

      const wrappedEndpoint = {
        sendData: async (
          message: Parameters<typeof agentEndpoint.sendData>[0],
        ) => {
          await recordTransportAudio(message);
          await agentEndpoint.sendData(message);
        },
        onData: agentEndpoint.onData,
      };

      const agentTask = runAgent({
        prompt:
          "You are a voice assistant. Keep answers under 10 words. Never speak your reasoning. When asked for the relay code, physically execute the fetchCode tool FIRST. Then say the fetched code ONCE over voice and physically execute the hangUp tool.",
        tools: [fetchCode, hangUp],
        maxIterations: 2,
        onMaxIterationsReached: () => {},
        timezoneIANA: "UTC",
        transport: {
          kind: "audio" as const,
          endpoint: wrappedEndpoint,
          voiceName: "Zephyr",
          participantName: "User",
        },
        onOutputEvent: (event) => {
          if (event.type === "own_utterance") outputTexts.push(event.text);
          return writeArtifact(event);
        },
        rewriteHistory: async () => {},
      });

      await testEndpoint.sendData({
        type: "text",
        text: "Tell me the relay code out loud now.",
        from: "tester",
      });

      await delay(20_000);
      await testEndpoint.sendData({ type: "close", from: "tester" });
      await agentTask;

      assertEquals(
        outputTexts.some((text) => text.toUpperCase().includes(fetchedCode)),
        true,
        JSON.stringify(outputTexts, null, 2),
      );
    }),
  ),
});
