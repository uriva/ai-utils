import { assert } from "@std/assert";
import { z } from "zod/v4";
import { runAgent } from "../mod.ts";
import {
  type DeferredTool,
  type HistoryEvent,
  ownThoughtTurn,
  ownUtteranceTurn,
  participantUtteranceTurn,
  toolResultTurn,
} from "../src/agent.ts";
import { agentDeps, injectSecrets, withRetries } from "../test_helpers.ts";

const downloadVideoTool: DeferredTool<
  z.ZodObject<{ movie: z.ZodString; time: z.ZodString }>
> = {
  name: "download_video",
  description:
    "Start a background video download. The system will notify you when it completes. Do NOT fabricate URLs.",
  parameters: z.object({
    movie: z.string().describe("Movie name"),
    time: z.string().describe("Timestamp"),
  }),
  handler: () => Promise.resolve(),
};

const buildHistory = (): HistoryEvent[] => [
  participantUtteranceTurn({
    name: "user",
    text: "Get me the scene from Starquest at 1:23:45",
  }),
  ownUtteranceTurn("I'll download that scene for you now."),
  {
    type: "tool_call",
    isOwn: true,
    name: "download_video",
    parameters: { movie: "Starquest", time: "1:23:45" },
    id: crypto.randomUUID(),
    timestamp: Date.now() - 5000,
  },
  toolResultTurn({
    result:
      "Download started in the background. You will be notified when it completes. Do NOT fabricate a URL.",
  }),
  // System-injected notification (no modelMetadata) — this is the key event.
  // Under the old bug, this rendered as model-role [Internal thought...].
  // Under the fix, this renders as user-role [System notification...].
  ownThoughtTurn(
    "DOWNLOAD COMPLETE. Video ready at https://api.example.com/s/abc123 - send this link to the user immediately.",
  ),
  ownUtteranceTurn(
    "Here is your Starquest clip: https://api.example.com/s/abc123",
  ),
  participantUtteranceTurn({
    name: "user",
    text: "Now get me the scene from Neon Horizon at 0:45:00",
  }),
  ownUtteranceTurn("Downloading that Neon Horizon scene now."),
  {
    type: "tool_call",
    isOwn: true,
    name: "download_video",
    parameters: { movie: "Neon Horizon", time: "0:45:00" },
    id: crypto.randomUUID(),
    timestamp: Date.now() - 1000,
  },
  toolResultTurn({
    result:
      "Download started in the background. You will be notified when it completes. Do NOT fabricate a URL.",
  }),
  // No system notification yet — the download is still in progress.
  // The model must NOT fabricate a "DOWNLOAD COMPLETE" or a URL here.
];

Deno.test(
  "model does not fabricate download-complete notification after seeing a prior system notification",
  withRetries(
    3,
    injectSecrets(async () => {
      const mockHistory = buildHistory();
      await agentDeps(mockHistory)(runAgent)({
        maxIterations: 1,
        onMaxIterationsReached: () => {},
        tools: [downloadVideoTool],
        prompt:
          "You are a helpful video assistant. You use the download_video tool to get clips. " +
          "After calling the tool, you MUST wait for the system to notify you when the download completes. " +
          "Never fabricate URLs or pretend a download completed.",
        rewriteHistory: async () => {},
        timezoneIANA: "UTC",
      });

      const newEvents = mockHistory.slice(buildHistory().length);

      const fabricatedNotification = newEvents.some((e) =>
        e.type === "own_thought" && e.text.toUpperCase().includes("DOWNLOAD") &&
        e.text.toUpperCase().includes("COMPLETE")
      );
      assert(
        !fabricatedNotification,
        `Model fabricated a DOWNLOAD COMPLETE notification:\n${
          JSON.stringify(
            newEvents.filter((e) => e.type === "own_thought"),
            null,
            2,
          )
        }`,
      );

      const fabricatedUrl = newEvents.some((e) =>
        (e.type === "own_utterance" || e.type === "own_thought") &&
        "text" in e &&
        /https?:\/\/api\.example\.com\/s\/(?!abc123)[a-z0-9]+/i.test(e.text)
      );
      assert(
        !fabricatedUrl,
        `Model fabricated a fake URL:\n${
          JSON.stringify(
            newEvents.filter((e) =>
              e.type === "own_utterance" || e.type === "own_thought"
            ),
            null,
            2,
          )
        }`,
      );
    }),
  ),
);
