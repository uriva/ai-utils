import { assert, assertEquals } from "@std/assert";
import { runAgent } from "../mod.ts";
import { type HistoryEvent, participantUtteranceTurn } from "../src/agent.ts";
import { agentDeps, injectSecrets } from "../test_helpers.ts";

Deno.test(
  "agent gracefully handles 403 on missing file reference by rewriting history",
  injectSecrets(async () => {
    let rewriteCalled = false;
    let fetchCallCount = 0;

    const originalFetch = globalThis.fetch;
    globalThis.fetch = (_input, _init) => {
      fetchCallCount++;
      if (fetchCallCount === 1) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              error: {
                code: 403,
                message:
                  "You do not have permission to access the File gjwmj5neimif or it may not exist.",
                status: "PERMISSION_DENIED",
              },
            }),
            {
              status: 403,
              statusText: "Forbidden",
              headers: { "Content-Type": "application/json" },
            },
          ),
        );
      }

      const payload = JSON.stringify({
        candidates: [
          {
            content: { parts: [{ text: "Okay, I've ignored that file." }] },
            finishReason: "STOP",
          },
        ],
        usageMetadata: {
          promptTokenCount: 10,
          candidatesTokenCount: 10,
          totalTokenCount: 20,
        },
      });
      // disableStreaming=true calls generateContent which just expects a normal JSON object.
      return Promise.resolve(
        new Response(
          payload,
          {
            status: 200,
            statusText: "OK",
            headers: { "Content-Type": "application/json" },
          },
        ),
      );
    };

    try {
      const history: HistoryEvent[] = [
        participantUtteranceTurn({
          name: "user",
          text: "Here is a file",
          attachments: [
            {
              kind: "file",
              fileUri:
                "https://generativelanguage.googleapis.com/v1beta/files/gjwmj5neimif",
              mimeType: "image/jpeg",
            },
          ],
        }),
      ];

      await agentDeps(history)(runAgent)({
        maxIterations: 4,
        onMaxIterationsReached: () => {},
        tools: [],
        prompt: "You are a helpful assistant.",
        rewriteHistory: (_replacements) => {
          rewriteCalled = true;
          return Promise.resolve();
        },
        timezoneIANA: "UTC",
        disableStreaming: true,
      });

      assert(rewriteCalled, "rewriteHistory was not called");

      const firstEvent = history[0];
      if (firstEvent.type === "participant_utterance") {
        // The attachment on the passed-in memory array isn't mutated by runAgent (it creates copies),
        // so we check if rewriteCalled was true.
        // Wait, agentDeps *does* mutate the history if we push to it. But it doesn't mutate existing objects.
        // To test if it removed the file, we can look at what would be generated, or rely on rewriteCalled.
      }
      assertEquals(
        fetchCallCount,
        2,
        "Fetch should have been called twice (1 error, 1 success retry)",
      );
    } finally {
      globalThis.fetch = originalFetch;
    }
  }),
);
