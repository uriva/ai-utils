import { assert } from "@std/assert";
import { type HistoryEvent, participantUtteranceTurn } from "../src/agent.ts";
import { injectRawUploadBlobToGemini } from "../src/gemini.ts";
import { geminiUploadJsonParseErrorMessage } from "../src/utils.ts";
import { agentDeps, runForAllProviders } from "../test_helpers.ts";

const alwaysFailingUpload = () =>
  Promise.reject(new SyntaxError(geminiUploadJsonParseErrorMessage));

runForAllProviders(
  "a persistently un-uploadable attachment does not brick the conversation",
  async (runAgentWithProvider) => {
    const ac = new AbortController();
    const server = Deno.serve({ port: 0, signal: ac.signal }, (req) => {
      const url = new URL(req.url);
      if (url.pathname === "/menu.pdf") {
        return new Response(new Uint8Array([1, 2, 3, 4]), {
          headers: { "content-type": "application/pdf" },
        });
      }
      return new Response("Not found", { status: 404 });
    });
    const addr = server.addr as Deno.NetAddr;
    const fileUrl = `http://localhost:${addr.port}/menu.pdf`;

    const history: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Here is my menu, can you read it?",
        attachments: [
          { kind: "file", mimeType: "application/pdf", fileUri: fileUrl },
        ],
      }),
    ];

    const rewriteHistory = (replacements: Record<string, HistoryEvent>) => {
      for (const [id, replacement] of Object.entries(replacements)) {
        const index = history.findIndex((e) => e.id === id);
        if (index !== -1) history[index] = replacement;
      }
      return Promise.resolve();
    };

    const spec = {
      maxIterations: 3,
      tools: [],
      prompt:
        "You are a helpful assistant. If a file could not be processed, say so and ask the user to resend.",
      lightModel: true,
      rewriteHistory,
      timezoneIANA: "UTC",
    };

    await injectRawUploadBlobToGemini(() => alwaysFailingUpload)(async () => {
      // Turn 1: the file turn. The upload will fail persistently.
      await agentDeps(history)(runAgentWithProvider)(spec);
      // Turn 2: a text-only follow-up. This must NOT be bricked by the
      // un-uploadable file still sitting in history.
      history.push(
        participantUtteranceTurn({
          name: "user",
          text: "Any update on my menu?",
        }),
      );
      await agentDeps(history)(runAgentWithProvider)(spec);
    })();

    ac.abort();
    try {
      await server.finished;
    } catch {
      // expected on abort
    }

    const replies = history.filter((e) => e.type === "own_utterance");
    assert(
      replies.length >= 2,
      `the agent must reply on both turns despite the un-uploadable attachment; got ${replies.length} replies`,
    );
  },
  3,
  true, // geminiOnly: the upload path is Gemini-specific
  false, // sanitizeResources
);
