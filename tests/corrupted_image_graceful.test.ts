import { assert } from "@std/assert";
import { type HistoryEvent, participantUtteranceTurn } from "../src/agent.ts";
import { agentDeps, runForAllProviders } from "../test_helpers.ts";

runForAllProviders(
  "handles corrupted/unsupported file attachments gracefully on gemini",
  async (runAgentWithProvider) => {
    const ac = new AbortController();
    const server = Deno.serve({ port: 0, signal: ac.signal }, (req) => {
      const url = new URL(req.url);
      if (url.pathname === "/corrupted.webp") {
        // Corrupted bytes that look like what we found in production
        const data = new Uint8Array([
          0xb8,
          0xc8,
          0x90,
          0x48,
          0x1b,
          0x52,
          0xf0,
          0xff,
          0xfb,
          0x92,
          0xd0,
          0x44,
        ]);
        return new Response(data, {
          headers: { "content-type": "image/webp" },
        });
      }
      return new Response("Not found", { status: 404 });
    });
    const addr = server.addr as Deno.NetAddr;
    const imageUrl = `http://localhost:${addr.port}/corrupted.webp`;

    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Please describe this image.",
        attachments: [
          { kind: "file", mimeType: "image/webp", fileUri: imageUrl },
        ],
      }),
    ];

    let rewriteHistoryCalled = false;
    const mockRewriteHistory = (
      replacements: Record<string, HistoryEvent>,
    ) => {
      rewriteHistoryCalled = true;
      for (const [id, replacement] of Object.entries(replacements)) {
        const index = mockHistory.findIndex((e) => e.id === id);
        if (index !== -1) {
          mockHistory[index] = replacement;
        }
      }
      return Promise.resolve();
    };

    await agentDeps(mockHistory)(runAgentWithProvider)({
      maxIterations: 3,
      tools: [],
      prompt:
        "You are a helpful assistant. If the user sent a corrupted, missing or unsupported file, explain that gracefully and ask them to re-send.",
      lightModel: true,
      rewriteHistory: mockRewriteHistory,
      timezoneIANA: "UTC",
    });

    ac.abort();
    try {
      await server.finished;
    } catch {
      // expected on abort
    }

    // Verify rewriteHistory was called and updated the attachment with the placeholder
    assert(rewriteHistoryCalled, "rewriteHistory should have been called");

    // Verify placeholder is present in the text
    const updatedUserMsg = mockHistory.find((e) =>
      e.type === "participant_utterance"
    );
    assert(updatedUserMsg, "User message should be in history");
    assert(
      updatedUserMsg.text?.includes("corrupted or unsupported"),
      `User message text should contain the corrupted placeholder. Text: ${updatedUserMsg.text}`,
    );

    // Verify AI replied gracefully about the corrupted/unsupported file
    const aiResponse = mockHistory.find((e) => e.type === "own_utterance");
    assert(aiResponse, "AI response should be in history");
    const aiTextLower = aiResponse.text.toLowerCase();
    const hasGracefulMessage = aiTextLower.includes("corrupt") ||
      aiTextLower.includes("unsupported") ||
      aiTextLower.includes("format") ||
      aiTextLower.includes("unable") ||
      aiTextLower.includes("error") ||
      aiTextLower.includes("image") ||
      aiTextLower.includes("file");

    assert(
      hasGracefulMessage,
      `AI should have replied gracefully about the image issue. Response: ${aiResponse.text}`,
    );
  },
  3,
  true, // geminiOnly = true
  false, // sanitizeResources = false
);
