import { assert } from "@std/assert";
import {
  estimateTokens,
  ownEditMessageTurn,
  participantUtteranceTurn,
} from "../src/agent.ts";
import { eventsToPlainText } from "../src/compaction.ts";

// A 12KB base64 blob. Before the fix this was BPE-tokenized character by
// character (thousands of "tokens") and shipped verbatim into plain-text
// projections (progress-check prompts, summarizer prompts).
const bigBase64 = "A".repeat(12_000);

const attachment = {
  kind: "inline" as const,
  mimeType: "image/jpeg",
  dataBase64: bigBase64,
};

Deno.test("token estimates do not tokenize inline base64 blobs", async () => {
  const event = participantUtteranceTurn({
    name: "user",
    text: "photo",
    attachments: [attachment],
  });
  const tokens = await estimateTokens(event);
  assert(
    tokens < 200,
    `expected a small estimate for an event whose payload is one image attachment, got ${tokens} (base64 must not be BPE-tokenized)`,
  );
});

Deno.test("plain-text projections never include inline base64 blobs", () => {
  const edit = ownEditMessageTurn({
    text: "see photo",
    onMessage: "m1",
    attachments: [attachment],
  });
  const projection = eventsToPlainText([edit]);
  assert(
    !projection.includes("AAAA"),
    "base64 leaked into the plain-text projection",
  );
});
