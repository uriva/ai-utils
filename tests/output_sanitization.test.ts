import { assertEquals } from "@std/assert";
import {
  appendInternalSentTimestamp,
  extractOpaqueIdentifiers,
  findNovelOpaqueIdentifiers,
  hasInternalSentTimestampSuffix,
  modelOutputHasNovelOpaqueIdentifiers,
  modelOutputLeaksInternalSentTimestamp,
  ownUtteranceTurn,
  participantUtteranceTurn,
  stripInternalSentTimestampSuffix,
} from "../mod.ts";

Deno.test("extractOpaqueIdentifiers handles urls and params", () => {
  assertEquals(
    extractOpaqueIdentifiers(
      "https://example.com/orders/ord_12345?session_id=sess_67890&slug=summer-sale",
    ).sort(),
    ["ord_12345", "sess_67890"],
  );
});

Deno.test("extractOpaqueIdentifiers ignores regular words", () => {
  assertEquals(
    extractOpaqueIdentifiers(
      "Use https://example.com/users/profile?tab=settings and mention checkout later",
    ),
    [],
  );
});

Deno.test("findNovelOpaqueIdentifiers detects ids absent from sources", () => {
  assertEquals(
    findNovelOpaqueIdentifiers(
      "Open https://example.com/orders/ord_NEW123?session_id=sess_67890",
      [
        "Prompt mentions https://example.com/orders/ord_OLD999?session_id=sess_67890",
        { result: "Known user usr_abc123" },
      ],
    ),
    ["ord_NEW123"],
  );
});

Deno.test("modelOutputHasNovelOpaqueIdentifiers flags unknown ids inside urls", () => {
  assertEquals(
    modelOutputHasNovelOpaqueIdentifiers(
      "Known session sess_67890",
      [participantUtteranceTurn({ name: "user", text: "continue" })],
      [
        ownUtteranceTurn(
          "Open https://example.com/orders/ord_NEW123?session_id=sess_67890",
        ),
      ],
    ),
    true,
  );
});

Deno.test("modelOutputHasNovelOpaqueIdentifiers flags find-scene short link ids", () => {
  assertEquals(
    modelOutputHasNovelOpaqueIdentifiers(
      "The download is still processing. No URL is available yet.",
      [participantUtteranceTurn({ name: "user", text: "wait for it" })],
      [
        ownUtteranceTurn(
          '<video controls><source src="https://api.find-scene.com/s/e53b21" type="video/mp4" /></video>',
        ),
      ],
    ),
    true,
  );
});

Deno.test("modelOutputHasNovelOpaqueIdentifiers allows legitimate known ids", () => {
  assertEquals(
    modelOutputHasNovelOpaqueIdentifiers(
      "Known order ord_12345",
      [
        participantUtteranceTurn({ name: "user", text: "continue" }),
        {
          type: "tool_result",
          isOwn: true,
          id: crypto.randomUUID(),
          timestamp: Date.now(),
          name: "fetchOrder",
          result: "Order id is ord_12345 and session sess_67890",
        },
      ],
      [
        ownUtteranceTurn(
          "Open https://example.com/orders/ord_12345?session_id=sess_67890",
        ),
      ],
    ),
    false,
  );
});

Deno.test("internal sent timestamp suffix matches the exact leaked example", () => {
  const message =
    "Here is your magic ball scene from The Prestige! 🎩✨ — sent Mar 21, 2026, 2:16 PM";
  assertEquals(hasInternalSentTimestampSuffix(message), true);
  assertEquals(
    stripInternalSentTimestampSuffix(message),
    "Here is your magic ball scene from The Prestige! 🎩✨",
  );
});

Deno.test("appendInternalSentTimestamp uses the same shape we detect", () => {
  assertEquals(
    appendInternalSentTimestamp(
      "Here is your magic ball scene from The Prestige! 🎩✨",
      Date.UTC(2026, 2, 21, 14, 16),
      "UTC",
    ),
    "Here is your magic ball scene from The Prestige! 🎩✨ — sent Mar 21, 2026, 2:16 PM",
  );
});

Deno.test("modelOutputLeaksInternalSentTimestamp flags leaked internal metadata", () => {
  assertEquals(
    modelOutputLeaksInternalSentTimestamp([
      ownUtteranceTurn(
        "Here is your magic ball scene from The Prestige! 🎩✨ — sent Mar 21, 2026, 2:16 PM",
      ),
    ]),
    true,
  );
});

Deno.test("stripInternalSentTimestampSuffix keeps the rest of the message intact", () => {
  assertEquals(
    stripInternalSentTimestampSuffix(
      "Here is your magic ball scene from The Prestige! 🎩✨\n\n\nIs this the one you were looking for? If you need any more iconic moments or even a frame as a sticker, just let me know! 🪄 — sent Mar 21, 2026, 2:16 PM",
    ),
    "Here is your magic ball scene from The Prestige! 🎩✨\n\n\nIs this the one you were looking for? If you need any more iconic moments or even a frame as a sticker, just let me know! 🪄",
  );
});
