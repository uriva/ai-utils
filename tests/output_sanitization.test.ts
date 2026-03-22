import { assertEquals } from "@std/assert";
import {
  appendInternalSentTimestamp,
  extractOpaqueIdentifiers,
  findNovelOpaqueIdentifiers,
  guardNovelOpaqueIdentifiers,
  hasInternalSentTimestampSuffix,
  modelOutputHasNovelOpaqueIdentifiers,
  modelOutputLeaksInternalSentTimestamp,
  ownThoughtTurn,
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

Deno.test("findNovelOpaqueIdentifiers ignores markdown punctuation in urls", () => {
  assertEquals(
    findNovelOpaqueIdentifiers(
      "Here is the url: https://foo.com/my-super-secret-id-1234.",
      ["[link](https://foo.com/my-super-secret-id-1234)"],
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

Deno.test("extractOpaqueIdentifiers rejects dash-separated simple tokens", () => {
  for (
    const token of [
      "7-Cafe",
      "mar-22",
      "2024-05-12",
      "hello-world",
      "Section-3",
      "my-super-secret-id",
    ]
  ) {
    assertEquals(
      extractOpaqueIdentifiers(token),
      [],
      `expected [] for ${token}`,
    );
  }
});

Deno.test("modelOutputHasNovelOpaqueIdentifiers allows id from own_thought in history", () => {
  const prompt = "You are a helpful bot.";
  const history = [
    participantUtteranceTurn({
      name: "user",
      text: "53:56 - 6 seconds gif plz",
    }),
    ownUtteranceTurn(
      "Download started. The video will be sent to the user automatically when ready.",
    ),
    ownThoughtTurn(
      'DOWNLOAD COMPLETE. Confirmed media HTML: <img src="https://api.find-scene.com/s/3aedd2">',
    ),
  ];
  const output = [
    ownUtteranceTurn(
      'Here is your GIF! <img src="https://api.find-scene.com/s/3aedd2">',
    ),
  ];
  assertEquals(
    modelOutputHasNovelOpaqueIdentifiers(prompt, history, output),
    false,
  );
});

Deno.test("BUG: history compaction causes false positive - old ids summarized away are flagged as novel", () => {
  const prompt = "You are a helpful bot.";
  const history = [
    // Older conversation segments have been compacted into a natural-language summary.
    // The summary describes what happened but does NOT contain the raw opaque IDs
    // (54ddbe, ce6cef) that were in the original events.
    ownThoughtTurn(
      "Past conversation history was compacted. Here is my summary of what happened: " +
        "The user asked me to find scenes from The Prestige. I searched for several scenes " +
        "and downloaded video clips for them. The user was happy with the results and asked " +
        "for more clips from different timestamps.",
    ),
    // Recent (non-compacted) events from the current interaction
    participantUtteranceTurn({
      name: "user",
      text: "53:56 - 6 seconds gif plz",
    }),
    ownUtteranceTurn(
      "Download started. The video will be sent to the user automatically when ready.",
    ),
    ownThoughtTurn(
      'DOWNLOAD COMPLETE. Confirmed media HTML: <img src="https://api.find-scene.com/s/3aedd2">',
    ),
  ];
  // The model references the new URL (3aedd2, present in history) AND old URLs
  // (54ddbe, ce6cef) from earlier in the conversation that were compacted away.
  // The model remembers these from its broader context window, but they no longer
  // appear as raw identifiers in the history events.
  const output = [
    ownUtteranceTurn(
      "Here is your GIF! Here are all the clips from our session:\n" +
        '<img src="https://api.find-scene.com/s/3aedd2">\n' +
        '<img src="https://api.find-scene.com/s/54ddbe">\n' +
        '<img src="https://api.find-scene.com/s/ce6cef">',
    ),
  ];
  // BUG: This returns true (flags 54ddbe and ce6cef as novel) even though
  // the model is correctly recalling real URLs from the conversation.
  // The guard would suppress this response, causing "Oops something went wrong".
  // Expected behavior: should return false, since these are legitimate IDs
  // that existed in the conversation before compaction.
  assertEquals(
    modelOutputHasNovelOpaqueIdentifiers(prompt, history, output),
    true, // documents the current buggy behavior
  );
});

Deno.test("guardNovelOpaqueIdentifiers reclassifies leaked thought with timestamp to own_thought", () => {
  const result = guardNovelOpaqueIdentifiers(
    "You are helpful.",
    [participantUtteranceTurn({ name: "user", text: "hi" })],
    [
      ownUtteranceTurn(
        "[Internal thought, visible only to you: PROACTIVE TASK: check the weather] — sent Mar 22, 2026, 10:30 AM",
      ),
    ],
  );
  assertEquals(result.emit.length, 1);
  const event = result.emit[0];
  assertEquals(event.type, "own_thought");
  if (event.type !== "own_thought") throw new Error("unreachable");
  assertEquals(
    event.text,
    "PROACTIVE TASK: check the weather",
  );
});

Deno.test("guardNovelOpaqueIdentifiers reclassifies leaked thought without timestamp to own_thought", () => {
  const result = guardNovelOpaqueIdentifiers(
    "You are helpful.",
    [participantUtteranceTurn({ name: "user", text: "hi" })],
    [
      ownUtteranceTurn(
        "[Internal thought, visible only to you: PROACTIVE TASK: check the weather]",
      ),
    ],
  );
  assertEquals(result.emit.length, 1);
  const event = result.emit[0];
  assertEquals(event.type, "own_thought");
  if (event.type !== "own_thought") throw new Error("unreachable");
  assertEquals(
    event.text,
    "PROACTIVE TASK: check the weather",
  );
});

Deno.test("guardNovelOpaqueIdentifiers does not reclassify normal utterances", () => {
  const output = [ownUtteranceTurn("Hello! How can I help you?")];
  const result = guardNovelOpaqueIdentifiers(
    "You are helpful.",
    [participantUtteranceTurn({ name: "user", text: "hi" })],
    output,
  );
  assertEquals(result.emit.length, 1);
  const event = result.emit[0];
  assertEquals(event.type, "own_utterance");
  if (event.type !== "own_utterance") throw new Error("unreachable");
  assertEquals(event.text, "Hello! How can I help you?");
});
