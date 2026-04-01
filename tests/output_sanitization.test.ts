import { assertEquals } from "@std/assert";
import {
  appendInternalSentTimestamp,
  hasInternalSentTimestampSuffix,
  modelOutputLeaksInternalSentTimestamp,
  ownUtteranceTurn,
  participantUtteranceTurn,
  sanitizeModelOutput,
  stripFabricatedUserMessages,
  stripInternalSentTimestampSuffix,
} from "../mod.ts";

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

Deno.test("sanitizeModelOutput reclassifies leaked thought with timestamp to own_thought", () => {
  const result = sanitizeModelOutput(
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

Deno.test("sanitizeModelOutput reclassifies leaked thought without timestamp to own_thought", () => {
  const result = sanitizeModelOutput(
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

Deno.test("sanitizeModelOutput does not reclassify normal utterances", () => {
  const output = [ownUtteranceTurn("Hello! How can I help you?")];
  const result = sanitizeModelOutput(
    [participantUtteranceTurn({ name: "user", text: "hi" })],
    output,
  );
  assertEquals(result.emit.length, 1);
  const event = result.emit[0];
  assertEquals(event.type, "own_utterance");
  if (event.type !== "own_utterance") throw new Error("unreachable");
  assertEquals(event.text, "Hello! How can I help you?");
});

Deno.test("stripFabricatedUserMessages reclassifies entirely fabricated user message to own_thought", () => {
  const names = new Set(["אורח/ת"]);
  const output = [
    ownUtteranceTurn(
      "אורח/ת: לא ממש משנה לי המחיר, משהו איכותי. שמעתי על Miele שהם טובים — sent Mar 30, 2026, 3:12 PM",
    ),
  ];
  const result = stripFabricatedUserMessages(names, output);
  assertEquals(result.length, 1);
  assertEquals(result[0].type, "own_thought");
});

Deno.test("stripFabricatedUserMessages strips fabricated lines and keeps real response", () => {
  const names = new Set(["אורח/ת"]);
  const output = [
    ownUtteranceTurn(
      "אורח/ת: אני מחפש תנור חדש\nבטח! יש לנו כמה אפשרויות מצוינות.",
    ),
  ];
  const result = stripFabricatedUserMessages(names, output);
  assertEquals(result.length, 1);
  assertEquals(result[0].type, "own_utterance");
  if (result[0].type !== "own_utterance") throw new Error("unreachable");
  assertEquals(result[0].text, "בטח! יש לנו כמה אפשרויות מצוינות.");
});

Deno.test("stripFabricatedUserMessages ignores normal utterances", () => {
  const names = new Set(["user"]);
  const output = [
    ownUtteranceTurn("Hello! How can I help you today?"),
  ];
  const result = stripFabricatedUserMessages(names, output);
  assertEquals(result.length, 1);
  assertEquals(result[0].type, "own_utterance");
  if (result[0].type !== "own_utterance") throw new Error("unreachable");
  assertEquals(result[0].text, "Hello! How can I help you today?");
});

Deno.test("stripFabricatedUserMessages handles no participant names", () => {
  const names = new Set<string>();
  const output = [
    ownUtteranceTurn("user: some text"),
  ];
  const result = stripFabricatedUserMessages(names, output);
  assertEquals(result.length, 1);
  assertEquals(result[0].type, "own_utterance");
});

Deno.test("sanitizeModelOutput strips fabricated user messages from model output", () => {
  const history = [
    participantUtteranceTurn({ name: "אורח/ת", text: "שלום, אני מחפש תנור" }),
  ];
  const output = [
    ownUtteranceTurn(
      "אורח/ת: לא ממש משנה לי המחיר, משהו איכותי\nבטח! הנה כמה אפשרויות.",
    ),
  ];
  const result = sanitizeModelOutput(
    history,
    output,
  );
  assertEquals(result.emit.length, 1);
  assertEquals(result.emit[0].type, "own_utterance");
  if (result.emit[0].type !== "own_utterance") throw new Error("unreachable");
  assertEquals(result.emit[0].text, "בטח! הנה כמה אפשרויות.");
});

Deno.test("sanitizeModelOutput reclassifies fully fabricated user message to own_thought", () => {
  const history = [
    participantUtteranceTurn({ name: "אורח/ת", text: "שלום" }),
  ];
  const output = [
    ownUtteranceTurn(
      "אורח/ת: אני אבדוק את הקישור ששלחת. תגיד, מה האחריות שיש על התנור הזה?",
    ),
  ];
  const result = sanitizeModelOutput(
    history,
    output,
  );
  assertEquals(result.emit.length, 1);
  assertEquals(result.emit[0].type, "own_thought");
});
