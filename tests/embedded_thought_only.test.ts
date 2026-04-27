import { assert, assertEquals } from "@std/assert";
import {
  geminiOutputToHistoryEvents,
  type GeminiPartOfInterest,
} from "../src/geminiAgent.ts";

// Repro for silent loop exit observed in production: when a Gemini text part
// contains only embedded `[Internal thought, visible only to you: ...]`
// patterns but does not match the strict anchored-thought regex, the part
// collapses to empty after stripping and the conversion previously returned
// `null`, dropping the event entirely. The agent loop then exited with no
// communication to the user. Expected behavior: emit an `own_thought` event
// carrying the embedded thought content so the loop continues.

Deno.test("text part with embedded thought followed by whitespace-only content becomes own_thought", () => {
  const parts: GeminiPartOfInterest[] = [
    {
      type: "text",
      text:
        "  [Internal thought, visible only to you: I have all three scenes.]  ",
      thoughtSignature: "sig-abc",
    },
  ];
  const events = geminiOutputToHistoryEvents(parts);
  assert(
    events.length > 0,
    `expected non-empty events, got ${JSON.stringify(events)}`,
  );
  assertEquals(events[0].type, "own_thought");
});

Deno.test("text part with embedded thought plus trailing whitespace becomes own_thought", () => {
  const parts: GeminiPartOfInterest[] = [
    {
      type: "text",
      text: "[Internal thought, visible only to you: ok done] \n",
      thoughtSignature: "sig-xyz",
    },
  ];
  const events = geminiOutputToHistoryEvents(parts);
  assert(
    events.length > 0,
    `expected non-empty events, got ${JSON.stringify(events)}`,
  );
  assertEquals(events[0].type, "own_thought");
});
