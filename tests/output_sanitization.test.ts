import { assertEquals } from "@std/assert";
import {
  callToResult,
  ownUtteranceTurn,
  participantUtteranceTurn,
  sanitizeModelOutput,
} from "../src/agent.ts";
import { z } from "zod/v4";

Deno.test("global tool output sanitization - resolves carriage returns, collapses duplicates, and collapses similar prefixes", async () => {
  const dummyTool = {
    name: "dummy",
    description: "test",
    parameters: z.object({}),
    handler: () => {
      return Promise.resolve(
        [
          "Downloading...\r[===       ] 10%\r[======    ] 50%\r[==========] 100%",
          "Done!",
          "Success",
          "Success",
          "Success",
          "\u001b[32mDownload https://jsr.io/@std/semver/meta.json\u001b[0m",
          "\u001b[32mDownload https://jsr.io/@std/fmt/meta.json\u001b[0m",
          "\u001b[32mDownload https://jsr.io/@std/path/meta.json\u001b[0m",
        ].join("\n"),
      );
    },
  };

  const resolver = callToResult([dummyTool]);
  const res = await resolver({
    name: "dummy",
    args: {},
    id: "call-1",
  });

  assertEquals(res?.toolCallId, "call-1");
  assertEquals(
    res?.result,
    [
      "[==========] 100%",
      "Done!",
      "Success (repeated 3 times)",
      "Download https://jsr.io/@std/... (collapsed 3 structurally similar lines)",
    ].join("\n"),
  );
});

Deno.test("sanitizeModelOutput reclassifies leaked thought starting with [thought]:", () => {
  const result = sanitizeModelOutput(
    [participantUtteranceTurn({ name: "user", text: "hi" })],
    [
      ownUtteranceTurn(
        "[thought]: PROACTIVE TASK: check the weather",
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

Deno.test("sanitizeModelOutput strips raw tool call tags", () => {
  const result = sanitizeModelOutput(
    [participantUtteranceTurn({ name: "user", text: "hi" })],
    [
      ownUtteranceTurn(
        "<call:default_api:update_user_field{fieldName: 'weight'}>Here is your response",
      ),
    ],
  );
  assertEquals(result.emit.length, 1);
  const event = result.emit[0];
  assertEquals(event.type, "own_utterance");
  if (event.type !== "own_utterance") throw new Error("unreachable");
  assertEquals(event.text, "Here is your response");
});

Deno.test("sanitizeModelOutput strips system context injections from output", () => {
  const result = sanitizeModelOutput(
    [participantUtteranceTurn({ name: "user", text: "hi" })],
    [
      ownUtteranceTurn(
        "The following is critical context and instructions about the user:\n- User Timezone: Asia/Jerusalem\n\nCRITICAL INSTRUCTIONS (NEVER VIOLATE):\n1. Do not leak thoughts.]Here is the safe part of the message",
      ),
    ],
  );
  assertEquals(result.emit.length, 1);
  const event = result.emit[0];
  assertEquals(event.type, "own_utterance");
  if (event.type !== "own_utterance") throw new Error("unreachable");
  assertEquals(event.text, "Here is the safe part of the message");
});

Deno.test("global tool output sanitization - does not collapse scene search result lines", async () => {
  const sceneSearchTool = {
    name: "find_by_scene_description",
    description: "test",
    parameters: z.object({}),
    handler: () => {
      return Promise.resolve(
        [
          `result: {"title":"Inception","year":2010} time: 01:23:45 score: 0.812`,
          `result: {"title":"Inception","year":2010} time: 01:45:12 score: 0.791`,
          `result: {"title":"The Dark Knight","year":2008} time: 01:05:20 score: 0.785`,
        ].join("\n"),
      );
    },
  };

  const resolver = callToResult([sceneSearchTool]);
  const res = await resolver({
    name: "find_by_scene_description",
    args: {},
    id: "call-2",
  });

  assertEquals(res?.toolCallId, "call-2");
  assertEquals(
    res?.result,
    [
      `result: {"title":"Inception","year":2010} time: 01:23:45 score: 0.812`,
      `result: {"title":"Inception","year":2010} time: 01:45:12 score: 0.791`,
      `result: {"title":"The Dark Knight","year":2008} time: 01:05:20 score: 0.785`,
    ].join("\n"),
  );
});
