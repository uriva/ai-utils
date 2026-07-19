import { assertEquals, assertThrows } from "@std/assert";
import {
  callToResult,
  ownUtteranceTurn,
  participantUtteranceTurn,
  sanitizeModelOutput,
  toolUseTurn,
} from "../src/agent.ts";
import { z } from "zod/v4";

const isRecord = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v);

const paramField = (params: unknown, key: string): unknown =>
  isRecord(params) ? params[key] : undefined;

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

// Gemini (flash in particular) intermittently renders a tool call as plain
// visible text instead of a real function_call. Observed verbatim in the wild
// (Agent FOMO) and reproduced from the production snapshot:
//   "startcall:default_api:run_command{command: event_discovery/query ...}"
// The intended action is recovered into a real tool_call so it actually runs.
// It must NEVER be reclassified as an own_thought: that would make the model
// believe the action already happened and relay fabricated results to the user.
Deno.test("sanitizeModelOutput recovers a mangled tool call leaked as visible text into a real tool_call", () => {
  const leaked =
    "startcall:default_api:run_command{command: event_discovery/query ,params:{location: Haifa}, spinnerText: בודקת אירועים קרובים בחיפה...}";
  const result = sanitizeModelOutput(
    [participantUtteranceTurn({
      name: "user",
      text: "מה קורה מחר בערב בחיפה?",
    })],
    [ownUtteranceTurn(leaked)],
  );
  assertEquals(result.emit.length, 1);
  const event = result.emit[0];
  assertEquals(event.type, "tool_call");
  if (event.type !== "tool_call") throw new Error("unreachable");
  assertEquals(event.name, "run_command");
  assertEquals(
    paramField(event.parameters, "command"),
    "event_discovery/query",
  );
});

Deno.test("sanitizeModelOutput recovers other Gemini mangled tool-call renderings into tool_calls", () => {
  const cases: { leaked: string; name: string }[] = [
    {
      leaked:
        "print(default_api.run_command(command='event_discovery/query', params={}))",
      name: "run_command",
    },
    {
      leaked: "default_api.learn_skill(skillName='event_discovery')",
      name: "learn_skill",
    },
    {
      leaked: "```tool_code\ndefault_api.query(location='Haifa')\n```",
      name: "query",
    },
  ];
  for (const { leaked, name } of cases) {
    const result = sanitizeModelOutput(
      [participantUtteranceTurn({ name: "user", text: "hi" })],
      [ownUtteranceTurn(leaked)],
    );
    assertEquals(result.emit.length, 1);
    const event = result.emit[0];
    assertEquals(
      event.type,
      "tool_call",
      `expected mangled tool call to be recovered: ${leaked}`,
    );
    if (event.type !== "tool_call") throw new Error("unreachable");
    assertEquals(event.name, name, `wrong recovered tool name for: ${leaked}`);
  }
});

Deno.test("sanitizeModelOutput never turns a mangled tool call into an own_thought", () => {
  const leaked =
    "startcall:default_api:run_command{command: event_discovery/query ,params:{location: Haifa}}";
  const result = sanitizeModelOutput(
    [participantUtteranceTurn({ name: "user", text: "hi" })],
    [ownUtteranceTurn(leaked)],
  );
  for (const event of result.emit) {
    assertEquals(
      event.type === "own_thought",
      false,
      "mangled tool calls must not be reclassified as thoughts",
    );
  }
});

Deno.test("sanitizeModelOutput throws when a mangled tool call cannot be recovered", () => {
  // Detected as a tool-call preamble but no tool name is recoverable. We must
  // throw (loud failure -> loop retries) rather than leak or drop the action.
  assertThrows(() =>
    sanitizeModelOutput(
      [participantUtteranceTurn({ name: "user", text: "hi" })],
      [ownUtteranceTurn("startcall:default_api:")],
    )
  );
});

Deno.test("sanitizeModelOutput recovers learn_skill args from a mangled call", () => {
  const result = sanitizeModelOutput(
    [participantUtteranceTurn({ name: "user", text: "hi" })],
    [ownUtteranceTurn("default_api.learn_skill(skillName='event_discovery')")],
  );
  const event = result.emit[0];
  assertEquals(event.type, "tool_call");
  if (event.type !== "tool_call") throw new Error("unreachable");
  assertEquals(paramField(event.parameters, "skillName"), "event_discovery");
});

Deno.test("sanitizeModelOutput leaves normal utterances that merely mention tools untouched", () => {
  const normal =
    "מחר בערב בחיפה די שקט כרגע. רוצה שאבדוק ביום אחר או בעיר אחרת?";
  const result = sanitizeModelOutput(
    [participantUtteranceTurn({
      name: "user",
      text: "מה קורה מחר בערב בחיפה?",
    })],
    [ownUtteranceTurn(normal)],
  );
  assertEquals(result.emit.length, 1);
  const event = result.emit[0];
  assertEquals(event.type, "own_utterance");
  if (event.type !== "own_utterance") throw new Error("unreachable");
  assertEquals(event.text, normal);
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

Deno.test("sanitizeModelOutput reclassifies an unclosed <thought> tag leak as own_thought", () => {
  const result = sanitizeModelOutput(
    [participantUtteranceTurn({ name: "user", text: "hi" })],
    [
      ownUtteranceTurn(
        "<thought> I should call the update tool first, then reply.",
      ),
      toolUseTurn({ name: "update_entry", args: { weight: 76 } }),
    ],
  );
  assertEquals(result.emit.length, 2);
  const [thought, toolCall] = result.emit;
  assertEquals(thought.type, "own_thought");
  assertEquals(toolCall.type, "tool_call");
  if (thought.type !== "own_thought") throw new Error("unreachable");
  assertEquals(
    thought.text,
    "I should call the update tool first, then reply.",
  );
});

Deno.test("sanitizeModelOutput reclassifies a full-output <thought> tag leak as own_thought", () => {
  const result = sanitizeModelOutput(
    [participantUtteranceTurn({ name: "user", text: "hi" })],
    [
      ownUtteranceTurn(
        "<thought> checking if it is morning or evening. it is evening. no action needed, reply politely.",
      ),
    ],
  );
  assertEquals(result.emit.length, 1);
  const event = result.emit[0];
  assertEquals(event.type, "own_thought");
  if (event.type !== "own_thought") throw new Error("unreachable");
  assertEquals(
    event.text,
    "checking if it is morning or evening. it is evening. no action needed, reply politely.",
  );
});

Deno.test("sanitizeModelOutput drops bare <thought> fragments entirely", () => {
  const outputs = ["<thought>", "<thought"].map((fragment) =>
    sanitizeModelOutput(
      [participantUtteranceTurn({ name: "user", text: "hi" })],
      [ownUtteranceTurn(fragment)],
    )
  );
  for (const result of outputs) {
    assertEquals(
      result.emit.filter((e) => e.type === "own_utterance").length,
      0,
    );
  }
});

Deno.test("sanitizeModelOutput splits a closed <thought> block from the visible reply", () => {
  const result = sanitizeModelOutput(
    [participantUtteranceTurn({ name: "user", text: "hi" })],
    [ownUtteranceTurn(
      "<thought>checking the dates first</thought>היי! מה נשמע?",
    )],
  );
  assertEquals(result.emit.length, 2);
  const [thought, utterance] = result.emit;
  assertEquals(thought.type, "own_thought");
  assertEquals(utterance.type, "own_utterance");
  if (thought.type !== "own_thought" || utterance.type !== "own_utterance") {
    throw new Error("unreachable");
  }
  assertEquals(thought.text, "checking the dates first");
  assertEquals(utterance.text, "היי! מה נשמע?");
});

Deno.test("sanitizeModelOutput reclassifies a leading reasoning utterance in a multi-utterance response", () => {
  const reasoning =
    "אני מבין שהמשתמש לא יהיה בבית מחר בבוקר, אז השקילה תהיה ביום שני. אני אענה לו בחום ואאשר את התוכנית.";
  const reply = "מובן לגמרי! הכל בסדר, נעדכן ביום שני בבוקר. 👌";
  const result = sanitizeModelOutput(
    [participantUtteranceTurn({ name: "user", text: "hi" })],
    [ownUtteranceTurn(reasoning), ownUtteranceTurn(reply)],
  );
  assertEquals(result.emit.length, 2);
  const [thought, utterance] = result.emit;
  assertEquals(thought.type, "own_thought");
  assertEquals(utterance.type, "own_utterance");
  if (thought.type !== "own_thought" || utterance.type !== "own_utterance") {
    throw new Error("unreachable");
  }
  assertEquals(thought.text, reasoning);
  assertEquals(utterance.text, reply);
});

Deno.test("sanitizeModelOutput reclassifies narration of a tool called in the same response", () => {
  const narration = "אני משתמש ב-send_reminder כדי לתזכר את המשתמש מחר בבוקר.";
  const result = sanitizeModelOutput(
    [participantUtteranceTurn({ name: "user", text: "hi" })],
    [
      ownUtteranceTurn(narration),
      toolUseTurn({ name: "send_reminder", args: {} }),
    ],
  );
  assertEquals(result.emit.length, 2);
  const [thought, toolCall] = result.emit;
  assertEquals(thought.type, "own_thought");
  assertEquals(toolCall.type, "tool_call");
  if (thought.type !== "own_thought") throw new Error("unreachable");
  assertEquals(thought.text, narration);
});

Deno.test("sanitizeModelOutput leaves an utterance naming a tool that is not called in the response", () => {
  const text = "יש לי כלי בשם send_reminder שמאפשר לי לתזכר אותך.";
  const result = sanitizeModelOutput(
    [participantUtteranceTurn({ name: "user", text: "hi" })],
    [ownUtteranceTurn(text)],
  );
  assertEquals(result.emit.length, 1);
  const event = result.emit[0];
  assertEquals(event.type, "own_utterance");
  if (event.type !== "own_utterance") throw new Error("unreachable");
  assertEquals(event.text, text);
});

Deno.test("sanitizeModelOutput leaves a user-facing preamble accompanying a tool call untouched", () => {
  const text = "רגע אחד, אני בודקת את היומן שלך 👀";
  const result = sanitizeModelOutput(
    [participantUtteranceTurn({ name: "user", text: "hi" })],
    [
      ownUtteranceTurn(text),
      toolUseTurn({ name: "check_calendar", args: {} }),
    ],
  );
  assertEquals(result.emit.length, 2);
  const event = result.emit[0];
  assertEquals(event.type, "own_utterance");
  if (event.type !== "own_utterance") throw new Error("unreachable");
  assertEquals(event.text, text);
});

Deno.test("sanitizeModelOutput splits and reclassifies paragraph-level tool narration", () => {
  const narration = "I will call the get_weather tool now.";
  const preamble = "Checking the weather for you.";
  const text = `${narration}\n\n${preamble}`;
  const result = sanitizeModelOutput(
    [participantUtteranceTurn({ name: "user", text: "hi" })],
    [
      ownUtteranceTurn(text),
      toolUseTurn({ name: "get_weather", args: {} }),
    ],
  );
  assertEquals(result.emit.length, 3);
  const [thought, utterance, toolCall] = result.emit;
  assertEquals(thought.type, "own_thought");
  assertEquals(utterance.type, "own_utterance");
  assertEquals(toolCall.type, "tool_call");
  if (thought.type !== "own_thought" || utterance.type !== "own_utterance") {
    throw new Error("unreachable");
  }
  assertEquals(thought.text, narration);
  assertEquals(utterance.text, preamble);
});
