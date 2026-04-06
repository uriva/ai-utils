import { assert } from "@std/assert";
import { buildReq } from "../src/geminiAgent.ts";
import { ownThoughtTurn } from "../src/agent.ts";

Deno.test(
  "own_thought with modelMetadata should not render as visible text in buildReq",
  () => {
    // deno-lint-ignore no-explicit-any
    const history: any[] = [
      {
        type: "participant_utterance",
        isOwn: false,
        id: "msg0",
        timestamp: 100,
        name: "user",
        text: "Hello",
      },
      // Model-generated thought with real thoughtSignature (PATH 1)
      {
        type: "own_thought",
        isOwn: true,
        id: "msg1",
        timestamp: 200,
        text: "I should greet them back.",
        modelMetadata: {
          type: "gemini",
          responseId: "resp-1",
          thoughtSignature: "real-sig-abc",
        },
      },
      // Model utterance from the same response
      {
        type: "own_utterance",
        isOwn: true,
        id: "msg2",
        timestamp: 200,
        text: "Hi there!",
        modelMetadata: {
          type: "gemini",
          responseId: "resp-1",
          thoughtSignature: "",
        },
      },
    ];

    const req = buildReq(false, false, "system prompt", [], "UTC", undefined)(
      history,
    );

    const allText = JSON.stringify(req.contents);

    // The thought text should NOT appear as "[Internal thought, visible only to you: ...]"
    assert(
      !allText.includes("Internal thought, visible only to you"),
      `Model-generated thought should not be rendered as [Internal thought] text, but found in contents:\n${
        JSON.stringify(req.contents, null, 2)
      }`,
    );
  },
);

Deno.test(
  "own_thought with modelMetadata and empty signature should not render as visible text",
  () => {
    // This simulates a reclassified leaked thought (PATH 2)
    // deno-lint-ignore no-explicit-any
    const history: any[] = [
      {
        type: "participant_utterance",
        isOwn: false,
        id: "msg0",
        timestamp: 100,
        name: "user",
        text: "Hello",
      },
      {
        type: "own_thought",
        isOwn: true,
        id: "msg3",
        timestamp: 300,
        text: "This is a reclassified leaked thought.",
        modelMetadata: {
          type: "gemini",
          responseId: "resp-2",
          thoughtSignature: "",
        },
      },
      {
        type: "own_utterance",
        isOwn: true,
        id: "msg4",
        timestamp: 300,
        text: "Hello!",
        modelMetadata: {
          type: "gemini",
          responseId: "resp-2",
          thoughtSignature: "",
        },
      },
    ];

    const req = buildReq(false, false, "system prompt", [], "UTC", undefined)(
      history,
    );

    const allText = JSON.stringify(req.contents);

    assert(
      !allText.includes("Internal thought, visible only to you"),
      `Reclassified leaked thought should not be rendered as [Internal thought] text, but found in contents:\n${
        JSON.stringify(req.contents, null, 2)
      }`,
    );
  },
);

Deno.test(
  "synthetic own_thought (no modelMetadata) should still render as System notification",
  () => {
    const syntheticThought = ownThoughtTurn("You should check for errors.");
    // deno-lint-ignore no-explicit-any
    const history: any[] = [
      {
        type: "participant_utterance",
        isOwn: false,
        id: "msg0",
        timestamp: 100,
        name: "user",
        text: "Hello",
      },
      syntheticThought,
    ];

    const req = buildReq(false, false, "system prompt", [], "UTC", undefined)(
      history,
    );

    const allText = JSON.stringify(req.contents);

    assert(
      allText.includes("System notification"),
      `Synthetic thought should render as [System notification], but got:\n${
        JSON.stringify(req.contents, null, 2)
      }`,
    );
  },
);
