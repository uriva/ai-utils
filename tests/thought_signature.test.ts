import { assert } from "@std/assert";
import { buildReq } from "../src/geminiAgent.ts";
import { z } from "zod/v4";

Deno.test(
  "thought_signature propagation to functionCall in buildReq",
  () => {
    // deno-lint-ignore no-explicit-any
    const tools: any = [{
      name: "get_weather",
      description: "Get the weather in a city",
      parameters: z.object({ city: z.string() }),
      handler: () => Promise.resolve("done"),
    }];

    const metadata = {
      type: "gemini" as const,
      responseId: "response-123",
      thoughtSignature: "sig-123",
    };

    // deno-lint-ignore no-explicit-any
    const history: any[] = [
      {
        type: "own_thought",
        isOwn: true,
        id: "msg1",
        timestamp: 123,
        text: "I should check the weather.",
        modelMetadata: { ...metadata, thoughtSignature: "sig-123" },
      },
      {
        type: "tool_call",
        isOwn: true,
        id: "msg2",
        timestamp: 123,
        name: "get_weather",
        parameters: { city: "Paris" },
        modelMetadata: { ...metadata, thoughtSignature: "" },
      },
    ];

    const reqBuilder = buildReq(
      false, // lightModel
      "system prompt",
      tools,
      "UTC",
      undefined,
    );

    const req = reqBuilder(history);

    // deno-lint-ignore no-explicit-any
    const contents: any = req.contents;
    console.log(JSON.stringify(contents, null, 2));

    // There should be a user turn added initially, then our model turn
    // deno-lint-ignore no-explicit-any
    const modelTurn = contents.find((c: any) => c.role === "model");

    const parts = modelTurn.parts;
    assert(parts && parts.length === 2, "Expected 2 parts");

    // deno-lint-ignore no-explicit-any
    const functionCallPart = parts.find((p: any) => p.functionCall);
    assert(functionCallPart, "Expected functionCallPart");

    // THE BUG: functionCallPart.thoughtSignature should be "sig-123", not "" or undefined
    if (functionCallPart.thoughtSignature !== "sig-123") {
      throw new Error(
        `Reproduced thought_signature missing bug: thoughtSignature is ${functionCallPart.thoughtSignature}`,
      );
    }
  },
);

Deno.test(
  "buildReq does not smear a signature onto synthesized placeholder parts (400 INVALID_ARGUMENT repro)",
  () => {
    // Root cause of the production 400: events sharing a responseId are combined
    // into one Content. A signature-less thought becomes a synthesized `{text: " "}`
    // placeholder. combineContent used to copy the sibling functionCall's
    // signature onto EVERY part — including that placeholder and a plain
    // utterance that never had one. Gemini rejects a signature on a part it
    // never returned one on with 400 INVALID_ARGUMENT.
    const md = (thoughtSignature = "") => ({
      type: "gemini" as const,
      responseId: "resp-1",
      thoughtSignature,
    });
    // deno-lint-ignore no-explicit-any
    const history: any[] = [
      {
        type: "participant_utterance",
        isOwn: false,
        name: "user",
        text: "deploy it",
        attachments: [],
        id: "u1",
        timestamp: 0,
      },
      // signature-less thought -> synthesized `{text: " "}` placeholder part
      {
        type: "own_thought",
        isOwn: true,
        id: "t1",
        timestamp: 0,
        text: "planning",
        modelMetadata: md(""),
      },
      // functionCall carrying the real signature, same responseId
      {
        type: "tool_call",
        isOwn: true,
        id: "tc1",
        timestamp: 0,
        name: "learn_skill",
        parameters: { skillName: "p2b-deno-deploy" },
        modelMetadata: md("sig-abc"),
      },
      // plain model utterance that never carried a signature, same responseId
      {
        type: "own_utterance",
        isOwn: true,
        id: "u2",
        timestamp: 0,
        text: "here you go",
        modelMetadata: md(""),
      },
    ];

    const req = buildReq(false, "system", [], "UTC", undefined)(history);
    // deno-lint-ignore no-explicit-any
    const contents: any[] = req.contents as any[];
    // deno-lint-ignore no-explicit-any
    const modelTurn = contents.find((c: any) => c.role === "model");
    // deno-lint-ignore no-explicit-any
    const textParts = modelTurn.parts.filter((p: any) =>
      typeof p.text === "string" && p.functionCall == null && !p.thought
    );
    for (const p of textParts) {
      if ("thoughtSignature" in p) {
        throw new Error(
          `Signature must not be smeared onto a plain/placeholder text part: ${
            JSON.stringify(p)
          }`,
        );
      }
    }

    // The functionCall must still carry its own signature (round-trip intact).
    // deno-lint-ignore no-explicit-any
    const fnPart = modelTurn.parts.find((p: any) => p.functionCall);
    assert(fnPart, "expected functionCall part");
    assert(
      fnPart.thoughtSignature === "sig-abc",
      `functionCall should retain its signature, got ${fnPart.thoughtSignature}`,
    );
  },
);

Deno.test(
  "buildReq preserves a legitimately-signed utterance's own signature (round-trip)",
  () => {
    // Gemini can return a signature on a plain text part; the round-trip
    // contract requires sending it back on that same part. We must NOT strip it.
    // deno-lint-ignore no-explicit-any
    const history: any[] = [
      {
        type: "participant_utterance",
        isOwn: false,
        name: "user",
        text: "hi",
        attachments: [],
        id: "u1",
        timestamp: 0,
      },
      {
        type: "own_utterance",
        isOwn: true,
        id: "u2",
        timestamp: 0,
        text: "hello there",
        modelMetadata: {
          type: "gemini" as const,
          responseId: "resp-2",
          thoughtSignature: "sig-own",
        },
      },
    ];
    const req = buildReq(false, "system", [], "UTC", undefined)(history);
    // deno-lint-ignore no-explicit-any
    const contents: any[] = req.contents as any[];
    // deno-lint-ignore no-explicit-any
    const modelTurn = contents.find((c: any) => c.role === "model");
    // deno-lint-ignore no-explicit-any
    const textPart = modelTurn.parts.find((p: any) => p.text === "hello there");
    assert(textPart, "expected the utterance text part");
    assert(
      textPart.thoughtSignature === "sig-own",
      `a legitimately-signed utterance must keep its own signature, got ${textPart.thoughtSignature}`,
    );
  },
);

Deno.test(
  "buildReq does not attach a signature to a do_nothing placeholder part",
  () => {
    // The production 400 turn ended with `do_nothing`, which renders as a
    // synthesized `" "` placeholder. It must not carry a thoughtSignature.
    // deno-lint-ignore no-explicit-any
    const history: any[] = [
      {
        type: "participant_utterance",
        isOwn: false,
        name: "user",
        text: "go",
        attachments: [],
        id: "u1",
        timestamp: 0,
      },
      {
        type: "do_nothing",
        isOwn: true,
        id: "dn1",
        timestamp: 0,
        modelMetadata: {
          type: "gemini" as const,
          responseId: "resp-3",
          thoughtSignature: "sig-dn",
        },
      },
    ];
    const req = buildReq(false, "system", [], "UTC", undefined)(history);
    // deno-lint-ignore no-explicit-any
    const contents: any[] = req.contents as any[];
    for (const c of contents) {
      for (const p of c.parts ?? []) {
        if ("thoughtSignature" in p && p.functionCall == null) {
          throw new Error(
            `do_nothing placeholder must not carry a signature: ${
              JSON.stringify(p)
            }`,
          );
        }
      }
    }
  },
);
