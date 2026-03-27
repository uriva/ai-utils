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
      false, // imageGen
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
