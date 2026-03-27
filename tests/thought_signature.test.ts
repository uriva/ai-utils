import { assert } from "@std/assert";
import { geminiAgentCaller } from "../src/geminiAgent.ts";
import { z } from "zod/v4";

Deno.test(
  "thought_signature propagation to functionCall",
  async () => {
    // This test ensures that when the model generates a tool call, and then we reply,
    // the subsequent turn doesn't fail with a thought_signature missing error.

    const tools = [{
      name: "get_weather",
      description: "Get the weather in a city",
      parameters: z.object({ city: z.string() }),
      handler: () => {
        return Promise.resolve("done");
      },
    }];

    const agent = geminiAgentCaller({
      maxIterations: 1,
      onMaxIterationsReached: () => {},
      timezoneIANA: "UTC",
      prompt:
        "You are a helpful assistant. Always use the get_weather tool when asked about weather.",
      tools,
      skills: [],
      rewriteHistory: () => Promise.resolve(), // Identity
    });

    // deno-lint-ignore no-explicit-any
    let history: any[] = [{
      type: "own_utterance",
      id: "u1",
      timestamp: Date.now(),
      isOwn: false,
      text: "What is the weather in Paris?",
    }];

    // First turn: model should decide to call get_weather
    const firstTurn = await agent(history);
    history = [...history, ...firstTurn];

    // Check if there's a tool call
    const toolCall = firstTurn.find((e: import("../mod.ts").HistoryEvent) =>
      e.type === "tool_call"
    );
    assert(toolCall, "Model did not emit a tool call");

    // Reply with the tool result
    history.push({
      type: "tool_result",
      id: "r1",
      timestamp: Date.now(),
      isOwn: false,
      toolCallId: toolCall.id,
      result: JSON.stringify({ temperature: "22C", condition: "Sunny" }),
    });

    // Second turn: this is where it might crash with a 400 Bad Request
    // if thought_signature is missing from the history.
    try {
      const secondTurn = await agent(history);
      history = [...history, ...secondTurn];
      const answer = secondTurn.find((e) =>
        e.type === "own_utterance" || e.type === "own_thought"
      );
      assert(answer, "Model should generate an answer");
    } catch (e) {
      if (e instanceof Error && e.message.includes("thought_signature")) {
        throw new Error(
          `Reproduced thought_signature missing bug: ${e.message}`,
        );
      }
      throw e;
    }
  },
);
