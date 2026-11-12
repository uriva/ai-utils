import { assert, assertEquals, assertThrows } from "@std/assert";
import { z } from "zod/v4";
import {
  participantUtteranceTurn,
  tool,
  validateSchema,
  validateZodSchema,
  zodToGeminiParameters,
} from "../mod.ts";
import type { HistoryEvent } from "../src/agent.ts";
import { agentDeps, runForAllProviders } from "../test_helpers.ts";

runForAllProviders(
  "runAgent handles boolean and numeric z.literal tool parameters without 400 enum TYPE_STRING error",
  async (runAgentWithProvider) => {
    const received: { name: string; confirmed: true; step: 1 }[] = [];
    const confirmTool = tool({
      name: "confirmMeeting",
      description: "Confirm a scheduled meeting after explicit user approval.",
      parameters: z.object({
        name: z.string(),
        confirmed: z.literal(true),
        step: z.literal(1),
      }),
      handler: (args) => {
        received.push(args);
        return Promise.resolve("Meeting confirmed.");
      },
    });

    validateZodSchema(confirmTool.parameters, "confirmMeeting");
    const geminiParams = zodToGeminiParameters(
      confirmTool.parameters,
    ) as unknown as {
      properties: Record<string, { enum?: unknown[]; description?: string }>;
    };
    assertEquals(geminiParams.properties.confirmed.enum, undefined);
    assertEquals(geminiParams.properties.step.enum, undefined);

    assertThrows(() =>
      validateSchema(
        {
          type: "object",
          properties: { confirmed: { type: "boolean", enum: [true] } },
        },
        "badSchema",
      )
    );

    const history: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text:
          'Call confirmMeeting now with name="Alice", confirmed=true, step=1, then reply "Done".',
      }),
    ];

    await agentDeps(history)(runAgentWithProvider)({
      maxIterations: 4,
      tools: [confirmTool],
      prompt: "You are a helpful assistant. Call tools when instructed.",
      timezoneIANA: "UTC",
    });

    assertEquals(received.length, 1);
    assertEquals(received[0], { name: "Alice", confirmed: true, step: 1 });
    assert(history.some((e) => e.type === "own_utterance"));
  },
);
