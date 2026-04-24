import { assert } from "@std/assert";
import { z } from "zod/v4";
import { tool } from "../mod.ts";
import type { HistoryEvent } from "../src/agent.ts";
import {
  agentDeps,
  noopRewriteHistory,
  runForAllProviders,
} from "../test_helpers.ts";

// Gemini-specific: thoughtSignature is a Gemini concept. The bug this test
// reproduces is that `geminiOutputPartToHistoryEvent` emits tool_call events
// without inline `modelMetadata`, causing downstream filters
// (`filterInvalidToolCalls`) to strip them with "missing thought signature".
runForAllProviders(
  "tool_call events emitted by Gemini carry inline modelMetadata.thoughtSignature",
  async (runAgent) => {
    const history: HistoryEvent[] = [
      {
        type: "participant_utterance",
        id: "user-1",
        timestamp: 1,
        isOwn: false,
        name: "user",
        text: "What is the weather in Paris? Use the get_weather tool.",
      },
    ];

    const weatherTool = tool({
      name: "get_weather",
      description: "Gets the current weather for a city.",
      parameters: z.object({ city: z.string() }),
      handler: ({ city }: { city: string }) =>
        Promise.resolve(`Sunny, 22C in ${city}`),
    });

    await agentDeps(history)(runAgent)({
      maxIterations: 1,
      onMaxIterationsReached: () => {},
      tools: [weatherTool],
      prompt:
        "You are a helpful assistant. You MUST use the provided tools to answer questions about the weather — never guess. When asked about weather, always call get_weather with the city.",
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });

    const toolCall = history.find((e) => e.type === "tool_call");
    assert(
      toolCall,
      `Expected a tool_call event in history, got: ${
        history.map((e) => e.type).join(", ")
      }`,
    );
    assert(
      "modelMetadata" in toolCall,
      `tool_call event missing modelMetadata key; event: ${
        JSON.stringify(toolCall)
      }`,
    );
    const meta = (toolCall as { modelMetadata?: { thoughtSignature?: string } })
      .modelMetadata;
    assert(
      meta?.thoughtSignature && meta.thoughtSignature.trim().length > 0,
      `tool_call.modelMetadata.thoughtSignature missing or empty: ${
        JSON.stringify(meta)
      }`,
    );
  },
  3,
  true,
);
