import {
  type AgentSpec,
  type CallModel,
  generateId,
  type HistoryEvent,
  injectAccessHistory,
  injectCallModel,
  injectOutputEvent,
  ownThoughtTurn,
  ownUtteranceTurn,
  participantUtteranceTurn,
  runAgent,
  tool,
  toolUseTurn,
  z,
} from "../mod.ts";
import { OpenAI } from "openai";

const apiKey = Deno.env.get("ZAI_API_KEY") || Deno.env.get("ZHIPU_API_KEY");
if (!apiKey) {
  throw new Error("ZAI_API_KEY or ZHIPU_API_KEY environment variable required");
}

const client = new OpenAI({
  apiKey,
  baseURL: "https://open.bigmodel.cn/api/paas/v4",
});

const model = "glm-5.3-flash";

// Build a clean OpenAI-compatible CallModel for GLM
const createGlmCaller = (spec: AgentSpec): CallModel => {
  return async (events: HistoryEvent[]): Promise<HistoryEvent[]> => {
    const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
      { role: "system", content: spec.prompt },
    ];

    const toolResultsById = new Map<string, string>();
    for (const e of events) {
      if (e.type === "tool_result") {
        const id = e.toolCallId || e.id;
        toolResultsById.set(
          id,
          typeof e.result === "string" ? e.result : JSON.stringify(e.result),
        );
      }
    }

    let i = 0;
    while (i < events.length) {
      const e = events[i];
      if (e.type === "participant_utterance") {
        messages.push({ role: "user", content: e.text });
        i++;
      } else if (e.type === "own_utterance") {
        messages.push({ role: "assistant", content: e.text });
        i++;
      } else if (e.type === "own_thought") {
        i++;
      } else if (e.type === "tool_call") {
        // Collect all consecutive tool_calls into a single assistant message
        const currentToolCalls:
          OpenAI.Chat.Completions.ChatCompletionMessageToolCall[] = [];
        while (i < events.length && events[i].type === "tool_call") {
          const tc = events[i] as Extract<HistoryEvent, { type: "tool_call" }>;
          currentToolCalls.push({
            id: tc.id,
            type: "function",
            function: {
              name: tc.name,
              arguments: JSON.stringify(tc.parameters),
            },
          });
          i++;
        }
        messages.push({
          role: "assistant",
          content: null,
          tool_calls: currentToolCalls,
        });
        // Immediately follow with the tool response messages for these tool calls
        for (const tc of currentToolCalls) {
          const res = toolResultsById.get(tc.id) ?? "[no result]";
          messages.push({
            role: "tool",
            tool_call_id: tc.id,
            content: res,
          });
        }
      } else if (e.type === "tool_result") {
        // Handled right after tool_calls
        i++;
      } else {
        i++;
      }
    }

    console.log(
      "=== SENDING TO GLM (messages count: " + messages.length + ") ===",
    );
    console.log(JSON.stringify(messages, null, 2));

    const tools: OpenAI.Chat.Completions.ChatCompletionTool[] = spec.tools.map((
      t,
    ) => ({
      type: "function",
      function: {
        name: t.name,
        description: t.description,
        parameters: z.toJSONSchema(t.parameters) as Record<string, unknown>,
      },
    }));

    const stream = await client.chat.completions.create({
      model,
      messages,
      tools: tools.length > 0 ? tools : undefined,
      stream: true,
      stream_options: { include_usage: true },
    });

    let reasoningText = "";
    let contentText = "";
    const toolCalls = new Map<
      number,
      { id: string; name: string; args: string }
    >();

    for await (const chunk of stream) {
      const delta = chunk.choices[0]?.delta as unknown as {
        content?: string;
        reasoning_content?: string;
        tool_calls?: Array<{
          index: number;
          id?: string;
          function?: { name?: string; arguments?: string };
        }>;
      };

      if (!delta) continue;

      if (delta.reasoning_content) {
        reasoningText += delta.reasoning_content;
      }
      if (delta.content) {
        contentText += delta.content;
      }
      if (delta.tool_calls) {
        for (const tc of delta.tool_calls) {
          const existing = toolCalls.get(tc.index) || {
            id: tc.id || generateId(),
            name: tc.function?.name || "",
            args: "",
          };
          if (tc.id) existing.id = tc.id;
          if (tc.function?.name) existing.name = tc.function.name;
          if (tc.function?.arguments) existing.args += tc.function.arguments;
          toolCalls.set(tc.index, existing);
        }
      }
    }

    const outputEvents: HistoryEvent[] = [];
    if (reasoningText) {
      outputEvents.push(ownThoughtTurn(reasoningText));
    }

    if (toolCalls.size > 0) {
      for (const tc of toolCalls.values()) {
        let args = {};
        try {
          args = JSON.parse(tc.args || "{}");
        } catch {
          args = { raw: tc.args };
        }
        outputEvents.push(toolUseTurn({
          id: tc.id,
          name: tc.name,
          args,
        }));
      }
    } else if (contentText) {
      outputEvents.push(ownUtteranceTurn(contentText));
    }

    return outputEvents;
  };
};

const main = async () => {
  console.log("Testing GLM-5.3-Flash agent with tool calling...");
  const history: HistoryEvent[] = [
    participantUtteranceTurn({
      name: "user",
      text:
        "What is the weather in Tokyo and in Paris? Call get_weather for both.",
    }),
  ];

  const weatherTool = tool({
    name: "get_weather",
    description: "Get current weather in a city",
    parameters: z.object({ city: z.string().describe("City name") }),
    handler: ({ city }: { city: string }) => {
      console.log(`[Tool Call executed: get_weather for ${city}]`);
      if (city.toLowerCase().includes("tokyo")) {
        return Promise.resolve("Tokyo: 18°C, sunny, wind 5km/h");
      }
      return Promise.resolve("Paris: 12°C, cloudy, light rain");
    },
  });

  const spec: AgentSpec = {
    provider: "moonshot",
    prompt: "You are a helpful assistant. Use tools when needed.",
    tools: [weatherTool],
    maxIterations: 5,
    timezoneIANA: "UTC",
  };

  const caller = createGlmCaller(spec);

  await injectAccessHistory(() => Promise.resolve(history))(
    injectOutputEvent((event) => {
      console.log(
        `[New Event: ${event.type}]`,
        event.type === "own_thought"
          ? "(thinking)"
          : (event as { text?: string; name?: string }).name ||
            (event as { text?: string }).text || "",
      );
      history.push(event);
      return Promise.resolve();
    })(
      injectCallModel(caller)(() => runAgent(spec)),
    ),
  )();

  console.log("\nAgent finished! Total history events:", history.length);
  for (const e of history) {
    if (e.type === "own_utterance") {
      console.log("\nFinal Answer:\n", e.text);
    }
  }
};

if (import.meta.main) {
  await main();
}
