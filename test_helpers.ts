import { assert } from "@std/assert";
import type { Injector } from "@uri/inject";
import { pipe } from "gamla";
import { z } from "zod/v4";
import {
  injectCacher,
  injectGeminiToken,
  injectOpenAiToken,
  tool,
} from "./mod.ts";
import {
  type HistoryEvent,
  injectAccessHistory,
  injectOutputEvent,
  type ToolReturn,
} from "./src/agent.ts";

export const injectSecrets = pipe(
  // @ts-expect-error passthrough cacher is sufficient for tests
  injectCacher(() => (f) => f),
  injectOpenAiToken(
    Deno.env.get("OPENAI_API_KEY") ?? "",
  ),
  injectGeminiToken(Deno.env.get("GEMINI_API_KEY") ?? ""),
);

export const agentDeps = (inMemoryHistory: HistoryEvent[]): Injector =>
  pipe(
    injectAccessHistory(() => Promise.resolve(inMemoryHistory)),
    injectOutputEvent((event) => {
      inMemoryHistory.push(event);
      return Promise.resolve();
    }),
  );

export const noopRewriteHistory = async () => {};

export const toolResult = "43212e8e";

export const someTool = {
  name: "doSomethingUnique",
  description: "Returns a unique string so we know the tool was called.",
  parameters: z.object({}),
  handler: () => Promise.resolve(toolResult),
};

const toBase64 = (u8: Uint8Array): string => {
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < u8.length; i += chunk) {
    binary += String.fromCharCode(...u8.subarray(i, i + chunk));
  }
  return btoa(binary);
};

const bytes = await Deno.readFile("./dog.jpg");

export const b64 = toBase64(bytes);

export const mediaTool = {
  name: "returnMedia",
  description: "Returns media via attachments",
  parameters: z.object({}),
  handler: () => {
    const ret: ToolReturn = {
      result: "image attached",
      attachments: [
        { kind: "inline" as const, mimeType: "image/jpeg", dataBase64: b64 },
      ],
    };
    return Promise.resolve(ret);
  },
};

export const mediaToolWithCaption = {
  name: "returnMediaWithCaption",
  description: "Returns media with caption via attachments",
  parameters: z.object({}),
  handler: () => {
    const ret: ToolReturn = {
      result: "image with caption attached",
      attachments: [
        {
          kind: "inline" as const,
          mimeType: "image/jpeg",
          dataBase64: b64,
          caption: "A friendly golden retriever sitting in the grass",
        },
      ],
    };
    return Promise.resolve(ret);
  },
};

export const recognizedTheDog = (e: HistoryEvent) =>
  e.type === "own_utterance" &&
  (e.text.toLowerCase().includes("dog") ||
    e.text.toLowerCase().includes("retriever") ||
    e.text.toLowerCase().includes("puppy"));

export const findTextualAnswer = (events: HistoryEvent[]) =>
  events.find((event): event is Extract<HistoryEvent, {
    type: "own_utterance";
    text: string;
  }> =>
    event.type === "own_utterance" && typeof event.text === "string" &&
    event.text.length > 0
  );

export const collectAttachment = (events: HistoryEvent[], toolName?: string) => {
  for (let i = events.length - 1; i >= 0; i--) {
    const event = events[i];
    if (
      event.type === "tool_result" && event.attachments?.length &&
      (!toolName || event.name === toolName)
    ) {
      return event.attachments[0];
    }
    if (event.type === "own_utterance" && event.attachments?.length) {
      return event.attachments[0];
    }
  }
  return undefined;
};

export const addition = tool({
  name: "add",
  description: "Add two numbers",
  parameters: z.object({ a: z.number(), b: z.number() }),
  handler: ({ a, b }) => Promise.resolve(`${a + b}`),
});

export const multiplication = tool({
  name: "multiply",
  description: "Multiply two numbers",
  parameters: z.object({ x: z.number(), y: z.number() }),
  handler: ({ x, y }) => Promise.resolve(`${x * y}`),
});

export const weatherSkill = {
  name: "weather",
  description: "Get weather information",
  instructions: "Always ask for location before checking weather",
  tools: [
    tool({
      name: "get_forecast",
      description: "Get weather forecast for a location",
      parameters: z.object({ location: z.string() }),
      handler: ({ location }) => Promise.resolve(`Sunny in ${location}`),
    }),
    tool({
      name: "get_temperature",
      description: "Get current temperature",
      parameters: z.object({ location: z.string() }),
      handler: ({ location }) => Promise.resolve(`25°C in ${location}`),
    }),
  ],
};

export const withRetries = (
  maxAttempts: number,
  fn: () => Promise<void>,
) =>
async () => {
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      await fn();
      return;
    } catch (e) {
      if (attempt === maxAttempts) throw e;
      console.log(`Attempt ${attempt}/${maxAttempts} failed, retrying...`);
    }
  }
};

export const llmTest = (
  name: string,
  fn: () => Promise<void>,
  retries = 3,
) =>
  Deno.test(name, withRetries(retries, fn));
