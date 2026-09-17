import { assertEquals } from "@std/assert";
import { z } from "zod/v4";
import { tool } from "../mod.ts";
import {
  createAudioSession,
  toolsToDeclarations,
} from "../src/geminiLiveSession.ts";

Deno.test(
  "toolsToDeclarations configures tools with NON_BLOCKING behavior for Gemini Live 3.8",
  () => {
    const testTool = tool({
      name: "get_weather",
      description: "Get weather for a given city",
      parameters: z.object({ city: z.string() }),
      handler: () => Promise.resolve("sunny"),
    });
    const declarations = toolsToDeclarations([testTool]);
    assertEquals(
      declarations[0]?.functionDeclarations[0]?.behavior,
      "NON_BLOCKING",
    );
  },
);

Deno.test("AudioSession exposes sendClientContent method", async () => {
  const apiKey = Deno.env.get("GEMINI_API_KEY");
  if (!apiKey) throw new Error("GEMINI_API_KEY is required");
  const session = await createAudioSession({
    apiKey,
    prompt: "You are a helpful assistant.",
    voiceName: "Aoede",
  });
  const hasSendClientContent = typeof session.sendClientContent === "function";
  await session.close();
  assertEquals(hasSendClientContent, true);
});

Deno.test(
  "AudioSession sendClientContent with turnComplete: false appends context without waiting",
  async () => {
    const apiKey = Deno.env.get("GEMINI_API_KEY");
    if (!apiKey) throw new Error("GEMINI_API_KEY is required");
    const session = await createAudioSession({
      apiKey,
      prompt: "You are a helpful assistant.",
      voiceName: "Aoede",
    });
    const events = await session.sendClientContent({
      turns: [
        {
          role: "user",
          parts: [{ text: "[System notification: User entered the room]" }],
        },
      ],
      turnComplete: false,
    });
    await session.close();
    assertEquals(events, []);
  },
);

Deno.test(
  "AudioSession sendClientContent with turnComplete: true triggers generation and returns events",
  async () => {
    const apiKey = Deno.env.get("GEMINI_API_KEY");
    if (!apiKey) throw new Error("GEMINI_API_KEY is required");
    const session = await createAudioSession({
      apiKey,
      prompt: "You are a helpful assistant. Reply with only one word: pong.",
      voiceName: "Aoede",
    });
    const events = await session.sendClientContent({
      turns: [
        {
          role: "user",
          parts: [{ text: "ping" }],
        },
      ],
      turnComplete: true,
    });
    await session.close();
    assertEquals(events.length > 0, true);
  },
);

Deno.test(
  "AudioSession handles tool call and responds with scheduling parameter",
  async () => {
    const apiKey = Deno.env.get("GEMINI_API_KEY");
    if (!apiKey) throw new Error("GEMINI_API_KEY is required");
    const testTool = tool({
      name: "fetch_city_info",
      description: "Fetches information about a city",
      parameters: z.object({ city: z.string() }),
      handler: () => Promise.resolve("Sunny, 25C"),
    });
    const session = await createAudioSession({
      apiKey,
      prompt:
        "You are an assistant. When the user asks about a city, call the fetch_city_info tool.",
      voiceName: "Aoede",
      tools: [testTool],
    });
    const events = await session.sendClientContent({
      turns: [
        {
          role: "user",
          parts: [
            { text: "What is the weather in Paris? Call fetch_city_info." },
          ],
        },
      ],
      turnComplete: true,
    });
    const toolCall = events.find((e) => e.type === "tool_call");
    assertEquals(toolCall?.type, "tool_call");
    if (toolCall && toolCall.type === "tool_call") {
      session.respondToToolCall({
        id: toolCall.id,
        name: toolCall.name,
        response: { weather: "Sunny, 25C" },
        scheduling: "WHEN_IDLE",
      });
    }
    await session.close();
  },
);
