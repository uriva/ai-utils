import { assertEquals } from "@std/assert";
import { z } from "zod/v4";
import { tool } from "../mod.ts";
import {
  buildLiveSetupMessage,
  createAudioSession,
  toolsToDeclarations,
} from "../src/geminiLiveSession.ts";
import { collectAllAudioTools } from "../src/audioTransportAgent.ts";

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

Deno.test(
  "collectAllAudioTools preserves composio, secrets_and_variables, agentdocs, and google-docs tools across large skill sets",
  () => {
    const dummyTool = (name: string) =>
      tool({
        name,
        description: `Tool ${name}`,
        parameters: z.object({}),
        handler: () => Promise.resolve("ok"),
      });
    const skills = [
      {
        name: "scheduling",
        description: "Scheduling",
        instructions: "",
        tools: Array.from({ length: 12 }, (_, i) => dummyTool(`sched_${i}`)),
      },
      {
        name: "google-docs",
        description: "Google Docs",
        instructions: "",
        tools: [
          ...Array.from({ length: 10 }, (_, i) => dummyTool(`sheet_${i}`)),
          dummyTool("fetchDocument"),
        ],
      },
      {
        name: "composio",
        description: "Composio",
        instructions: "",
        tools: [
          dummyTool("list_composio_tools"),
          dummyTool("inspect_composio_tool"),
          dummyTool("execute_composio_tool"),
        ],
      },
      {
        name: "secrets_and_variables",
        description: "Secrets",
        instructions: "",
        tools: [
          dummyTool("set_secret"),
          dummyTool("delete_secret"),
          dummyTool("list_env_variables"),
        ],
      },
      {
        name: "agentdocs",
        description: "AgentDocs",
        instructions: "",
        tools: [
          dummyTool("searchDocuments"),
          dummyTool("getDocument"),
          dummyTool("listDocuments"),
        ],
      },
    ];
    const collected = collectAllAudioTools(
      Array.from({ length: 8 }, (_, i) => dummyTool(`base_${i}`)),
      skills,
    );
    const names = new Set(collected.map((t) => t.name));
    assertEquals(names.has("list_composio_tools"), true);
    assertEquals(names.has("inspect_composio_tool"), true);
    assertEquals(names.has("execute_composio_tool"), true);
    assertEquals(names.has("list_env_variables"), true);
    assertEquals(names.has("searchDocuments"), true);
    assertEquals(names.has("getDocument"), true);
    assertEquals(names.has("fetchDocument"), true);
  },
);

Deno.test(
  "buildLiveSetupMessage compaction preserves tool parameters named 'description' so required[0] does not fail Gemini Live 1007 validation",
  () => {
    const taskTool = tool({
      name: "create_bot_task",
      description: "Create a scheduled task",
      parameters: z.object({
        description: z.string().describe("What the task should do"),
        preferredNetwork: z.string().optional().describe("Target network"),
      }),
      handler: () => Promise.resolve("ok"),
    });
    const decls = toolsToDeclarations([taskTool])[0]?.functionDeclarations;
    const setupJson = buildLiveSetupMessage({
      voiceName: "Charon",
      prompt: "x".repeat(70_000),
      declarations: decls,
    });
    const parsed = JSON.parse(setupJson);
    const compactedDecl = parsed.setup.tools[0].functionDeclarations[0];
    assertEquals(
      compactedDecl.parameters.required.includes("description"),
      true,
    );
    assertEquals(
      typeof compactedDecl.parameters.properties.description,
      "object",
    );
    assertEquals(
      compactedDecl.parameters.properties.description.description,
      undefined,
    );
  },
);
