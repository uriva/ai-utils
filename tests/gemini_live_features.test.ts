import { assertEquals } from "@std/assert";
import { z } from "zod/v4";
import { tool } from "../mod.ts";
import {
  buildLiveSetupMessage,
  createAudioSession,
  toolsToDeclarations,
} from "../src/geminiLiveSession.ts";
import {
  collectAllAudioTools,
  formatAudioSkillsPrompt,
  makeSessionEventHandler,
  sanitizeAudioSystemPrompt,
} from "../src/audioTransportAgent.ts";

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

Deno.test(
  "toolsToDeclarations rewrites schemaless record/object properties to JSON string for Gemini Live compatibility",
  () => {
    const execTool = tool({
      name: "execute_composio_tool",
      description: "Execute a Composio tool",
      parameters: z.object({
        tool_slug: z.string(),
        arguments: z.record(z.string(), z.unknown()).describe("Tool arguments"),
      }),
      handler: () => Promise.resolve("ok"),
    });
    const decl = toolsToDeclarations([execTool])[0]?.functionDeclarations[0];
    const props = decl?.parameters.properties as Record<
      string,
      { type: string; description?: string }
    >;
    assertEquals(props.tool_slug.type, "string");
    assertEquals(props.arguments.type, "string");
    assertEquals(
      props.arguments.description?.includes("JSON-encoded object string"),
      true,
    );
  },
);

Deno.test(
  "sanitizeAudioSystemPrompt removes '[no response]' instructions and makeSessionEventHandler flushes audio on 'no response' output transcript",
  () => {
    const rawPrompt =
      "You are Yaron.\nWhen you have nothing to say (e.g. the message is irrelevant), respond with exactly [no response] and nothing else.\nSimply stay silent with [no response].";
    const sanitized = sanitizeAudioSystemPrompt(rawPrompt);
    assertEquals(sanitized.toLowerCase().includes("no response"), false);

    let flushed = 0;
    const utterances: string[] = [];
    const handler = makeSessionEventHandler({
      onAudio: () => {},
      onUtterance: (t) => utterances.push(t),
      onFlush: () => {
        flushed++;
      },
      onTurnOutput: () => {},
    });
    handler.handle({
      type: "output_transcript",
      text: "no response",
      finished: true,
    });
    assertEquals(flushed, 1);
    assertEquals(utterances.length, 0);
  },
);

Deno.test(
  "formatAudioSkillsPrompt orders primary skills by primaryAudioSkillOrder so composio and gmail appear first",
  () => {
    const skills = [
      {
        name: "github@1.0.11",
        description: "GitHub",
        instructions: "GitHub instructions",
        tools: [],
      },
      {
        name: "composio",
        description: "Composio",
        instructions: "Composio instructions",
        tools: [],
      },
      {
        name: "gmail",
        description: "Gmail",
        instructions: "Gmail instructions",
        tools: [],
      },
    ];
    const formatted = formatAudioSkillsPrompt(skills);
    const composioIdx = formatted.indexOf("### Skill: composio");
    const gmailIdx = formatted.indexOf("### Skill: gmail");
    const githubIdx = formatted.indexOf("### Skill: github@1.0.11");
    assertEquals(
      composioIdx !== -1 && gmailIdx !== -1 && githubIdx !== -1,
      true,
    );
    assertEquals(composioIdx < gmailIdx && gmailIdx < githubIdx, true);
  },
);

Deno.test(
  "makeSessionEventHandler deduplicates cumulative output_transcript prefixes across flush boundaries",
  () => {
    const utterances: string[] = [];
    const { handle, flushPending } = makeSessionEventHandler({
      onAudio: () => {},
      onUtterance: (t) => utterances.push(t),
      onFlush: () => {},
      onTurnOutput: () => {},
    });
    handle({
      type: "output_transcript",
      text: "אני מבין, שלחתי",
      finished: false,
    });
    flushPending();
    handle({
      type: "output_transcript",
      text: "אני מבין, שלחתי לה את ההודעה",
      finished: true,
    });
    assertEquals(utterances, ["אני מבין, שלחתי", "לה את ההודעה"]);
  },
);
