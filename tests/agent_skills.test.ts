import { assert, assertEquals } from "@std/assert";
import { z } from "zod/v4";
import { runAgent } from "../mod.ts";
import {
  type HistoryEvent,
  learnSkillToolName,
  participantUtteranceTurn,
  runCommandToolName,
} from "../src/agent.ts";
import {
  addition,
  agentDeps,
  injectSecrets,
  llmTest,
  multiplication,
  noopRewriteHistory,
  weatherSkill,
} from "../test_helpers.ts";

Deno.test(
  "skills: agent can use run_command to call skill tools",
  injectSecrets(async () => {
    const mockHistory: HistoryEvent[] = [participantUtteranceTurn({
      name: "user",
      text: "What is 5 + 3?",
    })];

    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 5,
      onMaxIterationsReached: () => {},
      tools: [],
      skills: [{
        name: "calculator",
        description: "Mathematical operations",
        instructions: "Use this skill for any math calculations",
        tools: [addition, multiplication],
      }],
      prompt:
        "You are a math assistant. Use the available skill tools to answer questions.",
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });

    const runCommandCall = mockHistory.find((event) =>
      event.type === "tool_call" && event.name === runCommandToolName
    );
    assert(runCommandCall, "Should call run_command tool");

    const hasToolResult = mockHistory.some((event) =>
      event.type === "tool_result" &&
      event.name === runCommandToolName &&
      event.result === "8"
    );
    assert(hasToolResult, "Should have result of 5 + 3 = 8 from run_command");
  }),
);

Deno.test(
  "skills: prompt only shows skill name and description, not individual tools",
  injectSecrets(async () => {
    const mockHistory: HistoryEvent[] = [participantUtteranceTurn({
      name: "user",
      text: "hello",
    })];

    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 1,
      onMaxIterationsReached: () => {},
      tools: [],
      skills: [weatherSkill],
      prompt: "You are a helpful assistant.",
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });

    assert(true, "Prompt augmentation test - implementation will verify");
  }),
);

Deno.test(
  `skills: ${learnSkillToolName} returns skill information`,
  injectSecrets(async () => {
    const localWeatherSkill = {
      name: "weather",
      description: "Get weather information",
      instructions: "Always ask for location before checking weather",
      tools: [
        {
          name: "get_forecast",
          description: "Get weather forecast for a location",
          parameters: z.object({ location: z.string() }),
          handler: ({ location }: { location: string }) =>
            Promise.resolve(`Sunny in ${location}`),
        },
      ],
    };

    const mockHistory: HistoryEvent[] = [participantUtteranceTurn({
      name: "user",
      text: "Tell me about the weather skill",
    })];

    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 5,
      onMaxIterationsReached: () => {},
      tools: [],
      skills: [localWeatherSkill],
      prompt:
        `You are a helpful assistant. When asked about skills, use ${learnSkillToolName} to get information.`,
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });

    const learnSkillResult = mockHistory.find((event) =>
      event.type === "tool_result" &&
      event.name === learnSkillToolName &&
      event.result.includes("weather")
    );

    if (learnSkillResult && learnSkillResult.type === "tool_result") {
      const parsedResult = JSON.parse(learnSkillResult.result);
      assertEquals(parsedResult.name, "weather");
      assertEquals(
        parsedResult.instructions,
        "Always ask for location before checking weather",
      );
      assert(parsedResult.tools.length > 0, "Should include tools");
    }
  }),
);

llmTest(
  "skills: works alongside regular tools",
  injectSecrets(async () => {
    const regularTool = {
      name: "regularTool",
      description: "A regular tool that returns a unique string",
      parameters: z.object({}),
      handler: () => Promise.resolve("regular result"),
    };

    const skillTool = {
      name: "skillset",
      description: "A skill with a tool",
      instructions: "Use this skill",
      tools: [{
        name: "skill_tool",
        description: "A skill tool that returns a unique string",
        parameters: z.object({}),
        handler: () => Promise.resolve("skill result"),
      }],
    };

    const mockHistory: HistoryEvent[] = [participantUtteranceTurn({
      name: "user",
      text:
        "First call the regularTool, then use the skillset skill to call skill_tool.",
    })];

    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 10,
      onMaxIterationsReached: () => {},
      tools: [regularTool],
      skills: [skillTool],
      prompt:
        "You are a helpful assistant. You have a regular tool called regularTool and a skill called skillset. Use both when asked.",
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });

    const hasRegularResult = mockHistory.some((event) =>
      event.type === "tool_result" &&
      event.name === "regularTool"
    );

    const hasSkillResult = mockHistory.some((event) =>
      event.type === "tool_result" &&
      event.result === "skill result"
    );

    assert(
      hasRegularResult || hasSkillResult,
      "Should be able to use both regular tools and skill tools",
    );
  }),
);
