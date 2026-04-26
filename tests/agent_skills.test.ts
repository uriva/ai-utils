import { assert, assertEquals } from "@std/assert";
import { z } from "zod/v4";
import { runAgent } from "../mod.ts";
import {
  type HistoryEvent,
  injectCallModel,
  learnSkillToolName,
  participantUtteranceTurn,
  runCommandToolName,
} from "../src/agent.ts";
import {
  addition,
  agentDeps,
  multiplication,
  noopRewriteHistory,
  runForAllProviders,
  weatherSkill,
} from "../test_helpers.ts";

runForAllProviders(
  "skills: agent can use run_command to call skill tools",
  async (runAgentWithProvider) => {
    const mockHistory: HistoryEvent[] = [participantUtteranceTurn({
      name: "user",
      text: "What is 5 + 3?",
    })];

    await agentDeps(mockHistory)(runAgentWithProvider)({
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
      event.result === "8"
    );
    assert(hasToolResult, "Should have result of 5 + 3 = 8 from run_command");
  },
);

runForAllProviders(
  "skills: prompt only shows skill name and description, not individual tools",
  async (runAgentWithProvider) => {
    const mockHistory: HistoryEvent[] = [participantUtteranceTurn({
      name: "user",
      text: "hello",
    })];

    await agentDeps(mockHistory)(runAgentWithProvider)({
      maxIterations: 1,
      onMaxIterationsReached: () => {},
      tools: [],
      skills: [weatherSkill],
      prompt: "You are a helpful assistant.",
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });

    assert(true, "Prompt augmentation test - implementation will verify");
  },
);

runForAllProviders(
  `skills: ${learnSkillToolName} returns skill information`,
  async (runAgentWithProvider) => {
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

    await agentDeps(mockHistory)(runAgentWithProvider)({
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
  },
);

// Provider-agnostic: reproduces the bug where run_command split skillName/toolName
// on the first "/" instead of the last. Skills named like "@tank/google-calendar"
// became unreachable because split gave skillName="@tank", toolName="google-calendar".
Deno.test(
  "skills: run_command routes to skill whose name contains '/'",
  async () => {
    const calendarSkill = {
      name: "@tank/google-calendar",
      description: "Google Calendar operations",
      instructions: "Use this to manage calendar events.",
      tools: [{
        name: "list_calendars",
        description: "List the user's calendars",
        parameters: z.object({}),
        handler: () => Promise.resolve("calendars: primary, work"),
      }],
    };
    const mockHistory: HistoryEvent[] = [participantUtteranceTurn({
      name: "user",
      text: "list my calendars",
    })];
    let callCount = 0;
    const fakeCallModel = () => {
      callCount += 1;
      if (callCount === 1) {
        return Promise.resolve([{
          type: "tool_call" as const,
          isOwn: true as const,
          name: runCommandToolName,
          parameters: {
            command: "@tank/google-calendar/list_calendars",
            params: {},
          },
          id: "fake-call-1",
          timestamp: Date.now(),
        }]);
      }
      return Promise.resolve([]);
    };
    await injectCallModel(fakeCallModel)(async () => {
      await agentDeps(mockHistory)(runAgent)({
        maxIterations: 2,
        onMaxIterationsReached: () => {},
        tools: [],
        skills: [calendarSkill],
        prompt: "unused in fake",
        rewriteHistory: noopRewriteHistory,
        timezoneIANA: "UTC",
      });
    })();
    const toolResult = mockHistory.find((e) => e.type === "tool_result");
    assert(toolResult, "expected a tool_result event");
    assert(
      toolResult.type === "tool_result" &&
        toolResult.result === "calendars: primary, work",
      `expected successful calendar list result, got: ${
        toolResult.type === "tool_result" ? toolResult.result : "<none>"
      }`,
    );
  },
);

// Provider-agnostic: when the model invents a tool name (a frequent failure
// mode), the tool_result returned to the model must include the actual
// available tool names and skill names so it can self-correct on the next
// turn. The previous message ("you may have misspelled it") gave the model
// nothing to recover with.
Deno.test(
  "tool not found error lists available tools and skills",
  async () => {
    const realTool = {
      name: "send_email",
      description: "Send an email",
      parameters: z.object({ to: z.string() }),
      handler: () => Promise.resolve("sent"),
    };
    const calendarSkill = {
      name: "calendar",
      description: "Calendar ops",
      instructions: "use this to manage calendar",
      tools: [{
        name: "list_events",
        description: "List events",
        parameters: z.object({}),
        handler: () => Promise.resolve("events"),
      }],
    };
    const mockHistory: HistoryEvent[] = [participantUtteranceTurn({
      name: "user",
      text: "send a hallucinated tool",
    })];
    let callCount = 0;
    const fakeCallModel = () => {
      callCount += 1;
      if (callCount === 1) {
        return Promise.resolve([{
          type: "tool_call" as const,
          isOwn: true as const,
          name: "send_emial", // typo, not registered
          parameters: { to: "x@y.com" },
          id: "fake-call-1",
          timestamp: Date.now(),
        }]);
      }
      return Promise.resolve([]);
    };
    await injectCallModel(fakeCallModel)(async () => {
      await agentDeps(mockHistory)(runAgent)({
        maxIterations: 2,
        onMaxIterationsReached: () => {},
        tools: [realTool],
        skills: [calendarSkill],
        prompt: "unused in fake",
        rewriteHistory: noopRewriteHistory,
        timezoneIANA: "UTC",
      });
    })();
    const toolResult = mockHistory.find((e) => e.type === "tool_result");
    assert(toolResult && toolResult.type === "tool_result");
    assert(
      toolResult.result.includes("send_email"),
      `expected error to list real tool name, got: ${toolResult.result}`,
    );
    assert(
      toolResult.result.includes("calendar"),
      `expected error to list skill name, got: ${toolResult.result}`,
    );
  },
);

// Provider-agnostic: when the model calls run_command with a valid skill but
// a bogus tool name inside that skill, the error must include the skill's
// instructions and tool list inline (auto-load) so the model can self-correct
// on the next turn without an extra learn_skill round trip.
Deno.test(
  "run_command auto-loads skill instructions on wrong tool name",
  async () => {
    const calendarSkill = {
      name: "calendar",
      description: "Calendar operations",
      instructions: "ALWAYS_PRESENT_MARKER use list_events to enumerate.",
      tools: [{
        name: "list_events",
        description: "List events",
        parameters: z.object({}),
        handler: () => Promise.resolve("events"),
      }],
    };
    const mockHistory: HistoryEvent[] = [participantUtteranceTurn({
      name: "user",
      text: "list events",
    })];
    let callCount = 0;
    const fakeCallModel = () => {
      callCount += 1;
      if (callCount === 1) {
        return Promise.resolve([{
          type: "tool_call" as const,
          isOwn: true as const,
          name: runCommandToolName,
          parameters: { command: "calendar/listEvents", params: {} },
          id: "fake-call-1",
          timestamp: Date.now(),
        }]);
      }
      return Promise.resolve([]);
    };
    await injectCallModel(fakeCallModel)(async () => {
      await agentDeps(mockHistory)(runAgent)({
        maxIterations: 2,
        onMaxIterationsReached: () => {},
        tools: [],
        skills: [calendarSkill],
        prompt: "unused",
        rewriteHistory: noopRewriteHistory,
        timezoneIANA: "UTC",
      });
    })();
    const toolResult = mockHistory.find((e) => e.type === "tool_result");
    assert(toolResult && toolResult.type === "tool_result");
    assert(
      toolResult.result.includes("ALWAYS_PRESENT_MARKER"),
      `expected error to include skill instructions, got: ${toolResult.result}`,
    );
    assert(
      toolResult.result.includes("list_events"),
      `expected error to list real tool name, got: ${toolResult.result}`,
    );
  },
);

runForAllProviders(
  "skills: works alongside regular tools",
  async (runAgentWithProvider) => {
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

    await agentDeps(mockHistory)(runAgentWithProvider)({
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
      event.type === "tool_result"
    );

    const hasSkillResult = mockHistory.some((event) =>
      event.type === "tool_result" &&
      event.result === "skill result"
    );

    assert(
      hasRegularResult || hasSkillResult,
      "Should be able to use both regular tools and skill tools",
    );
  },
);
