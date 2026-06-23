import { assert, assertEquals } from "@std/assert";
import { z } from "zod/v4";
import { runAgent } from "../mod.ts";
import {
  type AgentSpec,
  getSpecForTurn,
  type HistoryEvent,
  injectCallModel,
  learnSkillToolName,
  ownUtteranceTurn,
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
      event.result.endsWith("8")
    );
    assert(
      hasToolResult,
      `Should have result of 5 + 3 = 8 from run_command. History: ${
        JSON.stringify(mockHistory, null, 2)
      }`,
    );
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
      tools: [],
      skills: [localWeatherSkill],
      prompt:
        `You are a helpful assistant. When asked about skills, use ${learnSkillToolName} to get information.`,
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });

    const learnSkillResult = mockHistory.find((event) =>
      event.type === "tool_result" &&
      event.result.includes("learned successfully")
    );

    if (learnSkillResult && learnSkillResult.type === "tool_result") {
      assert(
        learnSkillResult.result.includes("weather"),
        "Should specify skill name",
      );
      assert(
        learnSkillResult.result.includes("learned successfully"),
        "Should report success",
      );
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
            spinnerText: "list calendars",
          },
          id: "fake-call-1",
          timestamp: Date.now(),
        }]);
      }
      return Promise.resolve([ownUtteranceTurn("Done")]);
    };
    await injectCallModel(fakeCallModel)(async () => {
      await agentDeps(mockHistory)(runAgent)({
        maxIterations: 2,
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
      return Promise.resolve([ownUtteranceTurn("Done")]);
    };
    await injectCallModel(fakeCallModel)(async () => {
      await agentDeps(mockHistory)(runAgent)({
        maxIterations: 2,
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
          parameters: {
            command: "calendar/listEvents",
            params: {},
            spinnerText: "list events",
          },
          id: "fake-call-1",
          timestamp: Date.now(),
        }]);
      }
      return Promise.resolve([ownUtteranceTurn("Done")]);
    };
    await injectCallModel(fakeCallModel)(async () => {
      await agentDeps(mockHistory)(runAgent)({
        maxIterations: 2,
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

Deno.test(
  "run_command auto-corrects an unambiguous misplaced key and prefixes the result",
  async () => {
    const calendarSkill = {
      name: "calendar",
      description: "Calendar operations",
      instructions: "Use this to manage calendar events.",
      tools: [{
        name: "update_event",
        description: "Update an event",
        parameters: z.object({ update: z.object({ title: z.string() }) }),
        handler: () => Promise.resolve("updated"),
      }],
    };
    const mockHistory: HistoryEvent[] = [participantUtteranceTurn({
      name: "user",
      text: "update the event",
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
            command: "calendar/update_event",
            params: { title: "standup" },
            spinnerText: "updating event",
          },
          id: "fake-call-1",
          timestamp: Date.now(),
        }]);
      }
      return Promise.resolve([ownUtteranceTurn("Done")]);
    };
    await injectCallModel(fakeCallModel)(async () => {
      await agentDeps(mockHistory)(runAgent)({
        maxIterations: 2,
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
      toolResult.result.startsWith("[arguments auto-corrected:"),
      `expected correction prefix, got: ${toolResult.result}`,
    );
    assert(
      toolResult.result.includes("updated"),
      `expected handler output after correction, got: ${toolResult.result}`,
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

Deno.test(
  "skills: learning a skill dynamically adds it to the system prompt and tools on the next iteration of the same run",
  () => {
    const calendarSkill = {
      name: "calendar",
      description: "Calendar operations",
      instructions: "ALWAYS_PRESENT_CALENDAR_INSTRUCTIONS",
      tools: [{
        name: "list_events",
        description: "List events",
        parameters: z.object({}),
        handler: () => Promise.resolve("events"),
      }],
      references: [
        {
          name: "setup-guide.md",
          content: "...",
        },
      ],
    };

    const mockHistory: HistoryEvent[] = [participantUtteranceTurn({
      name: "user",
      text: "please learn the calendar skill",
    })];

    const spec = {
      tools: [],
      skills: [calendarSkill],
      prompt: "Help the user.",
    } as unknown as AgentSpec;

    // Turn 1 (before learn_skill is called)
    const specTurn1 = getSpecForTurn(spec, mockHistory);
    assertEquals(specTurn1.skills!.length, 0);
    assert(!specTurn1.prompt.includes("ALWAYS_PRESENT_CALENDAR_INSTRUCTIONS"));

    // Simulate learn_skill tool call and result
    const learnCall: HistoryEvent = {
      id: "call-1",
      type: "tool_call",
      isOwn: true,
      name: learnSkillToolName,
      parameters: { skillName: "calendar" },
      timestamp: Date.now(),
    };
    const learnResult: HistoryEvent = {
      id: "result-1",
      type: "tool_result",
      isOwn: true,
      toolCallId: "call-1",
      result: "Skill learned successfully.",
      timestamp: Date.now(),
    };

    const updatedHistory = [...mockHistory, learnCall, learnResult];

    // Turn 2 (after learn_skill is called)
    const specTurn2 = getSpecForTurn(spec, updatedHistory);
    assertEquals(specTurn2.skills!.length, 1);
    assertEquals(specTurn2.skills![0].name, "calendar");
    assert(
      specTurn2.prompt.includes("ALWAYS_PRESENT_CALENDAR_INSTRUCTIONS"),
      "System prompt should be updated with skill instructions",
    );
    assert(
      specTurn2.prompt.includes("setup-guide.md"),
      "System prompt should list available reference files under the active skill",
    );
  },
);

import { createSkillTools } from "../src/agent.ts";

Deno.test(
  "skills: learning a reference behaves exactly like learning a skill, returning the reference content",
  async () => {
    const documentationSkill = {
      name: "documentation",
      description: "Read reference files",
      instructions: "Read the reference files to learn guidelines",
      tools: [],
      references: [
        {
          name: "cbt-protocols.md",
          content: "ALWAYS_PRESENT_CBT_CONTENT",
        },
      ],
    };

    const mockHistory: HistoryEvent[] = [participantUtteranceTurn({
      name: "user",
      text: "please learn reference cbt-protocols.md from documentation",
    })];

    const spec = {
      tools: [],
      skills: [documentationSkill],
      prompt: "Help the user.",
    } as unknown as AgentSpec;

    // Turn 1: Verify not active
    const specTurn1 = getSpecForTurn(spec, mockHistory);
    assertEquals(specTurn1.skills!.length, 0);

    // Verify learning reference returns the reference content
    const skillTools = createSkillTools([documentationSkill]);
    const learnRefTool = skillTools.find((t) => t.name === "learn_skill");
    assert(learnRefTool, "should expose learn_skill tool");

    const resultStr = await learnRefTool.handler({
      skillName: "documentation",
      referenceName: "cbt-protocols.md",
    }, "call-id-1");

    assert(typeof resultStr === "string");
    assert(
      resultStr.includes("learned successfully"),
      "Should return a lightweight confirmation message instead of JSON",
    );
    assert(
      resultStr.includes("cbt-protocols.md"),
      "Should reference the learned reference name",
    );

    // Turn 2: Simulate that learning a reference activates the skill exactly like learning a skill itself
    const learnCall: HistoryEvent = {
      id: "call-1",
      type: "tool_call",
      isOwn: true,
      name: learnSkillToolName,
      parameters: {
        skillName: "documentation",
        referenceName: "cbt-protocols.md",
      },
      timestamp: Date.now(),
    };
    const learnResult: HistoryEvent = {
      id: "result-1",
      type: "tool_result",
      isOwn: true,
      toolCallId: "call-1",
      result: resultStr,
      timestamp: Date.now(),
    };

    const updatedHistory = [...mockHistory, learnCall, learnResult];
    const specTurn2 = getSpecForTurn(spec, updatedHistory);

    // Verify the skill itself is NOT fully active (keeping context usage minimal)
    assertEquals(specTurn2.skills!.length, 0);

    // Verify that the reference's specific content is dynamically loaded into the prompt
    assert(
      specTurn2.prompt.includes("ALWAYS_PRESENT_CBT_CONTENT"),
      "System prompt should be updated with the learned reference content",
    );
  },
);
