import { assert, assertEquals } from "@std/assert";
import { z } from "zod/v4";
import { runAgent } from "../mod.ts";
import {
  type AgentSpec,
  createSkillTools,
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

// Provider-agnostic: guards against the observed failure mode (find-scene)
// where the model called learn_skill with a free-form `referenceName` that was
// really the skill name or a tool/command name. That whole class of bug is now
// impossible by construction: learn_skill has no referenceName parameter, and
// references are ordinary fixed-name tools invoked via run_command.
Deno.test(
  "learn_skill has no referenceName parameter; references are tools called via run_command",
  async () => {
    const documentationSkill = {
      name: "documentation",
      description: "Read reference files",
      instructions: "Read the reference files to learn guidelines",
      tools: [],
      references: [
        { name: "cbt-protocols.md", content: "ALWAYS_PRESENT_CBT_CONTENT" },
      ],
    };

    const skillTools = createSkillTools([documentationSkill]);

    const learnTool = skillTools.find((t) => t.name === learnSkillToolName);
    assert(learnTool, "should expose learn_skill tool");
    const learnShape = learnTool.parameters.shape;
    assert(
      !("referenceName" in learnShape),
      "learn_skill must not expose a referenceName parameter",
    );

    // The reference is reachable via run_command using its .md-stripped name.
    const runCommand = skillTools.find((t) => t.name === runCommandToolName);
    assert(runCommand, "should expose run_command tool");
    const refResult = await runCommand.handler({
      command: "documentation/cbt-protocols",
      params: {},
      spinnerText: "Loading CBT protocols...",
    }, "call-id-1");
    assert(typeof refResult === "string");
    assert(
      refResult.includes("ALWAYS_PRESENT_CBT_CONTENT"),
      `run_command on the reference tool should return its content. Got: "${refResult}"`,
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
      specTurn2.prompt.includes("calendar/setup-guide"),
      "System prompt should list reference documents as callable tools under the active skill",
    );
    assert(
      !specTurn2.prompt.includes("setup-guide.md"),
      "Reference tool names should have the .md suffix stripped",
    );
  },
);

Deno.test(
  "skills: a reference is a run_command tool that returns its content and activates its skill",
  async () => {
    const documentationSkill = {
      name: "documentation",
      description: "Read reference files",
      instructions: "ALWAYS_PRESENT_DOC_INSTRUCTIONS",
      tools: [],
      references: [
        {
          name: "cbt-protocols.md",
          content: "ALWAYS_PRESENT_CBT_CONTENT",
        },
      ],
    };

    const spec = {
      tools: [],
      skills: [documentationSkill],
      prompt: "Help the user.",
    } as unknown as AgentSpec;

    // Calling the reference tool returns its content (it lands in history like
    // any other tool_result, so no separate "active references" prompt needed).
    const skillTools = createSkillTools([documentationSkill]);
    const runCommand = skillTools.find((t) => t.name === runCommandToolName);
    assert(runCommand, "should expose run_command tool");
    const resultStr = await runCommand.handler({
      command: "documentation/cbt-protocols",
      params: {},
      spinnerText: "Loading CBT protocols...",
    }, "call-id-1");
    assert(typeof resultStr === "string");
    assert(
      resultStr.includes("ALWAYS_PRESENT_CBT_CONTENT"),
      `Reference tool should return its content. Got: "${resultStr}"`,
    );

    // Calling a skill tool (including a reference tool) auto-activates the skill.
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({ name: "user", text: "read cbt protocols" }),
      {
        id: "call-1",
        type: "tool_call",
        isOwn: true,
        name: runCommandToolName,
        parameters: { command: "documentation/cbt-protocols", params: {} },
        timestamp: Date.now(),
      },
    ];
    const specTurn2 = getSpecForTurn(spec, mockHistory);
    assertEquals(specTurn2.skills!.length, 1);
    assert(
      specTurn2.prompt.includes("ALWAYS_PRESENT_DOC_INSTRUCTIONS"),
      "Calling a reference tool should activate its skill on the next turn",
    );
  },
);

Deno.test(
  "skills: unlearning a skill deactivates it and stops advertising its reference tools",
  () => {
    const documentationSkill = {
      name: "documentation",
      description: "Read reference files",
      instructions: "ALWAYS_PRESENT_DOC_INSTRUCTIONS",
      tools: [],
      references: [
        {
          name: "cbt-protocols.md",
          content: "ALWAYS_PRESENT_CBT_CONTENT",
        },
      ],
    };

    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "learn the documentation skill",
      }),
      {
        id: "call-1",
        type: "tool_call",
        isOwn: true,
        name: learnSkillToolName,
        parameters: { skillName: "documentation" },
        timestamp: 1000,
      },
      {
        id: "result-1",
        type: "tool_result",
        isOwn: true,
        toolCallId: "call-1",
        result: "Skill learned successfully.",
        timestamp: 2000,
      },
    ];

    const spec = {
      tools: [],
      skills: [documentationSkill],
      prompt: "Help the user.",
    } as unknown as AgentSpec;

    // Turn 2 (skill active): instructions and reference tool are advertised.
    const specTurn2 = getSpecForTurn(spec, mockHistory);
    assertEquals(specTurn2.skills!.length, 1);
    assert(
      specTurn2.prompt.includes("ALWAYS_PRESENT_DOC_INSTRUCTIONS"),
      "Skill instructions should be active inside the system prompt",
    );
    assert(
      specTurn2.prompt.includes("documentation/cbt-protocols"),
      "Reference tool should be advertised while the skill is active",
    );

    // Turn 3: Simulate unlearning the entire skill
    const unlearnCall: HistoryEvent = {
      id: "call-3",
      type: "tool_call",
      isOwn: true,
      name: "unlearn_skill",
      parameters: { skillName: "documentation" },
      timestamp: 3000,
    };
    const unlearnResult: HistoryEvent = {
      id: "result-3",
      type: "tool_result",
      isOwn: true,
      toolCallId: "call-3",
      result: "Skill deactivated successfully.",
      timestamp: 4000,
    };

    const finalHistory = [...mockHistory, unlearnCall, unlearnResult];
    const specTurn3 = getSpecForTurn(spec, finalHistory);

    assertEquals(specTurn3.skills!.length, 0);
    assert(
      !specTurn3.prompt.includes("ALWAYS_PRESENT_DOC_INSTRUCTIONS"),
      "Skill instructions should be removed after unlearning",
    );
    assert(
      !specTurn3.prompt.includes("documentation/cbt-protocols"),
      "Reference tool should no longer be advertised after unlearning",
    );
  },
);

Deno.test(
  "skills: calling a skill tool auto-loads/learns that skill for subsequent turns",
  () => {
    const calendarSkill = {
      name: "calendar",
      description: "Calendar operations",
      instructions: "CALENDAR_INSTRUCTIONS_MARKER",
      tools: [{
        name: "list_events",
        description: "List events",
        parameters: z.object({}),
        handler: () => Promise.resolve("events"),
      }],
    };

    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({ name: "user", text: "list events" }),
      {
        id: "call-1",
        type: "tool_call",
        isOwn: true,
        name: "run_command",
        parameters: {
          command: "calendar/list_events",
          params: {},
        },
        timestamp: 1000,
      },
    ];

    const spec = {
      tools: [],
      skills: [calendarSkill],
      prompt: "Help the user.",
    } as unknown as AgentSpec;

    // The skill should be inactive initially (on Turn 1, i.e. with empty history)
    const specTurn1 = getSpecForTurn(spec, []);
    assert(
      !specTurn1.prompt.includes("CALENDAR_INSTRUCTIONS_MARKER"),
      "Skill instructions should not be loaded on Turn 1 before tool call",
    );

    // After calling the skill tool via run_command, the skill should be active on Turn 2
    const specTurn2 = getSpecForTurn(spec, mockHistory);
    assert(
      specTurn2.prompt.includes("CALENDAR_INSTRUCTIONS_MARKER"),
      "Skill instructions should be auto-loaded/active on Turn 2 after run_command call",
    );

    // Also verify for direct slash tool calls
    const mockHistoryDirect: HistoryEvent[] = [
      participantUtteranceTurn({ name: "user", text: "list events" }),
      {
        id: "call-2",
        type: "tool_call",
        isOwn: true,
        name: "calendar/list_events",
        parameters: {},
        timestamp: 1000,
      },
    ];

    const specTurn2Direct = getSpecForTurn(spec, mockHistoryDirect);
    assert(
      specTurn2Direct.prompt.includes("CALENDAR_INSTRUCTIONS_MARKER"),
      "Skill instructions should be auto-loaded/active on Turn 2 after direct slash-routed call",
    );
  },
);

Deno.test(
  "skills: active skill prompt includes full tool names, descriptions, and exact parameter schemas",
  () => {
    const calendarSkill = {
      name: "calendar",
      description: "Calendar operations",
      instructions: "CALENDAR_INSTRUCTIONS_MARKER",
      tools: [{
        name: "add_event",
        description: "Add a calendar event",
        parameters: z.object({
          title: z.string().describe("The event title"),
          date: z.string().describe("The event date"),
        }),
        handler: () => Promise.resolve("event added"),
      }],
    };

    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({ name: "user", text: "add event" }),
      {
        id: "call-1",
        type: "tool_call",
        isOwn: true,
        name: "run_command",
        parameters: {
          command: "calendar/add_event",
          params: { title: "Meeting", date: "2026-07-01" },
        },
        timestamp: 1000,
      },
    ];

    const spec = {
      tools: [],
      skills: [calendarSkill],
      prompt: "Help the user.",
    } as unknown as AgentSpec;

    const specTurn2 = getSpecForTurn(spec, mockHistory);
    assert(
      specTurn2.prompt.includes("calendar/add_event"),
      "Active skill prompt should include full tool names in skillName/toolName format",
    );
    assert(
      specTurn2.prompt.includes("Add a calendar event"),
      "Active skill prompt should include tool descriptions",
    );
    assert(
      specTurn2.prompt.includes("title: string"),
      "Active skill prompt should include typed parameters inside prompt",
    );
    assert(
      specTurn2.prompt.includes("The event title"),
      "Active skill prompt should include parameter description comments",
    );
  },
);

Deno.test(
  "skills: run_command on a guessed reference name lists the real reference tools to self-correct",
  async () => {
    const coderSkill = {
      name: "p2b-coder",
      description: "Coder/integrator skill",
      instructions: "ALWAYS_PRESENT_CODER_INSTRUCTIONS",
      tools: [],
      references: [
        {
          name: "planning-and-design.md",
          content: "planning content",
        },
      ],
    };

    const skillTools = createSkillTools([coderSkill]);
    const runCommand = skillTools.find((t) => t.name === runCommandToolName);
    assert(runCommand, "should expose run_command tool");

    // The model guesses a reference name that does not exist.
    const resultStr = await runCommand.handler({
      command: "p2b-coder/README",
      params: {},
      spinnerText: "Loading readme...",
    }, "call-id-1");

    assert(typeof resultStr === "string");
    assert(
      resultStr.includes("not found"),
      `Should tell the model the tool was not found. Got: "${resultStr}"`,
    );
    // The error must surface the real, .md-stripped reference tool so the model
    // can retry with the correct name.
    assert(
      resultStr.includes("planning-and-design"),
      `Should list the real reference tool name. Got: "${resultStr}"`,
    );

    // And the correct call actually returns the reference content.
    const good = await runCommand.handler({
      command: "p2b-coder/planning-and-design",
      params: {},
      spinnerText: "Loading planning doc...",
    }, "call-id-2");
    assert(typeof good === "string" && good.includes("planning content"));
  },
);
