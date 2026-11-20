import { assert, assertEquals } from "@std/assert";
import { pipe } from "gamla";
import { z } from "zod/v4";
import {
  type HistoryEvent,
  injectCallModel,
  ownUtteranceTurn,
  participantUtteranceTurn,
  runAgent,
  type Skill,
  tool,
  toolResultTurn,
  toolUseTurn,
} from "../mod.ts";
import {
  injectCallModelWrapper,
  learnSkillToolName,
  unlearnSkillToolName,
} from "../src/agent.ts";
import { agentDeps } from "../test_helpers.ts";

const makeLongToolDescription = (summary: string): string =>
  `${summary} ${
    "Extended usage notes covering edge cases, retries, auth headers, and formatting details. "
      .repeat(14)
  }`;

const unlearnedSkillName = (e: HistoryEvent): string | undefined => {
  if (e.type !== "tool_call" || e.name !== unlearnSkillToolName) {
    return undefined;
  }
  const params = e.parameters;
  if (
    typeof params === "object" &&
    params !== null &&
    "skillName" in params &&
    typeof params.skillName === "string"
  ) {
    return params.skillName;
  }
  return undefined;
};

Deno.test(
  "runAgent deduplicates duplicate skills and duplicate skill tools in both catalog and active skills",
  async () => {
    const toolA = tool({
      name: "list_items",
      description: "List all items in the catalog.",
      parameters: z.object({ limit: z.number().optional() }),
      handler: () => Promise.resolve("ok"),
    });
    const duplicateToolA = tool({
      name: "list_items",
      description: "Duplicate list_items tool.",
      parameters: z.object({ limit: z.number().optional() }),
      handler: () => Promise.resolve("ok"),
    });

    const primarySkill: Skill = {
      name: "inventory",
      description: "Manage catalog inventory items",
      instructions: "UNIQUE_INVENTORY_INSTRUCTIONS_MARKER",
      tools: [toolA, duplicateToolA],
    };
    const duplicateSkill: Skill = {
      name: "Inventory",
      description: "Duplicate inventory skill entry",
      instructions: "UNIQUE_INVENTORY_INSTRUCTIONS_MARKER",
      tools: [toolA],
    };

    const history: HistoryEvent[] = [
      participantUtteranceTurn({ name: "user", text: "Check inventory." }),
      {
        ...toolUseTurn({
          name: learnSkillToolName,
          args: { skillName: "inventory" },
        }),
        id: "learn-inv-1",
      },
      toolResultTurn({
        toolCallId: "learn-inv-1",
        result: "Skill learned.",
      }),
    ];

    const seenPrompts: string[] = [];

    await pipe(
      injectCallModel(() => Promise.resolve([ownUtteranceTurn("Checked.")])),
      injectCallModelWrapper(({ systemPrompt, inner }) => {
        seenPrompts.push(systemPrompt);
        return inner;
      }),
      agentDeps(history),
    )(() =>
      runAgent({
        maxIterations: 4,
        tools: [],
        skills: [primarySkill, duplicateSkill],
        prompt: "You are a helpful assistant.",
        timezoneIANA: "UTC",
      })
    )();

    const prompt = seenPrompts[0] ?? "";
    const catalogToolOccurrences =
      prompt.split("inventory/list_items(params:").length - 1;
    assertEquals(
      catalogToolOccurrences,
      1,
      `Expected inventory/list_items once in catalog, found ${catalogToolOccurrences}`,
    );

    const activeInstructionsOccurrences =
      prompt.split("UNIQUE_INVENTORY_INSTRUCTIONS_MARKER").length - 1;
    assertEquals(
      activeInstructionsOccurrences,
      1,
      `Expected active skill instructions once, found ${activeInstructionsOccurrences}`,
    );
  },
);

Deno.test(
  "runAgent keeps inactive skill catalog compact while preserving parameter signatures and full active docs",
  async () => {
    const summarySentence =
      "Create a recurring billing report for a workspace.";
    const longEssay = makeLongToolDescription(summarySentence);
    assert(longEssay.length > 1_000);

    const billingSkill: Skill = {
      name: "billing",
      description: "Workspace billing and invoicing tools",
      instructions: "Use billing tools to generate workspace reports.",
      tools: [
        tool({
          name: "create_report",
          description: longEssay,
          parameters: z.object({
            workspaceId: z.string().describe("Target workspace identifier"),
            includeTaxes: z.boolean().optional(),
          }),
          handler: () => Promise.resolve("report created"),
        }),
      ],
    };

    const inactiveHistory: HistoryEvent[] = [
      participantUtteranceTurn({ name: "user", text: "Hello!" }),
    ];

    const inactivePrompts: string[] = [];
    await pipe(
      injectCallModel(() => Promise.resolve([ownUtteranceTurn("Hi there!")])),
      injectCallModelWrapper(({ systemPrompt, inner }) => {
        inactivePrompts.push(systemPrompt);
        return inner;
      }),
      agentDeps(inactiveHistory),
    )(() =>
      runAgent({
        maxIterations: 4,
        tools: [],
        skills: [billingSkill],
        prompt: "You are an assistant.",
        timezoneIANA: "UTC",
      })
    )();

    const inactivePrompt = inactivePrompts[0] ?? "";
    assert(
      inactivePrompt.includes(
        "billing/create_report(params: { workspaceId: string, includeTaxes?: boolean })",
      ),
      "Inactive catalog must preserve compact parameter signature for direct run_command",
    );
    assert(
      inactivePrompt.includes(summarySentence),
      "Inactive catalog must preserve the concise first-sentence summary",
    );
    assertEquals(
      inactivePrompt.includes(
        "Extended usage notes covering edge cases, retries, auth headers",
      ),
      false,
      "Inactive catalog must not dump verbose multi-sentence tool essays",
    );

    // Once active, full tool description must be present in the system prompt
    const activeHistory: HistoryEvent[] = [
      participantUtteranceTurn({ name: "user", text: "Generate report." }),
      {
        ...toolUseTurn({
          name: learnSkillToolName,
          args: { skillName: "billing" },
        }),
        id: "learn-billing-1",
      },
      toolResultTurn({
        toolCallId: "learn-billing-1",
        result: "Skill learned.",
      }),
    ];
    const activePrompts: string[] = [];
    await pipe(
      injectCallModel(() =>
        Promise.resolve([ownUtteranceTurn("Report ready.")])
      ),
      injectCallModelWrapper(({ systemPrompt, inner }) => {
        activePrompts.push(systemPrompt);
        return inner;
      }),
      agentDeps(activeHistory),
    )(() =>
      runAgent({
        maxIterations: 4,
        tools: [],
        skills: [billingSkill],
        prompt: "You are an assistant.",
        timezoneIANA: "UTC",
      })
    )();

    assert(
      (activePrompts[0] ?? "").includes(longEssay),
      "Active skill section must retain the full tool description",
    );
  },
);

Deno.test(
  "runAgent deterministically auto-unlearns idle skills from earlier sessions at turn exit even when another workflow tool runs",
  async () => {
    const staleSkill: Skill = {
      name: "legacy_analytics",
      description: "Query historical analytics exports",
      instructions: "STALE_ANALYTICS_INSTRUCTIONS_MARKER",
      tools: [
        tool({
          name: "export_csv",
          description: "Export analytics table to CSV.",
          parameters: z.object({ table: z.string() }),
          handler: () => Promise.resolve("exported"),
        }),
      ],
    };

    const pingTool = tool({
      name: "check_status",
      description: "Check live service status.",
      parameters: z.object({}),
      handler: () => Promise.resolve("healthy"),
    });

    const oldSessionTime = Date.now() - 45 * 60 * 1000;
    const now = Date.now();
    const history: HistoryEvent[] = [
      {
        id: "old-user-1",
        type: "participant_utterance",
        isOwn: false,
        name: "user",
        text: "Export last quarter analytics.",
        timestamp: oldSessionTime,
      },
      {
        id: "learn-legacy-1",
        type: "tool_call",
        isOwn: true,
        name: learnSkillToolName,
        parameters: { skillName: "legacy_analytics" },
        timestamp: oldSessionTime + 1_000,
      },
      {
        id: "learn-legacy-1-res",
        type: "tool_result",
        isOwn: true,
        toolCallId: "learn-legacy-1",
        result: "Skill learned.",
        timestamp: oldSessionTime + 1_100,
      },
      {
        id: "old-bot-1",
        type: "own_utterance",
        isOwn: true,
        text: "Analytics exported.",
        timestamp: oldSessionTime + 2_000,
      },
      {
        id: "new-user-1",
        type: "participant_utterance",
        isOwn: false,
        name: "user",
        text: "Is the service up right now?",
        timestamp: now,
      },
    ];

    const state = { step: 0 };
    await pipe(
      injectCallModel(() => {
        state.step++;
        return state.step === 1
          ? Promise.resolve([toolUseTurn({ name: "check_status", args: {} })])
          : Promise.resolve([ownUtteranceTurn("Yes, the service is healthy.")]);
      }),
      agentDeps(history),
    )(() =>
      runAgent({
        maxIterations: 5,
        tools: [pingTool],
        skills: [staleSkill],
        prompt: "You are an assistant.",
        timezoneIANA: "UTC",
      })
    )();

    const unlearnedNames = history
      .map(unlearnedSkillName)
      .filter((name): name is string => name !== undefined);
    assertEquals(
      unlearnedNames,
      ["legacy_analytics"],
      `Expected idle skill from 45m-old session to be auto-unlearned at turn exit, got ${
        JSON.stringify(unlearnedNames)
      }`,
    );

    // Next turn must no longer carry stale skill instructions
    history.push({
      id: "new-user-2",
      type: "participant_utterance",
      isOwn: false,
      name: "user",
      text: "Great, thanks!",
      timestamp: now + 10_000,
    });
    const nextPrompts: string[] = [];
    await pipe(
      injectCallModel(() => Promise.resolve([ownUtteranceTurn("Anytime!")])),
      injectCallModelWrapper(({ systemPrompt, inner }) => {
        nextPrompts.push(systemPrompt);
        return inner;
      }),
      agentDeps(history),
    )(() =>
      runAgent({
        maxIterations: 4,
        tools: [pingTool],
        skills: [staleSkill],
        prompt: "You are an assistant.",
        timezoneIANA: "UTC",
      })
    )();

    assertEquals(
      (nextPrompts[0] ?? "").includes("STALE_ANALYTICS_INSTRUCTIONS_MARKER"),
      false,
      "Unlearned idle skill instructions must be absent on the following turn",
    );
  },
);

Deno.test(
  "runAgent auto-unlearns skills idle for 4+ user turns while keeping recently used skills active",
  async () => {
    const oldTurnSkill: Skill = {
      name: "alpha_tools",
      description: "Alpha helper tools",
      instructions: "ALPHA_INSTRUCTIONS",
      tools: [
        tool({
          name: "alpha_op",
          description: "Run alpha operation.",
          parameters: z.object({}),
          handler: () => Promise.resolve("alpha"),
        }),
      ],
    };
    const recentSkill: Skill = {
      name: "beta_tools",
      description: "Beta helper tools",
      instructions: "BETA_INSTRUCTIONS",
      tools: [
        tool({
          name: "beta_op",
          description: "Run beta operation.",
          parameters: z.object({}),
          handler: () => Promise.resolve("beta"),
        }),
      ],
    };

    const base = Date.now() - 5 * 60 * 1000;
    const history: HistoryEvent[] = [
      // Turn 1: learn alpha_tools
      {
        id: "u1",
        type: "participant_utterance",
        isOwn: false,
        name: "user",
        text: "Run alpha.",
        timestamp: base,
      },
      {
        id: "learn-alpha",
        type: "tool_call",
        isOwn: true,
        name: learnSkillToolName,
        parameters: { skillName: "alpha_tools" },
        timestamp: base + 1_000,
      },
      {
        id: "learn-alpha-res",
        type: "tool_result",
        isOwn: true,
        toolCallId: "learn-alpha",
        result: "Skill learned.",
        timestamp: base + 1_100,
      },
      {
        id: "a1",
        type: "own_utterance",
        isOwn: true,
        text: "Done alpha.",
        timestamp: base + 2_000,
      },
      // Turn 2 (idle for alpha: 1)
      {
        id: "u2",
        type: "participant_utterance",
        isOwn: false,
        name: "user",
        text: "Question 2",
        timestamp: base + 60_000,
      },
      {
        id: "a2",
        type: "own_utterance",
        isOwn: true,
        text: "Answer 2",
        timestamp: base + 61_000,
      },
      // Turn 3 (idle for alpha: 2)
      {
        id: "u3",
        type: "participant_utterance",
        isOwn: false,
        name: "user",
        text: "Question 3",
        timestamp: base + 120_000,
      },
      {
        id: "a3",
        type: "own_utterance",
        isOwn: true,
        text: "Answer 3",
        timestamp: base + 121_000,
      },
      // Turn 4 (idle for alpha: 3; learn beta_tools)
      {
        id: "u4",
        type: "participant_utterance",
        isOwn: false,
        name: "user",
        text: "Run beta.",
        timestamp: base + 180_000,
      },
      {
        id: "learn-beta",
        type: "tool_call",
        isOwn: true,
        name: learnSkillToolName,
        parameters: { skillName: "beta_tools" },
        timestamp: base + 181_000,
      },
      {
        id: "learn-beta-res",
        type: "tool_result",
        isOwn: true,
        toolCallId: "learn-beta",
        result: "Skill learned.",
        timestamp: base + 181_100,
      },
      {
        id: "a4",
        type: "own_utterance",
        isOwn: true,
        text: "Done beta.",
        timestamp: base + 182_000,
      },
      // Turn 5 (idle for alpha: 4; idle for beta: 1)
      {
        id: "u5",
        type: "participant_utterance",
        isOwn: false,
        name: "user",
        text: "One more general question.",
        timestamp: base + 240_000,
      },
    ];

    await pipe(
      injectCallModel(() => Promise.resolve([ownUtteranceTurn("Answered.")])),
      agentDeps(history),
    )(() =>
      runAgent({
        maxIterations: 4,
        tools: [],
        skills: [oldTurnSkill, recentSkill],
        prompt: "You are an assistant.",
        timezoneIANA: "UTC",
      })
    )();

    const unlearnedNames = history
      .map(unlearnedSkillName)
      .filter((name): name is string => name !== undefined);

    assertEquals(
      unlearnedNames,
      ["alpha_tools"],
      `Expected only alpha_tools (4 idle turns) to be unlearned and beta_tools (1 idle turn) to remain active, got ${
        JSON.stringify(unlearnedNames)
      }`,
    );
  },
);
