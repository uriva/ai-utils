import { assert, assertEquals } from "@std/assert";
import {
  getSpecForTurn,
  type HistoryEvent,
  learnSkillToolName,
  participantUtteranceTurn,
  runCommandToolName,
} from "../src/agent.ts";
import {
  addition,
  agentDeps,
  multiplication,
  runForAllProviders,
} from "../test_helpers.ts";

const calculatorSkill = {
  name: "calculator",
  description: "Mathematical operations for arithmetic and formulas",
  instructions:
    "You are a math assistant. When asked to calculate, use calculator/add or calculator/multiply.",
  tools: [addition, multiplication],
};

runForAllProviders(
  "decision model skill activation: agent invokes skill tool directly on turn 1 without wasting a turn on learn_skill",
  async (runAgentWithProvider) => {
    const history: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Please calculate 15 + 27.",
      }),
    ];

    // In a budget-constrained scenario, wasting turn 1 on learn_skill delays execution.
    // With decision model auto-activation, turn 1 must execute the skill tool directly.
    await agentDeps(history)(runAgentWithProvider)({
      maxIterations: 5,
      tools: [],
      skills: [calculatorSkill],
      prompt: "You are a helpful assistant.",
      timezoneIANA: "UTC",
    });

    // Verify decision model auto-injected the synthetic learn_skill event before model execution
    const autoLearnCall = history.find(
      (e) => e.type === "tool_call" && e.id.startsWith("auto-learn-"),
    );
    assert(
      autoLearnCall,
      "Should have auto-injected learn_skill from decision model",
    );

    // Verify the agent NEVER wasted a turn calling learn_skill on its own
    const explicitLearnCalls = history.filter(
      (e) =>
        e.type === "tool_call" &&
        e.name === learnSkillToolName &&
        !e.id.startsWith("auto-learn-"),
    );
    assertEquals(
      explicitLearnCalls.length,
      0,
      `Agent should not call learn_skill explicitly when decision model auto-activates. Found: ${
        JSON.stringify(explicitLearnCalls)
      }`,
    );

    const runCommandCall = history.find(
      (e) => e.type === "tool_call" && e.name === runCommandToolName,
    );
    assert(runCommandCall, "Should call run_command on turn 1 directly");
  },
  1, // retries = 1 so failure surfaces immediately
);

runForAllProviders(
  "decision model skill deactivation: skill is automatically turned off when conversation topic changes",
  async (runAgentWithProvider) => {
    // Turn 1: user asks a math question -> calculator is used
    const turn1History: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "What is 7 * 8?",
      }),
    ];

    await agentDeps(turn1History)(runAgentWithProvider)({
      maxIterations: 5,
      tools: [],
      skills: [calculatorSkill],
      prompt: "You are a helpful assistant.",
      timezoneIANA: "UTC",
    });

    // Turn 2: topic changes to general conversation with no math
    const turn2History: HistoryEvent[] = [
      ...turn1History,
      participantUtteranceTurn({
        name: "user",
        text: "Thanks! What is your favorite color?",
      }),
    ];

    await agentDeps(turn2History)(runAgentWithProvider)({
      maxIterations: 3,
      tools: [],
      skills: [calculatorSkill],
      prompt: "You are a helpful assistant.",
      timezoneIANA: "UTC",
    });

    // Inspect the spec computed for turn 2: calculator must NOT be in active skills
    const specTurn2 = getSpecForTurn(
      {
        prompt: "You are a helpful assistant.",
        skills: [calculatorSkill],
        tools: [],
      },
      turn2History,
    );

    assert(
      !specTurn2.prompt.includes("### Active Skill: calculator"),
      "Turn 2 should automatically turn off the calculator skill when the topic changes",
    );
  },
  1, // retries = 1
);
