import { assertEquals } from "@std/assert";
import {
  type AgentInputs,
  getSpecForTurn,
  type HistoryEvent,
  participantUtteranceTurn,
} from "../src/agent.ts";
import { learnedSkillCall, weatherSkill } from "../test_helpers.ts";

const baseSpec: AgentInputs = {
  tools: [],
  skills: [weatherSkill],
  prompt: "You are a helpful assistant.",
};

Deno.test("getSpecForTurn caches per history reference without going stale", () => {
  const plainHistory: HistoryEvent[] = [
    participantUtteranceTurn({ name: "user", text: "hello" }),
  ];

  const first = getSpecForTurn(baseSpec, plainHistory);
  const second = getSpecForTurn(baseSpec, plainHistory);
  assertEquals(second, first);
  assertEquals(
    first.prompt.includes("Active Skill"),
    false,
    "no learned skill → no active-skills section",
  );

  const withLearnedSkill: HistoryEvent[] = [
    participantUtteranceTurn({ name: "user", text: "hello" }),
    { ...learnedSkillCall(weatherSkill.name), timestamp: Date.now() + 1000 },
  ];
  const activated = getSpecForTurn(baseSpec, withLearnedSkill);
  assertEquals(
    activated.prompt.includes(`### Active Skill: ${weatherSkill.name}`),
    true,
    "a newly learned skill must activate on the next distinct history",
  );
  assertEquals(activated.skills?.map((s) => s.name), [weatherSkill.name]);

  const deactivated = getSpecForTurn(baseSpec, plainHistory);
  assertEquals(
    deactivated.prompt.includes("Active Skill"),
    false,
    "reverting to the earlier history must not reuse the activated prompt",
  );
});
