import { assert, assertEquals } from "@std/assert";
import { z } from "zod/v4";
import { runAgent } from "../mod.ts";
import { agentDeps } from "../test_helpers.ts";
import {
  type AgentSpec,
  getSpecForTurn,
  type HistoryEvent,
  injectCallModel,
  ownUtteranceTurn,
  participantUtteranceTurn,
  runCommandToolName,
  skillUnlearnAppliedMarkerText,
  skillUnlearnSkippedMarkerText,
  unlearnSkillToolName,
} from "../src/agent.ts";

const lookupSkillName = "lookup";
const lookupSkillToolName = "find";

const lookupSkill = {
  name: lookupSkillName,
  description: "Look things up",
  instructions: "LOOKUP_INSTRUCTIONS_MARKER",
  tools: [{
    name: lookupSkillToolName,
    description: "Look up a query",
    parameters: z.object({ query: z.string() }),
    handler: ({ query }: { query: string }) =>
      Promise.resolve(`result for ${query}`),
  }],
};

const lookupCommand = (query: string): HistoryEvent => ({
  type: "tool_call",
  id: `call-${query}`,
  name: runCommandToolName,
  parameters: {
    command: `${lookupSkillName}/${lookupSkillToolName}`,
    params: { query },
  },
  description: "Looking up",
  isOwn: true,
  timestamp: 1,
});

const unlearnCall = (skillName: string): HistoryEvent => ({
  type: "tool_call",
  id: `unlearn-${skillName}`,
  name: unlearnSkillToolName,
  parameters: { skillName },
  description: "Deactivating",
  isOwn: true,
  timestamp: 1,
});

const baseSpec = {
  maxIterations: 20,
  tools: [],
  skills: [lookupSkill],
  prompt: "You are a lookup assistant.",
  timezoneIANA: "UTC",
} satisfies AgentSpec;

const runScriptedTurns = async (turns: HistoryEvent[][]) => {
  const history: HistoryEvent[] = [
    participantUtteranceTurn({ name: "user", text: "find it for me" }),
  ];
  let turn = 0;
  const fakeCallModel = (): Promise<HistoryEvent[]> =>
    Promise.resolve(turns[turn++] ?? [ownUtteranceTurn("Done.")]);
  await injectCallModel(fakeCallModel)(async () => {
    await agentDeps(history)(runAgent)({
      ...baseSpec,
      maxIterations: 20,
    });
  })();
  return history;
};

const resultsFor = (history: HistoryEvent[], name: string) => {
  const callIds = new Set(
    history
      .filter((e) => e.type === "tool_call" && e.name === name)
      .map((e) => e.id),
  );
  return history
    .filter((e) =>
      e.type === "tool_result" && e.toolCallId && callIds.has(e.toolCallId)
    )
    .map((e) => e.type === "tool_result" ? e.result : "");
};

const activeSkillNamesIn = (history: HistoryEvent[]) =>
  getSpecForTurn(baseSpec, history).skills.map((s) => s.name);

Deno.test(
  "runAgent - unlearn_skill stops reporting success for a skill that stays in use mid-request",
  async () => {
    const history = await runScriptedTurns([
      [lookupCommand("first")],
      [unlearnCall(lookupSkillName)],
      [lookupCommand("second")],
      [unlearnCall(lookupSkillName)],
      [unlearnCall(lookupSkillName)],
      [ownUtteranceTurn("Found it.")],
    ]);

    const unlearnResults = resultsFor(history, unlearnSkillToolName);
    assertEquals(unlearnResults.length, 3);

    const [first, ...rest] = unlearnResults;
    assert(
      first.includes(skillUnlearnAppliedMarkerText),
      `First deactivation of an in-use skill should still apply. Got: "${first}"`,
    );
    assertEquals(
      rest.filter((r) => r.includes(skillUnlearnAppliedMarkerText)).length,
      0,
      `Deactivating a skill that was just re-used must not report success again. Got: ${
        JSON.stringify(rest, null, 2)
      }`,
    );
    for (const result of rest) {
      assert(
        result.includes(skillUnlearnSkippedMarkerText),
        `A refused deactivation should tell the model it changed nothing. Got: "${result}"`,
      );
    }

    assertEquals(
      activeSkillNamesIn(history),
      [lookupSkillName],
      "A refused deactivation must leave the skill active, not deactivate it in history replay",
    );
  },
);

Deno.test(
  "runAgent - unlearn_skill on a skill that is not active changes nothing",
  async () => {
    const history = await runScriptedTurns([
      [unlearnCall(lookupSkillName)],
      [unlearnCall(lookupSkillName)],
      [ownUtteranceTurn("Nothing to look up.")],
    ]);

    const unlearnResults = resultsFor(history, unlearnSkillToolName);
    assertEquals(unlearnResults.length, 2);
    for (const result of unlearnResults) {
      assert(
        !result.includes(skillUnlearnAppliedMarkerText),
        `Deactivating an inactive skill must not claim it removed tools. Got: "${result}"`,
      );
      assert(
        result.includes(skillUnlearnSkippedMarkerText),
        `A no-op deactivation should say so. Got: "${result}"`,
      );
    }
    assertEquals(activeSkillNamesIn(history), []);
  },
);

Deno.test(
  "runAgent - unlearn_skill still deactivates a skill once the request stops using it",
  async () => {
    const history = await runScriptedTurns([
      [lookupCommand("first")],
      [unlearnCall(lookupSkillName)],
      [ownUtteranceTurn("Found it.")],
    ]);

    const unlearnResults = resultsFor(history, unlearnSkillToolName);
    assertEquals(unlearnResults.length, 1);
    assert(
      unlearnResults[0].includes(skillUnlearnAppliedMarkerText),
      `A single deactivation at the end of a request should still apply. Got: "${
        unlearnResults[0]
      }"`,
    );
    assertEquals(activeSkillNamesIn(history), []);
  },
);
