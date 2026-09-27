import { assertEquals } from "@std/assert";
import {
  disambiguateDuplicateToolCalls,
  type HistoryEvent,
  participantUtteranceTurn,
  projectHistoryToModelContext,
  toolResultTurn,
  toolUseTurn,
} from "../mod.ts";
import { agentDeps, runForAllProviders, someTool } from "../test_helpers.ts";

const seedHistory = (): HistoryEvent[] => {
  const call1 = toolUseTurn({
    id: "call-shared",
    name: someTool.name,
    args: {},
  });
  const call2 = toolUseTurn({
    id: "call-shared",
    name: someTool.name,
    args: {},
  });
  const history: HistoryEvent[] = [
    participantUtteranceTurn({
      name: "user",
      text: "Please call the doSomethingUnique tool.",
    }),
    call1,
    toolResultTurn({ result: "43212e8e", toolCallId: "call-shared" }),
    participantUtteranceTurn({
      name: "user",
      text: "Now please call it again.",
    }),
    call2,
    toolResultTurn({ result: "43212e8e", toolCallId: "call-shared" }),
    participantUtteranceTurn({
      name: "user",
      text: "Thanks! Acknowledge briefly in one short sentence.",
    }),
  ];
  history.forEach((event, index) => {
    event.timestamp = index + 1;
  });
  return history;
};

Deno.test(
  "disambiguateDuplicateToolCalls assigns unique IDs and remaps matching results",
  () => {
    const history = seedHistory();
    const disambiguated = disambiguateDuplicateToolCalls(history);
    const toolCalls = disambiguated.filter((e) => e.type === "tool_call");
    assertEquals(toolCalls.length, 2);
    assertEquals(toolCalls[0].id !== toolCalls[1].id, true);
    const toolResults = disambiguated.filter((e) => e.type === "tool_result");
    assertEquals(toolResults.length, 2);
    assertEquals(toolResults[0].toolCallId, toolCalls[0].id);
    assertEquals(toolResults[1].toolCallId, toolCalls[1].id);
  },
);

Deno.test(
  "projectHistoryToModelContext preserves all results for duplicate tool_call IDs without synthetic results",
  async () => {
    const history = seedHistory();
    const projected = await projectHistoryToModelContext({
      rawHistory: history,
    });
    const hasSynthetic = projected.some((e) =>
      e.id.includes("synthetic-result")
    );
    assertEquals(hasSynthetic, false);
    const toolResults = projected.filter((e) => e.type === "tool_result");
    assertEquals(toolResults.length, 2);
  },
);

runForAllProviders(
  "duplicate tool_call id across turns keeps the provider request valid and answers",
  async (runAgentWithProvider) => {
    const history = seedHistory();
    await agentDeps(history)(runAgentWithProvider)({
      maxIterations: 3,
      tools: [someTool],
      prompt: "You are a helpful assistant. Be brief.",
      timezoneIANA: "UTC",
    });
    const reply = history.find(
      (event) => event.type === "own_utterance" && event.timestamp > 7,
    );
    if (!reply) throw new Error("expected the agent to acknowledge");
  },
);
