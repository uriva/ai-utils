import { assertEquals } from "@std/assert";
import { pipe } from "gamla";
import {
  cleanActiveMemoryToolName,
  extractCandidateToolEpisodes,
  type HistoryEvent,
  injectDecisionModel,
  ownUtteranceTurn,
  participantUtteranceTurn,
  runAgent,
  toolResultTurn,
  toolUseTurn,
} from "../mod.ts";
import { injectCallModel } from "../src/agent.ts";
import { agentDeps, someTool } from "../test_helpers.ts";

Deno.test("extractCandidateToolEpisodes: protects N-1 and excludes current in-flight turn", () => {
  const history: HistoryEvent[] = [
    // Turn 1: concluded tool turn
    participantUtteranceTurn({ name: "user", text: "Download dataset A" }),
    toolUseTurn({ name: "download_data", args: { id: "A" } }),
    toolResultTurn({ toolCallId: "id-1", result: "Downloaded dataset A" }),
    ownUtteranceTurn("Dataset A downloaded successfully."),

    // Turn 2: concluded tool turn
    participantUtteranceTurn({ name: "user", text: "Parse dataset A" }),
    toolUseTurn({ name: "parse_data", args: { id: "A" } }),
    toolResultTurn({ toolCallId: "id-2", result: "Parsed 500 rows" }),
    ownUtteranceTurn("Parsed 500 rows from dataset A."),

    // Turn 3: immediate prior turn (N-1) - MUST BE PROTECTED
    participantUtteranceTurn({
      name: "user",
      text: "Train model on dataset A",
    }),
    toolUseTurn({ name: "train_model", args: { epochs: 10 } }),
    toolResultTurn({
      toolCallId: "id-3",
      result: "Model trained with 94% accuracy",
    }),
    ownUtteranceTurn("Trained model with 94% accuracy."),

    // Turn 4: active current turn (unanswered)
    participantUtteranceTurn({
      name: "user",
      text: "Now generate evaluation metrics",
    }),
  ];

  const candidates = extractCandidateToolEpisodes(history);

  // Must only contain Turn 1 and Turn 2! Turn 3 (N-1) must be protected, Turn 4 is in-flight.
  assertEquals(candidates.length, 2, "Expected exactly 2 candidate episodes");
  assertEquals(candidates[0].turnIndex, 1);
  assertEquals(candidates[0].userText, "Download dataset A");
  assertEquals(candidates[1].turnIndex, 2);
  assertEquals(candidates[1].userText, "Parse dataset A");
});

Deno.test("extractCandidateToolEpisodes: excludes episodes that are already compacted", () => {
  const t1 = 1700000000000;
  const history: HistoryEvent[] = [
    // Turn 1: concluded tool turn
    {
      ...participantUtteranceTurn({ name: "user", text: "Step 1" }),
      timestamp: t1,
    },
    {
      ...toolUseTurn({ name: "tool1", args: {} }),
      timestamp: t1 + 100,
    },
    {
      ...toolResultTurn({ toolCallId: "id-1", result: "ok" }),
      timestamp: t1 + 200,
    },
    {
      ...ownUtteranceTurn("Step 1 done."),
      timestamp: t1 + 300,
    },

    // An existing clean_active_memory directive covering Turn 1
    {
      id: "clean-call-1",
      type: "tool_call",
      name: cleanActiveMemoryToolName,
      parameters: {
        start_time: new Date(t1).toISOString(),
        end_time: new Date(t1 + 300).toISOString(),
        summary: "Step 1 completed.",
      },
      timestamp: t1 + 400,
      isOwn: true,
    },
    {
      id: "clean-res-1",
      type: "tool_result",
      result: "Successfully summarized 4 events...",
      timestamp: t1 + 401,
      isOwn: true,
      toolCallId: "clean-call-1",
    },

    // Turn 2: concluded tool turn
    {
      ...participantUtteranceTurn({ name: "user", text: "Step 2" }),
      timestamp: t1 + 1000,
    },
    {
      ...toolUseTurn({ name: "tool2", args: {} }),
      timestamp: t1 + 1100,
    },
    {
      ...toolResultTurn({ toolCallId: "id-2", result: "ok" }),
      timestamp: t1 + 1200,
    },
    {
      ...ownUtteranceTurn("Step 2 done."),
      timestamp: t1 + 1300,
    },

    // Turn 3: immediate prior turn (N-1)
    {
      ...participantUtteranceTurn({ name: "user", text: "Step 3" }),
      timestamp: t1 + 2000,
    },
    {
      ...toolUseTurn({ name: "tool3", args: {} }),
      timestamp: t1 + 2050,
    },
    {
      ...toolResultTurn({ toolCallId: "id-3", result: "ok" }),
      timestamp: t1 + 2080,
    },
    {
      ...ownUtteranceTurn("Step 3 done."),
      timestamp: t1 + 2100,
    },

    // Turn 4: active current turn
    {
      ...participantUtteranceTurn({ name: "user", text: "Step 4" }),
      timestamp: t1 + 3000,
    },
  ];

  const candidates = extractCandidateToolEpisodes(history);

  // Turn 1 is already compacted; Turn 3 is N-1; so only Turn 2 is a candidate
  assertEquals(
    candidates.length,
    1,
    "Expected only 1 uncompacted candidate episode",
  );
  assertEquals(candidates[0].turnIndex, 2);
});

Deno.test("runAgent auto-cleanup with decision model: automatically emits clean_active_memory for concluded episodes", async () => {
  const baseTime = 1700000000000;
  // Generate filler tool output so total tokens exceeds the 8,000 token pre-gate
  const filler = "verbose diagnostic log output line from build system ".repeat(
    500,
  ); // ~26,000 chars

  const history: HistoryEvent[] = [
    // Turn 1 (concluded)
    {
      ...participantUtteranceTurn({
        name: "user",
        text: "Run system diagnostics",
      }),
      timestamp: baseTime + 1000,
    },
    {
      ...toolUseTurn({ name: "diag_tool", args: {} }),
      timestamp: baseTime + 1100,
      id: "diag-call-1",
    },
    {
      ...toolResultTurn({
        toolCallId: "diag-call-1",
        result: `Diagnostics output: ${filler}`,
      }),
      timestamp: baseTime + 1200,
      id: "diag-res-1",
    },
    {
      ...ownUtteranceTurn(
        "System diagnostics ran cleanly. All 14 subsystems healthy.",
      ),
      timestamp: baseTime + 1300,
    },

    // Turn 2 (concluded)
    {
      ...participantUtteranceTurn({ name: "user", text: "Run lint and check" }),
      timestamp: baseTime + 2000,
    },
    {
      ...toolUseTurn({ name: "lint_tool", args: {} }),
      timestamp: baseTime + 2100,
      id: "lint-call-2",
    },
    {
      ...toolResultTurn({
        toolCallId: "lint-call-2",
        result: `Lint output: ${filler}`,
      }),
      timestamp: baseTime + 2200,
      id: "lint-res-2",
    },
    {
      ...ownUtteranceTurn("Lint check passed with 0 errors."),
      timestamp: baseTime + 2300,
    },

    // Turn 3: immediate prior turn (N-1) - protected
    {
      ...participantUtteranceTurn({ name: "user", text: "Deploy to staging" }),
      timestamp: baseTime + 3000,
    },
    {
      ...toolUseTurn({ name: "deploy_tool", args: {} }),
      timestamp: baseTime + 3100,
      id: "deploy-call-3",
    },
    {
      ...toolResultTurn({
        toolCallId: "deploy-call-3",
        result: "Deployed to staging-v1.",
      }),
      timestamp: baseTime + 3200,
      id: "deploy-res-3",
    },
    {
      ...ownUtteranceTurn("Successfully deployed to staging."),
      timestamp: baseTime + 3300,
    },

    // Turn 4: active turn
    {
      ...participantUtteranceTurn({
        name: "user",
        text: "Great, now send the release notes to the team.",
      }),
      timestamp: baseTime + 4000,
    },
  ];

  let modelReceivedHistory: HistoryEvent[] = [];

  // Inject a decision model that decides to compact Turn 1 and Turn 2
  const mockDecisionCaller = (
    _state: unknown,
    // deno-lint-ignore no-explicit-any
    questions: Record<string, any>,
    // deno-lint-ignore no-explicit-any
  ): Promise<Record<string, any>> => {
    // deno-lint-ignore no-explicit-any
    const answers: Record<string, any> = {};
    for (const key of Object.keys(questions)) {
      answers[key] = { type: "noul", noul: 0.92 }; // high score -> compact
    }
    return Promise.resolve(answers);
  };

  let it = 0;
  const fakeCallModel = (received: HistoryEvent[]): Promise<HistoryEvent[]> => {
    it++;
    if (it === 1) {
      return Promise.resolve([
        toolUseTurn({ name: "someTool", args: {} }),
      ]);
    }
    modelReceivedHistory = JSON.parse(JSON.stringify(received));
    return Promise.resolve([
      ownUtteranceTurn("I have sent the release notes to the team."),
    ]);
  };

  await pipe(
    injectDecisionModel(mockDecisionCaller),
    injectCallModel(fakeCallModel),
    agentDeps(history),
  )(async () => {
    await runAgent({
      provider: "anthropic",
      maxIterations: 2,
      tools: [someTool],
      prompt: "You are an operations assistant.",
      timezoneIANA: "UTC",
    });
  })();

  // 1. Synthetic clean_active_memory events were emitted
  const cleanCalls = history.filter(
    (e) => e.type === "tool_call" && e.name === cleanActiveMemoryToolName,
  );
  assertEquals(
    cleanCalls.length,
    2,
    "Expected 2 auto-cleanup calls for Turn 1 and Turn 2",
  );

  // 2. Model context has Turn 1 compacted and Turn 3 (N-1) preserved
  assertEquals(
    modelReceivedHistory.some((e) => e.id === "diag-call-1"),
    false,
    "Turn 1 tool call should be compacted in model context",
  );
  assertEquals(
    modelReceivedHistory.some((e) => e.id === "deploy-call-3"),
    true,
    "Turn 3 (N-1) must be preserved in model context",
  );

  // 2. Storage (history) retains the raw events for uncompromised auditability
  assertEquals(
    history.some((e) => e.id === "diag-res-1"),
    true,
    "Raw storage retains original tool result",
  );
  assertEquals(
    history.some((e) => e.id === "lint-res-2"),
    true,
    "Raw storage retains original tool result",
  );
});
