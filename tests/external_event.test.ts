import { assertEquals } from "@std/assert";
import { externalEventTurn, type HistoryEvent } from "../src/agent.ts";
import { agentDeps, runForAllProviders } from "../test_helpers.ts";

// An external_event is authoritative world data that entered the conversation
// from outside the model's own action loop (e.g. an async command completion).
// It must (a) serialize on every provider without throwing, and (b) be treated
// as ground truth the model can act on — distinct from own_thought (reasoning)
// and from user input.
runForAllProviders(
  "external_event is treated as authoritative ground truth",
  async (runAgentWithProvider) => {
    const mockHistory: HistoryEvent[] = [
      {
        type: "participant_utterance",
        id: crypto.randomUUID(),
        timestamp: Date.now(),
        text: "What exit code did the background build finish with?",
        name: "user",
        isOwn: false,
      },
      externalEventTurn(
        "Background command deno-build finished with exit code 137.",
      ),
    ];

    await agentDeps(mockHistory)(runAgentWithProvider)({
      maxIterations: 1,
      tools: [],
      prompt:
        "You are a helpful assistant. Answer the user's question using the information available in the conversation history.",
      rewriteHistory: () => Promise.resolve(),
      timezoneIANA: "UTC",
    });

    const mentioned137 = mockHistory.some((r: HistoryEvent) =>
      r.type === "own_utterance" && "text" in r && r.text?.includes("137")
    );
    const didNothing = mockHistory.some((r: HistoryEvent) =>
      r.type === "do_nothing"
    );

    assertEquals(
      didNothing,
      false,
      "Agent should answer using the external_event, not do_nothing",
    );
    assertEquals(
      mentioned137,
      true,
      "Agent should relay the exit code reported by the external_event",
    );
  },
  3,
  false,
);
