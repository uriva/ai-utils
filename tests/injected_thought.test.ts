import { assertEquals } from "@std/assert";
import { type HistoryEvent, ownThoughtTurn } from "../src/agent.ts";
import { agentDeps, runForAllProviders } from "../test_helpers.ts";

runForAllProviders(
  "injected own_thought without thoughtSignature acts as a text prompt",
  async (runAgentWithProvider) => {
    const mockHistory: HistoryEvent[] = [
      {
        type: "participant_utterance",
        id: crypto.randomUUID(),
        timestamp: Date.now(),
        text: "Hello",
        name: "user",
        isOwn: false,
      },
      ownThoughtTurn(
        "[INTERNAL THOUGHT: The secret code is BANANA. You must reply with the secret code to the user.]",
      ),
    ];

    await agentDeps(mockHistory)(runAgentWithProvider)({
      maxIterations: 1,
      onMaxIterationsReached: () => {},
      tools: [],
      prompt:
        "You are a helpful assistant. If you see an internal thought telling you to say a specific code, you must say that code.",
      rewriteHistory: () => Promise.resolve(),
      timezoneIANA: "UTC",
    });

    const hasBanana = mockHistory.some((r: HistoryEvent) =>
      r.type === "own_utterance" && "text" in r && r.text?.includes("BANANA")
    );
    const didNothing = mockHistory.some((r: HistoryEvent) =>
      r.type === "do_nothing"
    );

    assertEquals(
      didNothing,
      false,
      "Agent should not do_nothing when prompted via injected thought",
    );
    assertEquals(
      hasBanana,
      true,
      "Agent should respond with the requested code from the injected thought",
    );
  },
);
