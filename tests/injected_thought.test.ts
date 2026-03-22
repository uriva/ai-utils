import { assertEquals } from "@std/assert";
import { type HistoryEvent, ownThoughtTurn } from "../src/agent.ts";
import { geminiAgentCaller } from "../src/geminiAgent.ts";
import { injectGeminiToken } from "../src/gemini.ts";

Deno.test("injected own_thought without thoughtSignature acts as a text prompt", async () => {
  if (!Deno.env.get("GEMINI_API_KEY")) return; // Skip if no API key

  await injectGeminiToken(Deno.env.get("GEMINI_API_KEY") as string)(
    async () => {
      const rewriteHistory = () => Promise.resolve();

      const caller = geminiAgentCaller({
        prompt:
          "You are a helpful assistant. If you see an internal thought telling you to say a specific code, you must say that code.",
        tools: [],
        rewriteHistory,
        timezoneIANA: "UTC",
        maxIterations: 5,
        onMaxIterationsReached: () => {},
      });

      const events: HistoryEvent[] = [
        {
          type: "participant_utterance",
          id: crypto.randomUUID(),
          timestamp: Date.now(),
          text: "Hello",
          name: "user",
          isOwn: false,
        },
        // This simulates injectContextNotification without a thoughtSignature
        ownThoughtTurn(
          "[INTERNAL THOUGHT: The secret code is BANANA. You must reply with the secret code to the user.]",
        ),
      ];

      // deno-lint-ignore no-explicit-any
      const result = await caller(events as any);

      console.log("AGENT RESULT:", JSON.stringify(result, null, 2));

      // The agent should output a text response containing BANANA, not do_nothing.
      const hasBanana = result.some((r: HistoryEvent) =>
        r.type === "own_utterance" && "text" in r && r.text?.includes("BANANA")
      );
      const didNothing = result.some((r: HistoryEvent) =>
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
  )();
});
