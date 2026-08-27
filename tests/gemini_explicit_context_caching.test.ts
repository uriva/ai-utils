import { assertEquals } from "@std/assert";
import {
  type AgentSpec,
  type HistoryEvent,
  injectCallModelWrapper,
  participantUtteranceTurn,
  tool,
  z,
} from "../mod.ts";
import { injectTokenUsage, type TokenUsage } from "../src/geminiAgent.ts";
import {
  agentDeps,
  injectSecrets,
  noopRewriteHistory,
  runWithProvider,
} from "../test_helpers.ts";

Deno.test({
  name:
    "Gemini Explicit Context Caching: system instructions are explicitly cached and reused across turns",
  fn: injectSecrets(async () => {
    // Large static prompt with domain instructions (~15,000 tokens)
    const largePrompt =
      "Knowledge base regarding solar systems, planets, atmospheric properties, geological formations, and celestial mechanics. "
        .repeat(1000);

    const recordedUsages: TokenUsage[] = [];

    const history: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "User",
        text:
          "What is the climate of Planet Zephyr? Look it up in the database.",
      }),
    ];

    const searchTool = tool({
      name: "search_planetary_database",
      description:
        "Search the planetary database for information about a planet",
      parameters: z.object({ planet: z.string() }),
      handler: ({ planet }) =>
        Promise.resolve(
          `Database result: ${planet} has a temperate oceanic climate.`,
        ),
    });

    const spec: AgentSpec = {
      provider: "google",
      prompt: largePrompt,
      tools: [searchTool],
      rewriteHistory: noopRewriteHistory,
      maxIterations: 5,
      timezoneIANA: "UTC",
    };

    // Bypass rmmbr test cacher to measure real provider token usage metrics
    await injectCallModelWrapper(({ inner }) => inner)(
      injectTokenUsage((usage) => {
        recordedUsages.push(usage);
      })(agentDeps(history)(runWithProvider("google"))),
    )(spec);

    // Verify agent completed multi-turn interaction (tool_call + final reply)
    const toolCalls = history.filter((e) => e.type === "tool_call");
    assertEquals(toolCalls.length >= 1, true, "Expected at least 1 tool call");

    assertEquals(
      recordedUsages.length >= 2,
      true,
      `Expected at least 2 LLM iterations (got ${recordedUsages.length})`,
    );

    // With explicit context caching, cachedContentTokenCount must cover the vast majority of prompt tokens,
    // and uncached fresh prompt tokens (promptTokenCount - cachedContentTokenCount) must be minimal (< 500 tokens).
    for (let i = 0; i < recordedUsages.length; i++) {
      const usage = recordedUsages[i];
      const cached = usage.cachedContentTokenCount ?? 0;
      const totalPrompt = usage.promptTokenCount ?? 0;
      const freshPrompt = totalPrompt - cached;
      assertEquals(
        cached > 15000,
        true,
        `Expected turn ${
          i + 1
        } to have > 15,000 cached tokens, but got ${cached}`,
      );
      assertEquals(
        freshPrompt < 500,
        true,
        `Expected turn ${
          i + 1
        } fresh prompt tokens (delta) to be < 500 tokens, but got ${freshPrompt} (total: ${totalPrompt}, cached: ${cached})`,
      );
    }
  }),
});
