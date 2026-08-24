import { participantUtteranceTurn } from "../src/agent.ts";
import type { AgentSpec } from "../src/agent.ts";
import {
  agentDeps,
  noopRewriteHistory,
  runForAllProviders,
} from "../test_helpers.ts";

runForAllProviders(
  "agent leaves globalThis.fetch untouched",
  async (runAgentWithProvider: (spec: AgentSpec) => Promise<void>) => {
    const fetchBefore = globalThis.fetch;
    const history = [participantUtteranceTurn({ name: "User", text: "hi" })];
    await agentDeps(history)(runAgentWithProvider)({
      maxIterations: 1,
      tools: [],
      prompt: "Reply with exactly: ok",
      lightModel: true,
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
    const fetchAfter = globalThis.fetch;
    if (fetchBefore !== fetchAfter) {
      throw new Error(
        "runAgent replaced globalThis.fetch — instrumentation must not touch native globals",
      );
    }
  },
  3,
  true,
);
