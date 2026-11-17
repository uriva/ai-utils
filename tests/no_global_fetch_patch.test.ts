import { overrideTime } from "../mod.ts";
import { participantUtteranceTurn } from "../src/agent.ts";
import type { AgentSpec } from "../src/agent.ts";
import { agentDeps, runForAllProviders } from "../test_helpers.ts";

const fixedTestTimestamp = 1_700_000_000_000;

runForAllProviders(
  "agent leaves globalThis.fetch untouched",
  overrideTime(() => fixedTestTimestamp)(
    async (runAgentWithProvider: (spec: AgentSpec) => Promise<void>) => {
      const fetchBefore = globalThis.fetch;
      const history = [participantUtteranceTurn({ name: "User", text: "hi" })];
      await agentDeps(history)(runAgentWithProvider)({
        maxIterations: 1,
        tools: [],
        prompt: "Reply with exactly: ok",
        timezoneIANA: "UTC",
      });
      const fetchAfter = globalThis.fetch;
      if (fetchBefore !== fetchAfter) {
        throw new Error(
          "runAgent replaced globalThis.fetch — instrumentation must not touch native globals",
        );
      }
    },
  ),
  3,
  true,
);
