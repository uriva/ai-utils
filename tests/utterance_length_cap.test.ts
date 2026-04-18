import { assert } from "@std/assert";
import { runAgent } from "../mod.ts";
import {
  type HistoryEvent,
  injectCallModel,
  maxUtteranceChars,
  ownUtteranceTurn,
  participantUtteranceTurn,
} from "../src/agent.ts";
import { agentDeps, noopRewriteHistory } from "../test_helpers.ts";

// Provider-agnostic: enforces the runAbstractAgent invariant that any
// own_utterance emitted to the outside world fits within the protocol cap
// (assertTextLengthOk in @alice-and-bot/core, 4000 chars).
Deno.test(
  "runAgent splits oversized own_utterance from model into capped chunks",
  async () => {
    const huge = Array.from(
      { length: 8 },
      (_, i) => `Section ${i + 1}.\n${"lorem ipsum ".repeat(120)}`,
    ).join("\n\n");
    assert(huge.length > maxUtteranceChars);

    const fakeCallModel = () => Promise.resolve([ownUtteranceTurn(huge)]);

    const history: HistoryEvent[] = [
      participantUtteranceTurn({ name: "user", text: "long please" }),
    ];

    await injectCallModel(fakeCallModel)(async () => {
      await agentDeps(history)(runAgent)({
        maxIterations: 1,
        onMaxIterationsReached: () => {},
        tools: [],
        prompt: "unused in fake",
        rewriteHistory: noopRewriteHistory,
        timezoneIANA: "UTC",
      });
    })();

    const utterances = history.filter((e) => e.type === "own_utterance");
    assert(utterances.length > 1, "expected oversized output to be split");
    utterances.forEach((u) => {
      if (u.type !== "own_utterance") throw new Error("unreachable");
      assert(
        u.text.length <= maxUtteranceChars,
        `chunk length ${u.text.length} exceeds cap ${maxUtteranceChars}`,
      );
    });
  },
);
