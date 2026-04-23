import { assert } from "@std/assert";
import { runAgent } from "../mod.ts";
import {
  type HistoryEvent,
  injectCallModel,
  ownUtteranceTurn,
  ownUtteranceTurnWithMetadata,
  participantUtteranceTurn,
} from "../src/agent.ts";
import { agentDeps, noopRewriteHistory } from "../test_helpers.ts";

// Provider-agnostic: when the model hits its output token budget it returns
// a truncated own_utterance (signalled by truncated=true). runAbstractAgent
// must NOT emit the partial text; it must inject a correctional own_thought
// and re-run the turn so the model produces a complete (briefer) response.
Deno.test(
  "runAgent retries with correctional thought when response is truncated",
  async () => {
    let call = 0;
    const fakeCallModel = () => {
      call++;
      if (call === 1) {
        const partial = ownUtteranceTurnWithMetadata(
          "שלום, זוהי התחלה של תשובה שנקט",
          undefined,
        );
        return Promise.resolve([{ ...partial, truncated: true }]);
      }
      return Promise.resolve([ownUtteranceTurn("תשובה מלאה וקצרה.")]);
    };

    const history: HistoryEvent[] = [
      participantUtteranceTurn({ name: "user", text: "hi" }),
    ];

    await injectCallModel(fakeCallModel)(async () => {
      await agentDeps(history)(runAgent)({
        maxIterations: 3,
        onMaxIterationsReached: () => {},
        tools: [],
        prompt: "unused in fake",
        rewriteHistory: noopRewriteHistory,
        timezoneIANA: "UTC",
      });
    })();

    assert(call === 2, `expected 2 calls after truncation retry, got ${call}`);
    const utterances = history.filter((e) => e.type === "own_utterance");
    assert(
      utterances.length === 1,
      `expected 1 final utterance, got ${utterances.length}`,
    );
    if (utterances[0].type !== "own_utterance") throw new Error("unreachable");
    assert(
      utterances[0].text === "תשובה מלאה וקצרה.",
      `expected complete reply, got: ${utterances[0].text}`,
    );
    assert(
      !history.some(
        (e) =>
          e.type === "own_utterance" &&
          e.text.includes("נקט"),
      ),
      "truncated partial must not leak to history",
    );
  },
);

Deno.test(
  "runAgent gives up after max truncation retries and emits last partial",
  async () => {
    let call = 0;
    const fakeCallModel = () => {
      call++;
      const partial = ownUtteranceTurnWithMetadata(
        `attempt ${call} partial`,
        undefined,
      );
      return Promise.resolve([{ ...partial, truncated: true }]);
    };

    const history: HistoryEvent[] = [
      participantUtteranceTurn({ name: "user", text: "hi" }),
    ];

    await injectCallModel(fakeCallModel)(async () => {
      await agentDeps(history)(runAgent)({
        maxIterations: 10,
        onMaxIterationsReached: () => {},
        tools: [],
        prompt: "unused in fake",
        rewriteHistory: noopRewriteHistory,
        timezoneIANA: "UTC",
      });
    })();

    assert(call <= 4, `retries must be bounded, got ${call} calls`);
    const utterances = history.filter((e) => e.type === "own_utterance");
    assert(
      utterances.length === 1,
      `expected 1 fallback utterance, got ${utterances.length}`,
    );
  },
);
