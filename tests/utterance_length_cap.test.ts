import { assert, assertEquals } from "@std/assert";
import { pipe, sleep } from "gamla";
import { runAgent } from "../mod.ts";
import {
  type HistoryEvent,
  injectAccessHistory,
  injectCallModel,
  injectOutputEvent,
  maxUtteranceChars,
  ownThoughtTurn,
  ownUtteranceTurn,
  participantUtteranceTurn,
} from "../src/agent.ts";
import { agentDeps } from "../test_helpers.ts";

// Provider-agnostic: enforces the runAbstractAgent invariant that any
// own_utterance emitted to the outside world fits within the protocol cap
// (assertTextLengthOk in @alice-and-bot/core, 4000 chars).
Deno.test(
  "runAgent splits oversized own_utterance from model into capped chunks",
  async () => {
    const huge = Array.from(
      { length: 8 },
      (_, i) =>
        `Section ${i + 1}.\n${
          Array.from(
            { length: 120 },
            (_, j) => `word-${i}-${j}`,
          ).join(" ")
        }`,
    ).join("\n\n");
    assert(huge.length > maxUtteranceChars);

    const fakeCallModel = () => Promise.resolve([ownUtteranceTurn(huge)]);

    const history: HistoryEvent[] = [
      participantUtteranceTurn({ name: "user", text: "long please" }),
    ];

    await injectCallModel(fakeCallModel)(async () => {
      await agentDeps(history)(runAgent)({
        maxIterations: 1,
        tools: [],
        prompt: "unused in fake",
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

Deno.test(
  "runAgent awaits each split utterance chunk delivery sequentially before emitting the next chunk",
  async () => {
    const chunk1 = `Day 1 to Day 4.\n${
      Array.from({ length: 230 }, (_, j) => `first-part-${j}`).join(" ")
    }`;
    const chunk2 = `Day 5 conclusion.\n${
      Array.from({ length: 150 }, (_, j) => `second-part-${j}`).join(" ")
    }`;
    const huge = `${chunk1}\n\n${chunk2}`;
    assert(huge.length > maxUtteranceChars);

    const history: HistoryEvent[] = [
      participantUtteranceTurn({ name: "user", text: "5 day plan" }),
    ];
    const deliveredTexts: string[] = [];

    const run = pipe(
      injectAccessHistory(() => Promise.resolve(history)),
      injectOutputEvent(async (event: HistoryEvent) => {
        if (event.type === "own_utterance" && event.text.startsWith("Day 1")) {
          await sleep(40);
        }
        history.push(event);
        if (event.type === "own_utterance") {
          deliveredTexts.push(event.text.split("\n")[0]);
        }
      }),
      injectCallModel(() => Promise.resolve([ownUtteranceTurn(huge)])),
    )(() =>
      runAgent({
        maxIterations: 1,
        tools: [],
        prompt: "unused in fake",
        timezoneIANA: "UTC",
      })
    );

    await run();
    assertEquals(deliveredTexts, ["Day 1 to Day 4.", "Day 5 conclusion."]);
  },
);

Deno.test(
  "runAgent starts own_utterance in parallel with own_thought without waiting for thought persistence",
  async () => {
    const history: HistoryEvent[] = [
      participantUtteranceTurn({ name: "user", text: "hello" }),
    ];
    const startOrder: string[] = [];
    const finishOrder: string[] = [];

    const run = pipe(
      injectAccessHistory(() => Promise.resolve(history)),
      injectOutputEvent(async (event: HistoryEvent) => {
        startOrder.push(event.type);
        if (event.type === "own_thought") {
          await sleep(40);
        }
        history.push(event);
        finishOrder.push(event.type);
      }),
      injectCallModel(() =>
        Promise.resolve([
          ownThoughtTurn("thinking about response"),
          ownUtteranceTurn("Fast reply"),
        ])
      ),
    )(() =>
      runAgent({
        maxIterations: 1,
        tools: [],
        prompt: "unused in fake",
        timezoneIANA: "UTC",
      })
    );

    await run();
    assertEquals(startOrder, ["own_thought", "own_utterance"]);
    assertEquals(finishOrder, ["own_utterance", "own_thought"]);
  },
);
