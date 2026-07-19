import { assert, assertEquals } from "@std/assert";
import { pipe } from "gamla";
import { runAgent } from "../mod.ts";
import {
  injectCallModel,
  ownUtteranceTurn,
  participantUtteranceTurn,
} from "../src/agent.ts";
import { agentDeps, noopRewriteHistory } from "../test_helpers.ts";

// When the model reasons "out loud" it emits the reasoning as the leading
// visible text part and the actual reply as the final one. Only the reply may
// reach the user; the reasoning must be demoted to an own_thought so it is
// neither delivered nor replayed back to the model as its own utterance
// (which teaches it to keep leaking). Injected fake callModel keeps this
// provider-agnostic and deterministic.
Deno.test("runAgent delivers only the reply when the model emits leading reasoning parts", async () => {
  const reasoning =
    "אני מבין שהמשתמש לא יהיה בבית מחר בבוקר, אז השקילה תהיה ביום שני. אני אענה לו בחום ואאשר את התוכנית.";
  const reply = "מובן לגמרי! הכל בסדר, נעדכן ביום שני בבוקר. 👌";
  const history = [participantUtteranceTurn({ name: "user", text: "היי" })];
  await pipe(
    injectCallModel(() =>
      Promise.resolve([ownUtteranceTurn(reasoning), ownUtteranceTurn(reply)])
    ),
    agentDeps(history),
  )(async () => {
    await runAgent({
      maxIterations: 1,
      tools: [],
      prompt: "unused",
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
  })();
  const utterances = history.filter((e) => e.type === "own_utterance");
  assertEquals(utterances.length, 1);
  assertEquals(
    utterances[0].type === "own_utterance" && utterances[0].text,
    reply,
  );
  assert(
    history.some((e) => e.type === "own_thought" && e.text === reasoning),
    "expected the reasoning to be recorded as own_thought",
  );
});
