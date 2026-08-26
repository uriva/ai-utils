import { assert, assertEquals } from "@std/assert";
import { runAgent } from "../mod.ts";
import {
  type HistoryEvent,
  injectCallModel,
  ownThoughtTurn,
  ownUtteranceTurn,
  participantUtteranceTurn,
} from "../src/agent.ts";
import { agentDeps, noopRewriteHistory } from "../test_helpers.ts";

const hour = 60 * 60 * 1000;

const proactiveNote = (match: number) =>
  `PROACTIVE TASK: You have a task to complete. This event just landed and its computed taste match for this user is ${match}%. This is only a moderate match so consider carefully whether it is worth interrupting the user before delivering it.`;

const compactedSummaryText =
  "[This summary covers the period from 2026-08-20 09:00 to 2026-08-20 18:00]\n\nThe user asked about jazz concerts and the bot recommended two events in Tel Aviv.";

const stalePlainThought = (
  text: string,
  ageMs: number,
): HistoryEvent => ({
  ...ownThoughtTurn(text),
  timestamp: Date.now() - ageMs,
});

Deno.test("stale platform notifications fold to one-line digests in model context", async () => {
  let captured: HistoryEvent[] = [];
  const fakeCallModel = (events: HistoryEvent[]) => {
    captured = events;
    return Promise.resolve([ownUtteranceTurn("Noted.")]);
  };
  await injectCallModel(fakeCallModel)(async () => {
    await agentDeps([
      {
        ...participantUtteranceTurn({
          name: "user",
          text: "Keep me posted on events.",
        }),
        timestamp: Date.now() - 3 * hour,
      },
      stalePlainThought(proactiveNote(86), 2 * hour),
      stalePlainThought(proactiveNote(88), 1 * hour),
      {
        ...ownThoughtTurn(compactedSummaryText),
        timestamp: Date.now() - 2 * hour,
      },
      stalePlainThought("A note from before the user last spoke.", 5 * hour),
    ])(runAgent)({
      maxIterations: 1,
      tools: [],
      prompt: "You are a helper.",
      lightModel: true,
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
  })();

  const thoughts = captured.filter((e) => e.type === "own_thought");
  const digests = thoughts.filter((e) =>
    e.text.startsWith("[Earlier platform notification (")
  );
  assertEquals(digests.length, 2);
  for (const d of digests) {
    assert(d.text.length < 200, `digest too long: ${d.text.length}`);
    assert(!d.text.includes("consider carefully whether"));
  }
  assertEquals(
    thoughts.some((e) => e.text === proactiveNote(86)),
    false,
    "stale note must not survive verbatim",
  );
});

Deno.test("recent platform notifications and compaction summaries keep full text", async () => {
  let captured: HistoryEvent[] = [];
  const fakeCallModel = (events: HistoryEvent[]) => {
    captured = events;
    return Promise.resolve([ownUtteranceTurn("Noted.")]);
  };
  const recentNote = proactiveNote(92);
  await injectCallModel(fakeCallModel)(async () => {
    await agentDeps([
      {
        ...participantUtteranceTurn({
          name: "user",
          text: "Keep me posted on events.",
        }),
        timestamp: Date.now() - 3 * hour,
      },
      stalePlainThought(recentNote, 5 * 60 * 1000),
      {
        ...ownThoughtTurn(compactedSummaryText),
        timestamp: Date.now() - 3 * hour - 1000,
      },
    ])(runAgent)({
      maxIterations: 1,
      tools: [],
      prompt: "You are a helper.",
      lightModel: true,
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
  })();

  const thoughts = captured.filter((e) => e.type === "own_thought");
  assertEquals(
    thoughts.some((e) => e.text === recentNote),
    true,
    "recent notification must keep full text",
  );
  assertEquals(
    thoughts.some((e) => e.text === compactedSummaryText),
    true,
    "compaction summaries must never be folded",
  );
});
