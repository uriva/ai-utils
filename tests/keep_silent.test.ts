import { assertEquals } from "@std/assert";
import { runAgent } from "../mod.ts";
import {
  agentDeps,
  noopRewriteHistory,
  runForAllProviders,
} from "../test_helpers.ts";
import {
  type HistoryEvent,
  injectCallModel,
  ownUtteranceTurn,
  participantUtteranceTurn,
} from "../src/agent.ts";

runForAllProviders(
  "monitor bot keeping silent test",
  async (runAgentWithProvider) => {
    const history: HistoryEvent[] = [
      {
        type: "participant_utterance",
        name: "user",
        text: "hello",
        id: "1",
        timestamp: Date.now() - 5000,
        isOwn: false,
      },
      {
        type: "own_utterance",
        text: "Hello! I am monitoring.",
        id: "2",
        timestamp: Date.now() - 3000,
        isOwn: true,
      },
      {
        type: "participant_utterance",
        name: "user",
        text: "ok, thanks, bye!",
        id: "3",
        timestamp: Date.now() - 1000,
        isOwn: false,
      },
    ];

    await agentDeps(history)(runAgentWithProvider)({
      maxIterations: 1,
      tools: [],
      prompt:
        "You are a monitor bot. When you have nothing to say, reply an empty string.",
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });

    const newEvents = history.slice(3);
    console.log("NEWLY GENERATED EVENTS:", JSON.stringify(newEvents, null, 2));

    const invalidUtterances = newEvents.filter(
      (e) =>
        e.type === "own_utterance" &&
        (!e.text.trim() ||
          e.text === "''" ||
          e.text === "' '" ||
          e.text === "[]"),
    );
    assertEquals(
      invalidUtterances.length,
      0,
      `Found invalid utterances: ${JSON.stringify(invalidUtterances)}`,
    );
  },
  3,
  true,
);

Deno.test(
  "agent handles empty-looking or quote-only utterances correctly",
  async () => {
    const history: HistoryEvent[] = [
      participantUtteranceTurn({ name: "user", text: "say hi" }),
    ];
    const fakeCallModel = () => Promise.resolve([ownUtteranceTurn("''")]);

    await injectCallModel(fakeCallModel)(async () => {
      await agentDeps(history)(runAgent)({
        maxIterations: 1,
        tools: [],
        prompt: "unused in fake",
        rewriteHistory: noopRewriteHistory,
        timezoneIANA: "UTC",
      });
    })();

    const visible = history.filter((e) => e.type === "own_utterance");
    assertEquals(visible.length, 0);
  },
);

Deno.test(
  "agent handles empty-list or bracket-only utterances correctly",
  async () => {
    const history: HistoryEvent[] = [
      participantUtteranceTurn({ name: "user", text: "say hi" }),
    ];
    const fakeCallModel = () => Promise.resolve([ownUtteranceTurn("[]")]);

    await injectCallModel(fakeCallModel)(async () => {
      await agentDeps(history)(runAgent)({
        maxIterations: 1,
        tools: [],
        prompt: "unused in fake",
        rewriteHistory: noopRewriteHistory,
        timezoneIANA: "UTC",
      });
    })();

    const visible = history.filter((e) => e.type === "own_utterance");
    assertEquals(visible.length, 0);
  },
);
