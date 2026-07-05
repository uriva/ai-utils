import { assertEquals } from "@std/assert";
import { injectGeminiToken, runAgent } from "../mod.ts";
import { agentDeps, noopRewriteHistory } from "../test_helpers.ts";
import {
  type HistoryEvent,
  injectCallModel,
  participantUtteranceTurn,
} from "../src/agent.ts";
import { genJsonOverride } from "../src/genJson.ts";

Deno.test(
  "REPRO: agent exits silently on a new run when history contains a previous stop thought",
  async () => {
    // History contains a previous stop thought injected by checkProgress in a prior run
    const history: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "do something that gets stuck",
      }),
      {
        type: "own_thought",
        isOwn: true,
        id: "stop-thought-id",
        timestamp: Date.now() - 2000,
        text:
          "I'm working on this for some time and not making progress. I should instead stop and ask the user for help.",
      },
      participantUtteranceTurn({
        name: "user",
        text: "Are you done?", // new user message
      }),
    ];

    let hasStopThoughtInCall = true;

    // Mock model to return empty/do_nothing
    const fakeCallModel = (events?: HistoryEvent[]) => {
      const hist = events || history;
      hasStopThoughtInCall = hist.some(
        (e) =>
          e.type === "own_thought" &&
          e.text.startsWith(
            "I'm working on this for some time and not making progress.",
          ),
      );
      return Promise.resolve([
        {
          type: "do_nothing" as const,
          isOwn: true,
          id: "do-nothing-id",
          timestamp: Date.now(),
        },
      ]);
    };

    const mockGenJson = (
      _opts: unknown,
      _systemMsg: string,
      _zodType: unknown,
    ) => {
      return (_userMsg: string, _attachments?: unknown) => {
        return Promise.resolve({
          shouldContinue: false,
          thoughtInjection:
            "I'm working on this for some time and not making progress. I should instead stop.",
        });
      };
    };

    await injectGeminiToken("fake-token")(async () => {
      await genJsonOverride.inject(() => mockGenJson)(async () => {
        await injectCallModel(fakeCallModel)(async () => {
          await agentDeps(history)(runAgent)({
            maxIterations: 30, // normal user turn has maxIterations: 30
            tools: [],
            prompt: "You are a helpful assistant.",
            rewriteHistory: noopRewriteHistory,
            timezoneIANA: "UTC",
          });
        })();
      })();
    })();

    // Verify that the stop thought was filtered out from the history passed to the model
    assertEquals(hasStopThoughtInCall, false);

    // Buggy behavior: the agent finished silently with 0 new utterances, ignoring the user's "Are you done?"
    const visible = history.filter((e) => e.type === "own_utterance");
    assertEquals(visible.length, 0); // Exits silently!
  },
);
