import { assert, assertEquals } from "@std/assert";
import { z } from "zod/v4";
import {
  injectCallModelWrapper,
  runAgent,
  tool,
  turnTimeBudgetWrapUpThought,
} from "../mod.ts";
import {
  forcedStopUtterance,
  type HistoryEvent,
  injectCallModel,
  ownUtteranceTurn,
  participantUtteranceTurn,
  stopThoughtDefault,
  toolUseTurn,
} from "../src/agent.ts";
import { injectGeminiToken } from "../src/gemini.ts";
import { genJsonOverride } from "../src/genJson.ts";
import { agentDeps, runForAllProviders } from "../test_helpers.ts";

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

Deno.test(
  "progress check gate stops runaway tool loops on compacted/external-event histories without emitting unsolicited user utterance",
  async () => {
    const history: HistoryEvent[] = [
      {
        id: "summary-1000-2000",
        type: "own_thought",
        isOwn: true,
        timestamp: 1000,
        text:
          "[This summary covers the period from Jan 1, 2026, 10:00 AM to Jan 1, 2026, 10:15 AM]\n\nPast conversation history was compacted into a structured summary.",
      },
      {
        id: "ext-1",
        type: "external_event",
        isOwn: false,
        timestamp: 2500,
        text: "A background VM command finished with exit code 0.",
      },
    ];

    const noopTool = tool({
      name: "read_chunk",
      description: "Read a chunk of output",
      parameters: z.object({ offset: z.number() }),
      handler: ({ offset }) => Promise.resolve(`chunk at ${offset}`),
    });

    const counts = { model: 0, audit: 0 };
    const seenInputs: HistoryEvent[][] = [];

    await injectGeminiToken("fake-token")(
      genJsonOverride.inject(
        () =>
        (_opts: unknown, _sys: string, _zod: unknown) =>
        (): Promise<{ shouldContinue: boolean }> => {
          counts.audit++;
          return Promise.resolve({ shouldContinue: false });
        },
      )(
        injectCallModel((events) => {
          counts.model++;
          seenInputs.push(events);
          return Promise.resolve([
            toolUseTurn({
              name: "read_chunk",
              args: { offset: counts.model },
            }),
          ]);
        })(async () => {
          await agentDeps(history)(runAgent)({
            maxIterations: 2,
            tools: [noopTool],
            prompt: "You are a background worker thread.",
            timezoneIANA: "UTC",
          });
        }),
      ),
    )();

    assertEquals(
      counts.audit,
      2,
      "progress auditor must run on compacted/external-event loops",
    );
    assertEquals(
      counts.model,
      2,
      "run must terminate after soft stop + escalated stop",
    );
    assert(
      seenInputs[1].some((e) =>
        e.type === "own_thought" && e.text.startsWith(stopThoughtDefault)
      ),
      "second model call must receive the soft-stop thought",
    );
    assertEquals(
      history.some((e) =>
        e.type === "own_utterance" && e.text === forcedStopUtterance
      ),
      false,
      "must not emit unsolicited forcedStopUtterance when no user message is unanswered",
    );
  },
);

Deno.test(
  "maxTurnDurationMs injects wrap-up thought when time budget expires and allows model to conclude cleanly",
  async () => {
    const history: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Investigate the ingestion spike and summarize findings.",
      }),
    ];

    const slowQueryTool = tool({
      name: "run_query",
      description: "Run an analytics query",
      parameters: z.object({ q: z.string() }),
      handler: async ({ q }) => {
        await delay(30);
        return `rows for ${q}`;
      },
    });

    const state = { calls: 0 };
    await injectCallModel((events) => {
      state.calls++;
      const sawTimeBudgetStop = events.some((e) =>
        e.type === "own_thought" && e.text === turnTimeBudgetWrapUpThought
      );
      if (sawTimeBudgetStop) {
        return Promise.resolve([
          ownUtteranceTurn("Summary of findings based on completed queries."),
        ]);
      }
      return Promise.resolve([
        toolUseTurn({
          name: "run_query",
          args: { q: `select ${state.calls}` },
        }),
      ]);
    })(async () => {
      await agentDeps(history)(runAgent)({
        maxIterations: 30,
        maxTurnDurationMs: 45,
        tools: [slowQueryTool],
        prompt: "You are an investigator.",
        timezoneIANA: "UTC",
      });
    })();

    assertEquals(
      state.calls,
      3,
      "model should get one wrap-up call once time budget expires",
    );
    const replies = history.filter((e) => e.type === "own_utterance");
    assertEquals(replies.length, 1);
    assert(
      replies[0].type === "own_utterance" &&
        replies[0].text.includes("Summary of findings"),
    );
  },
);

Deno.test(
  "maxTurnDurationMs force-stops runaway tool loop if model ignores wrap-up thought",
  async () => {
    const history: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Keep digging forever.",
      }),
    ];

    const slowQueryTool = tool({
      name: "run_query",
      description: "Run an analytics query",
      parameters: z.object({ q: z.string() }),
      handler: async ({ q }) => {
        await delay(30);
        return `rows for ${q}`;
      },
    });

    const state = { calls: 0 };
    await injectCallModel(() => {
      state.calls++;
      return Promise.resolve([
        toolUseTurn({
          name: "run_query",
          args: { q: `select ${state.calls}` },
        }),
      ]);
    })(async () => {
      await agentDeps(history)(runAgent)({
        maxIterations: 30,
        maxTurnDurationMs: 45,
        tools: [slowQueryTool],
        prompt: "You are an investigator.",
        timezoneIANA: "UTC",
      });
    })();

    assertEquals(
      state.calls,
      3,
      "run must force-stop on the iteration after the wrap-up thought is ignored",
    );
    assert(
      history.some((e) =>
        e.type === "own_utterance" && e.text === forcedStopUtterance
      ),
      "unanswered user turn must receive forcedStopUtterance on forced stop",
    );
  },
);

Deno.test(
  "maxTurnDurationMs disables tools on the wrap-up call once time budget expires",
  async () => {
    const history: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Investigate the latency spike and summarize findings.",
      }),
    ];

    const slowQueryTool = tool({
      name: "run_query",
      description: "Run an analytics query",
      parameters: z.object({ q: z.string() }),
      handler: async ({ q }) => {
        await delay(30);
        return `rows for ${q}`;
      },
    });

    const seenDisableTools: boolean[] = [];
    const state = { calls: 0 };
    await injectCallModelWrapper(({ disableTools, inner }) => (events) => {
      seenDisableTools.push(Boolean(disableTools));
      return inner(events);
    })(
      injectCallModel((events) => {
        state.calls++;
        const sawTimeBudgetStop = events.some((e) =>
          e.type === "own_thought" && e.text === turnTimeBudgetWrapUpThought
        );
        if (sawTimeBudgetStop) {
          return Promise.resolve([
            ownUtteranceTurn("Final summary based on completed queries."),
          ]);
        }
        return Promise.resolve([
          toolUseTurn({
            name: "run_query",
            args: { q: `select ${state.calls}` },
          }),
        ]);
      })(async () => {
        await agentDeps(history)(runAgent)({
          maxIterations: 30,
          maxTurnDurationMs: 45,
          tools: [slowQueryTool],
          prompt: "You are an investigator.",
          timezoneIANA: "UTC",
        });
      }),
    )();

    assertEquals(
      seenDisableTools,
      [false, false, true],
      "wrap-up model call must set disableTools=true while earlier calls allow tools",
    );
  },
);

runForAllProviders(
  "maxTurnDurationMs structurally disables tools on wrap-up so real models conclude with findings instead of calling another tool",
  async (runAgentWithProvider) => {
    const history: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Fetch all error log pages and summarize the errors found.",
      }),
    ];

    const fetchedPages: number[] = [];
    const fetchPageTool = tool({
      name: "fetch_page",
      description: "Fetch a page of error logs",
      parameters: z.object({ page: z.number() }),
      handler: async ({ page }) => {
        fetchedPages.push(page);
        await delay(35);
        return JSON.stringify({
          page,
          errors: [`Timeout on worker-${page} (count: 7)`],
          hasMore: true,
        });
      },
    });

    await agentDeps(history)(runAgentWithProvider)({
      maxIterations: 20,
      maxTurnDurationMs: 25,
      tools: [fetchPageTool],
      prompt:
        "You are an incident investigator. Whenever fetch_page returns hasMore: true and you still have tools available, you MUST call fetch_page for the next page (even if an internal thought suggests stopping or wrapping up). Only summarize partial findings if tools are disabled or unavailable.",
      timezoneIANA: "UTC",
    });

    assertEquals(
      fetchedPages,
      [1],
      `Expected model to stop calling tools on wrap-up, but fetched pages: ${
        JSON.stringify(fetchedPages)
      }`,
    );
    const replies = history.filter((e) => e.type === "own_utterance");
    assert(replies.length >= 1, "expected a user-facing summary reply");
    assertEquals(
      replies.some((e) =>
        e.type === "own_utterance" && e.text === forcedStopUtterance
      ),
      false,
      "must not fall back to generic forcedStopUtterance when wrap-up disables tools",
    );
  },
);
