import { assert, assertEquals } from "@std/assert";
import {
  auditUtteranceForHallucination,
  runAgent,
  safetyWarningText,
} from "../mod.ts";
import {
  type HistoryEvent,
  injectCallModel,
  ownThoughtTurn,
  ownUtteranceTurn,
  ownUtteranceTurnWithMetadata,
  participantUtteranceTurn,
  toolResultTurn,
} from "../src/agent.ts";
import { injectDecisionModel } from "../src/decisionModel.ts";
import { agentDeps, injectSecrets } from "../test_helpers.ts";

Deno.test(
  "runAgent delivers utterance directly without hallucination gate re-invocation",
  async () => {
    const userQuery = "What is my flight number?";
    const history: HistoryEvent[] = [
      participantUtteranceTurn({ name: "user", text: userQuery }),
    ];
    let callCount = 0;
    const seenThoughts: string[] = [];

    const scriptedModel = (events: HistoryEvent[]) => {
      callCount++;
      const thoughts = events
        .filter((e) => e.type === "own_thought")
        .map((e) => ("text" in e && typeof e.text === "string" ? e.text : ""));
      seenThoughts.push(...thoughts);

      return Promise.resolve([
        ownUtteranceTurn("Your flight number is IZ 595."),
      ]);
    };

    let decisionModelCalled = false;
    const mockDecisionModel = (
      _state: unknown,
      questions: Record<string, unknown>,
    ) => {
      if (questions && "is_hallucination" in questions) {
        decisionModelCalled = true;
      }
      return Promise.resolve({
        is_hallucination: {
          type: "choice" as const,
          choice: "false",
        },
      });
    };

    await injectDecisionModel(mockDecisionModel)(
      injectCallModel(scriptedModel)(async () => {
        await agentDeps(history)(runAgent)({
          maxIterations: 3,
          prompt: "You are a travel assistant.",
          tools: [],
          timezoneIANA: "UTC",
        });
      }),
    )();

    assertEquals(
      callCount,
      1,
      "Model should only be called once without being blocked by hallucination gate",
    );
    assertEquals(
      decisionModelCalled,
      false,
      "Decision model hallucination check should not be called",
    );
    assert(
      !seenThoughts.some((t) =>
        t.includes(
          "SYSTEM AUDIT: Your proposed response was flagged as an off-topic hallucination",
        )
      ),
    );
  },
);

Deno.test(
  "hallucination gate - allows valid responses to pass without re-invoking model",
  async () => {
    const history: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "What is my flight number?",
      }),
    ];
    let callCount = 0;

    const scriptedModel = () => {
      callCount++;
      return Promise.resolve([
        ownUtteranceTurn("Your flight number is IZ 595."),
      ]);
    };

    const mockDecisionModel = () =>
      Promise.resolve({
        is_hallucination: {
          type: "choice" as const,
          choice: "false",
        },
      });

    await injectDecisionModel(mockDecisionModel)(
      injectCallModel(scriptedModel)(async () => {
        await agentDeps(history)(runAgent)({
          maxIterations: 3,
          prompt: "You are a travel assistant.",
          tools: [],
          timezoneIANA: "UTC",
        });
      }),
    )();

    assertEquals(
      callCount,
      1,
      "Model should only be called once when response is valid",
    );
    const emitted = history.filter((e) => e.type === "own_utterance");
    assertEquals(emitted.length, 1);
    assertEquals(emitted[0].text, "Your flight number is IZ 595.");
  },
);

Deno.test(
  "hallucination gate - does not evaluate scheduled proactive tasks against stale user messages",
  async () => {
    const history: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Here is a photo of my dinner",
      }),
      ownUtteranceTurn("Looks delicious! Enjoy your evening!"),
      ownThoughtTurn("PROACTIVE TASK: Send a warm morning greeting."),
    ];
    let callCount = 0;

    const scriptedModel = () => {
      callCount++;
      return Promise.resolve([
        ownUtteranceTurn(
          "Good morning! Wishing you a wonderful and energetic day ahead!",
        ),
      ]);
    };

    let decisionModelCalled = false;
    const mockDecisionModel = (
      _state?: unknown,
      questions?: Record<string, unknown>,
    ) => {
      if (questions && "is_hallucination" in questions) {
        decisionModelCalled = true;
      }
      return Promise.resolve({
        is_hallucination: {
          type: "choice" as const,
          choice: "true",
        },
      });
    };

    await injectDecisionModel(mockDecisionModel)(
      injectCallModel(scriptedModel)(async () => {
        await agentDeps(history)(runAgent)({
          maxIterations: 3,
          prompt: "You are a personal assistant.",
          tools: [],
          timezoneIANA: "UTC",
        });
      }),
    )();

    assertEquals(
      decisionModelCalled,
      false,
      "Decision model should NOT be called to audit a proactive task against stale user messages",
    );
    assertEquals(
      callCount,
      1,
      "Model should only be called once without being blocked by hallucination gate",
    );
    const emitted = history.filter((e) => e.type === "own_utterance");
    assertEquals(emitted.length, 2);
    assertEquals(
      emitted[1].text,
      "Good morning! Wishing you a wonderful and energetic day ahead!",
    );
  },
);

Deno.test(
  "hallucination gate - does not evaluate proactive tasks when intervening platform task follows unanswered user message",
  async () => {
    const history: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Here is a photo of my dinner",
      }),
      ownThoughtTurn("PROACTIVE TASK: Send a warm morning greeting."),
    ];
    let callCount = 0;

    const scriptedModel = () => {
      callCount++;
      return Promise.resolve([
        ownUtteranceTurn(
          "Good morning! Wishing you a wonderful and energetic day ahead!",
        ),
      ]);
    };

    let decisionModelCalled = false;
    const mockDecisionModel = (
      _state?: unknown,
      questions?: Record<string, unknown>,
    ) => {
      if (questions && "is_hallucination" in questions) {
        decisionModelCalled = true;
      }
      return Promise.resolve({
        is_hallucination: {
          type: "choice" as const,
          choice: "true",
        },
      });
    };

    await injectDecisionModel(mockDecisionModel)(
      injectCallModel(scriptedModel)(async () => {
        await agentDeps(history)(runAgent)({
          maxIterations: 3,
          prompt: "You are a personal assistant.",
          tools: [],
          timezoneIANA: "UTC",
        });
      }),
    )();

    assertEquals(
      decisionModelCalled,
      false,
      "Decision model should NOT be called to audit a proactive task even if past user message had no reply",
    );
    assertEquals(
      callCount,
      1,
      "Model should only be called once without being blocked by hallucination gate",
    );
    const emitted = history.filter((e) => e.type === "own_utterance");
    assertEquals(emitted.length, 1);
    assertEquals(
      emitted[0].text,
      "Good morning! Wishing you a wonderful and energetic day ahead!",
    );
  },
);

Deno.test(
  "hallucination gate - sends exact projected history events to decision model without false positives",
  async () => {
    const rawUserMsg =
      '[replying to you: "בדקתי, ולשבוע הקרוב אין כרגע במאגר מסיבות"]\nמסיבות 30+ במרכז';
    const history: HistoryEvent[] = [
      participantUtteranceTurn({ name: "user", text: rawUserMsg }),
    ];
    let callCount = 0;
    let auditedEvents: unknown[] | undefined;

    const scriptedModel = () => {
      callCount++;
      return Promise.resolve([
        ownUtteranceTurn("הנה מסיבות במרכז: Oktoberfest Beerfest."),
      ]);
    };

    const mockDecisionModel = (state: unknown) => {
      const stateObj = state as { conversation_history_events?: unknown[] };
      auditedEvents = stateObj?.conversation_history_events;
      return Promise.resolve({
        is_hallucination: {
          type: "choice" as const,
          choice: "false",
        },
      });
    };

    await injectDecisionModel(mockDecisionModel)(
      injectCallModel(scriptedModel)(async () => {
        await agentDeps(history)(runAgent)({
          maxIterations: 3,
          prompt: "You are an events guide.",
          tools: [],
          timezoneIANA: "UTC",
        });
      }),
    )();

    assertEquals(callCount, 1);
    assertEquals(auditedEvents, undefined);
  },
);

Deno.test(
  "auditUtteranceForHallucination - live decision model evaluation",
  injectSecrets(async () => {
    // 1. Off-topic response should be flagged
    const isBad = await auditUtteranceForHallucination(
      [participantUtteranceTurn({
        name: "user",
        text: "What is the flight number that we booked with Arkia?",
      })],
      "I reviewed the invoice: No VAT was charged, total is $3,756 paid via credit card and BUYME vouchers.",
    );
    assertEquals(
      isBad,
      true,
      "Off-topic invoice response must be flagged as hallucination",
    );

    // 2. Direct answer should NOT be flagged
    const isGood = await auditUtteranceForHallucination(
      [
        participantUtteranceTurn({
          name: "user",
          text: "What is the flight number that we booked with Arkia?",
        }),
        toolResultTurn({
          toolCallId: "call_1",
          result: "Flight booking confirmed: Outbound IZ 595, Inbound IZ 690.",
        }),
      ],
      "Your flight numbers are IZ 595 (outbound) and IZ 690 (return).",
    );
    assertEquals(
      isGood,
      false,
      "Direct flight number answer must not be flagged",
    );
  }),
);

Deno.test(
  "hallucination gate - does not audit or retry safety block utterances",
  async () => {
    const history: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Generate prohibited content",
      }),
    ];
    let callCount = 0;

    const scriptedModel = () => {
      callCount++;
      return Promise.resolve([
        ownUtteranceTurnWithMetadata(
          safetyWarningText,
          { isSafetyBlock: true },
        ),
      ]);
    };

    let decisionModelCalled = false;
    const mockDecisionModel = (
      _state?: unknown,
      questions?: Record<string, unknown>,
    ) => {
      if (questions && "is_hallucination" in questions) {
        decisionModelCalled = true;
      }
      return Promise.resolve({
        is_hallucination: {
          type: "choice" as const,
          choice: "true",
        },
      });
    };

    await injectDecisionModel(mockDecisionModel)(
      injectCallModel(scriptedModel)(async () => {
        await agentDeps(history)(runAgent)({
          maxIterations: 3,
          prompt: "You are a helpful assistant.",
          tools: [],
          timezoneIANA: "UTC",
        });
      }),
    )();

    assertEquals(
      decisionModelCalled,
      false,
      "Decision model should never be called for safety block utterances",
    );
    assertEquals(
      callCount,
      1,
      "Model should only be called once when safety block occurs without retrying",
    );
    const emitted = history.filter((e) => e.type === "own_utterance");
    assertEquals(emitted.length, 1);
    assertEquals(emitted[0].text, safetyWarningText);
  },
);
