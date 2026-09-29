import { assert, assertEquals } from "@std/assert";
import { auditUtteranceForHallucination, runAgent } from "../mod.ts";
import {
  type HistoryEvent,
  injectCallModel,
  ownThoughtTurn,
  ownUtteranceTurn,
  participantUtteranceTurn,
} from "../src/agent.ts";
import { injectDecisionModel } from "../src/decisionModel.ts";
import { agentDeps, injectSecrets } from "../test_helpers.ts";

Deno.test(
  "hallucination gate - intercepts off-topic utterance and re-invokes model with correctional thought",
  async () => {
    const userQuery = "What is my flight number?";
    const history: HistoryEvent[] = [
      participantUtteranceTurn({ name: "user", text: userQuery }),
    ];
    let callCount = 0;
    const seenThoughts: string[] = [];

    // Scripted model: initially hallucinates about invoices, then course-corrects on retry
    const scriptedModel = (events: HistoryEvent[]) => {
      callCount++;
      const thoughts = events
        .filter((e) => e.type === "own_thought")
        .map((e) => ("text" in e && typeof e.text === "string" ? e.text : ""));
      seenThoughts.push(...thoughts);

      if (callCount === 1) {
        return Promise.resolve([
          ownUtteranceTurn(
            "I checked your invoice: No VAT was charged and the total is $3,756.",
          ),
        ]);
      }
      return Promise.resolve([
        ownUtteranceTurn("Your flight number is IZ 595."),
      ]);
    };

    // Decision model mock: flags the first invoice utterance as hallucination, passes the second
    const mockDecisionModel = (
      state: unknown,
      _questions: Record<string, unknown>,
    ) => {
      const stateObj = state as { assistant_response?: string };
      const isInvoice = Boolean(
        stateObj?.assistant_response?.includes("invoice"),
      );
      return Promise.resolve({
        is_hallucination: {
          type: "choice" as const,
          choice: isInvoice ? "true" : "false",
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
          enableHallucinationAudit: true,
        });
      }),
    )();

    assertEquals(
      callCount,
      2,
      "Model should be re-invoked after hallucination is blocked",
    );
    assert(
      seenThoughts.some((t) =>
        t.includes(
          "SYSTEM AUDIT: Your proposed response was flagged as an off-topic hallucination",
        ) &&
        t.includes(userQuery)
      ),
      "Model must receive correctional thought with user query",
    );

    const emittedUtterances = history.filter((e) => e.type === "own_utterance");
    assertEquals(
      emittedUtterances.length,
      1,
      "Only the final non-hallucinated utterance should be emitted to user",
    );
    assertEquals(
      emittedUtterances[0].text,
      "Your flight number is IZ 595.",
      "Emitted message must be the corrected flight number",
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
          enableHallucinationAudit: true,
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
    const mockDecisionModel = () => {
      decisionModelCalled = true;
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
          enableHallucinationAudit: true,
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
    const mockDecisionModel = () => {
      decisionModelCalled = true;
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
          enableHallucinationAudit: true,
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
  "hallucination gate - default spec skips hallucination audit without decision model calls or retry loops",
  async () => {
    const userQuery = "What is my flight number?";
    const history: HistoryEvent[] = [
      participantUtteranceTurn({ name: "user", text: userQuery }),
    ];
    let callCount = 0;
    let decisionModelCalled = false;

    const scriptedModel = () => {
      callCount++;
      return Promise.resolve([
        ownUtteranceTurn("Here is the answer."),
      ]);
    };

    const mockDecisionModel = () => {
      decisionModelCalled = true;
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
          prompt: "You are a travel assistant.",
          tools: [],
          timezoneIANA: "UTC",
        });
      }),
    )();

    assertEquals(callCount, 1);
    assertEquals(decisionModelCalled, false);
    const emitted = history.filter((e) => e.type === "own_utterance");
    assertEquals(emitted.length, 1);
    assertEquals(emitted[0].text, "Here is the answer.");
  },
);

Deno.test(
  "auditUtteranceForHallucination - live decision model evaluation",
  injectSecrets(async () => {
    // 1. Off-topic response should be flagged
    const isBad = await auditUtteranceForHallucination(
      "What is the flight number that we booked with Arkia?",
      "I reviewed the invoice: No VAT was charged, total is $3,756 paid via credit card and BUYME vouchers.",
      "Flight booking confirmed: Outbound IZ 595, Inbound IZ 690.",
    );
    assertEquals(
      isBad,
      true,
      "Off-topic invoice response must be flagged as hallucination",
    );

    // 2. Direct answer should NOT be flagged
    const isGood = await auditUtteranceForHallucination(
      "What is the flight number that we booked with Arkia?",
      "Your flight numbers are IZ 595 (outbound) and IZ 690 (return).",
      "Flight booking confirmed: Outbound IZ 595, Inbound IZ 690.",
    );
    assertEquals(
      isGood,
      false,
      "Direct flight number answer must not be flagged",
    );
  }),
);
