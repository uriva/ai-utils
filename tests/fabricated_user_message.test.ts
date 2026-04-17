import { assert } from "@std/assert";
import {
  type HistoryEvent,
  ownUtteranceTurn,
  participantUtteranceTurn,
  sanitizeModelOutput,
} from "../src/agent.ts";
import { agentDeps, runForAllProviders } from "../test_helpers.ts";

const participantName = "אורח/ת";

const buildHistory = (): HistoryEvent[] => [
  participantUtteranceTurn({
    name: participantName,
    text: "שלום, אני מחפש תנור אפייה חדש למטבח",
  }),
  ownUtteranceTurn(
    "שלום! אשמח לעזור לך למצוא תנור אפייה. יש לך העדפות מסוימות? תקציב? גודל?",
  ),
  participantUtteranceTurn({
    name: participantName,
    text: "תנור בילד-אין, תקציב עד 5000 שקל",
  }),
  ownUtteranceTurn(
    "מעולה! יש לנו כמה אפשרויות בטווח המחירים הזה. למשל, תנור Bosch HBG5780S6 ב-4,500 ₪, או Electrolux EOD5H70X ב-3,800 ₪.",
  ),
  participantUtteranceTurn({
    name: participantName,
    text: "מה ההבדל ביניהם?",
  }),
  ownUtteranceTurn(
    "ל-Bosch יש 71 ליטר נפח ו-13 תכניות אפייה, עם ניקוי פירוליטי. ל-Electrolux יש 72 ליטר עם 10 תכניות וניקוי קטליטי. שניהם איכותיים מאוד.",
  ),
];

runForAllProviders(
  "guard strips fabricated user messages from model output that mimics participant format",
  async (runAgentWithProvider) => {
    const mockHistory = buildHistory();
    await agentDeps(mockHistory)(runAgentWithProvider)({
      maxIterations: 1,
      onMaxIterationsReached: () => {},
      tools: [],
      prompt:
        "You are a sales assistant for a home appliance store. Continue the conversation naturally in Hebrew. " +
        "The customer is asking about ovens. Be helpful and provide recommendations.",
      rewriteHistory: async () => {},
      timezoneIANA: "Asia/Jerusalem",
    });

    const newEvents = mockHistory.slice(buildHistory().length);

    const fabricatedUserMessage = newEvents.some((e) =>
      e.type === "own_utterance" &&
      new RegExp(`^${participantName}:\\s`, "m").test(e.text)
    );
    assert(
      !fabricatedUserMessage,
      `Guard failed to strip fabricated user message from model output:\n${
        JSON.stringify(
          newEvents.filter((e) => e.type === "own_utterance"),
          null,
          2,
        )
      }`,
    );
  },
);

Deno.test("sanitizeModelOutput self-corrects fabricated user message with timestamp", () => {
  const history = [
    participantUtteranceTurn({ name: participantName, text: "שלום" }),
    ownUtteranceTurn("שלום! איך אני יכול לעזור?"),
    participantUtteranceTurn({
      name: participantName,
      text: "אני מחפש תנור חדש",
    }),
  ];
  const output = [
    ownUtteranceTurn(
      `${participantName}: לא ממש משנה לי המחיר, משהו איכותי. שמעתי על Miele שהם טובים — sent Mar 30, 2026, 3:12 PM`,
    ),
  ];
  const result = sanitizeModelOutput(
    history,
    output,
  );
  assert(
    result.emit.every((e) =>
      e.type !== "own_utterance" ||
      !new RegExp(`^${participantName}:\\s`, "m").test(e.text)
    ),
    "Guard should have stripped or reclassified fabricated user message",
  );
});

Deno.test("sanitizeModelOutput preserves legitimate response mixed with fabricated line", () => {
  const history = [
    participantUtteranceTurn({ name: "user", text: "tell me about ovens" }),
  ];
  const output = [
    ownUtteranceTurn(
      "user: I want a big oven\nSure! Here are some great options for large ovens.",
    ),
  ];
  const result = sanitizeModelOutput(
    history,
    output,
  );
  assert(result.emit.length === 1);
  const event = result.emit[0];
  assert(event.type === "own_utterance");
  if (event.type === "own_utterance") {
    assert(
      !event.text.startsWith("user:"),
      "Fabricated user line should have been stripped",
    );
    assert(
      event.text.includes("great options"),
      "Legitimate response should be preserved",
    );
  }
});
