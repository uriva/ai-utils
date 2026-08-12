import { assert } from "@std/assert";
import {
  type HistoryEvent,
  ownUtteranceTurn,
  participantUtteranceTurn,
  sanitizeModelOutput,
} from "../src/agent.ts";
import { agentDeps, runForAllProviders } from "../test_helpers.ts";

const participantName = "Guest";

const buildHistory = (): HistoryEvent[] => [
  participantUtteranceTurn({
    name: participantName,
    text: "Hello, I am looking for a new baking oven for my kitchen",
  }),
  ownUtteranceTurn(
    "Hello! I'd love to help you find an oven. Do you have any specific preferences? Budget? Size?",
  ),
  participantUtteranceTurn({
    name: participantName,
    text: "A built-in oven, budget up to $1500",
  }),
  ownUtteranceTurn(
    "Great! We have several options in that price range. For example, Bosch HBG5780S6 for $1200 or Electrolux EOD5H70X for $900.",
  ),
  participantUtteranceTurn({
    name: participantName,
    text: "What is the difference between them?",
  }),
  ownUtteranceTurn(
    "The Bosch has a 71-liter capacity with pyrolytic cleaning. The Electrolux has a 72-liter capacity with catalytic cleaning. Both are high quality.",
  ),
];

runForAllProviders(
  "guard strips fabricated user messages from model output that mimics participant format",
  async (runAgentWithProvider) => {
    const mockHistory = buildHistory();
    await agentDeps(mockHistory)(runAgentWithProvider)({
      maxIterations: 1,
      tools: [],
      prompt:
        "You are a sales assistant for a home appliance store. Continue the conversation naturally in English. " +
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
    participantUtteranceTurn({ name: participantName, text: "Hello" }),
    ownUtteranceTurn("Hello! How can I help you?"),
    participantUtteranceTurn({
      name: participantName,
      text: "I am looking for a new oven",
    }),
  ];
  const output = [
    ownUtteranceTurn(
      `${participantName}: Price doesn't matter to me, something high quality. I heard Miele is good — sent Mar 30, 2026, 3:12 PM`,
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
