import { assert, assertEquals } from "@std/assert";
import {
  type HistoryEvent,
  injectCallModel,
  ownThoughtTurn,
  ownUtteranceTurn,
  participantUtteranceTurn,
  runAgent,
  searchPastHistoryToolName,
  toolResultTurn,
  toolUseTurn,
} from "../mod.ts";
import { searchPastHistoryToolRaw } from "../src/historySearch.ts";
import { agentDeps } from "../test_helpers.ts";

const createSampleHistory = (): HistoryEvent[] => {
  const baseTime = 1788500000000;
  return [
    {
      ...participantUtteranceTurn({
        name: "Alice",
        text: "Can you help me book a flight to Paris for next Friday?",
      }),
      timestamp: baseTime + 1000,
    },
    {
      ...ownThoughtTurn(
        "User wants flight booking to Paris. Searching flight database.",
      ),
      timestamp: baseTime + 2000,
    },
    {
      ...toolUseTurn({
        name: "search_flights",
        args: { destination: "CDG", date: "2026-10-02" },
      }),
      id: "call-flight-search",
      timestamp: baseTime + 3000,
    },
    {
      ...toolResultTurn({
        toolCallId: "call-flight-search",
        result:
          "Found 2 options: Air France AF123 ($450, departs 08:00) and EasyJet U2456 ($210, departs 14:30).",
      }),
      timestamp: baseTime + 4000,
    },
    {
      ...ownUtteranceTurn(
        "I found two flights to Paris: AF123 for $450 and U2456 for $210.",
      ),
      timestamp: baseTime + 5000,
    },
    {
      ...participantUtteranceTurn({
        name: "Alice",
        text: "Please book flight AF123 for me.",
      }),
      timestamp: baseTime + 60000,
    },
    {
      ...toolUseTurn({
        name: "book_ticket",
        args: {
          flight: "AF123",
          passenger: "Alice",
          confirmationCode: "CONF-987654",
        },
      }),
      id: "call-book",
      timestamp: baseTime + 61000,
    },
    {
      ...toolResultTurn({
        toolCallId: "call-book",
        result:
          "Booking successful! Order confirmation code: CONF-987654. Seat 14B reserved.",
      }),
      timestamp: baseTime + 62000,
    },
    {
      ...ownUtteranceTurn(
        "Booked! Your confirmation code is CONF-987654 for flight AF123.",
      ),
      timestamp: baseTime + 63000,
    },
  ];
};

Deno.test("searchPastHistoryToolRaw - searches by keyword across utterances, thoughts, and tools", async () => {
  const history = createSampleHistory();
  const searchTool = searchPastHistoryToolRaw(() => Promise.resolve(history));

  const result = await searchTool.handler({
    query: "AF123",
    order: "asc",
    limit: 20,
    offset: 0,
  });

  assert(
    typeof result === "string" && result.includes("Found 5 matching events"),
    `Expected 5 matches for AF123, got: ${result}`,
  );
  assert(result.includes("AF123 ($450"));
  assert(result.includes("CONF-987654"));
});

Deno.test("searchPastHistoryToolRaw - case-insensitive regex search", async () => {
  const history = createSampleHistory();
  const searchTool = searchPastHistoryToolRaw(() => Promise.resolve(history));

  const result = await searchTool.handler({
    query: "(?i)easyjet",
    order: "asc",
    limit: 20,
    offset: 0,
  });

  assert(
    typeof result === "string" && result.includes("Found 1 matching events"),
    `Expected 1 match for EasyJet, got: ${result}`,
  );
  assert(result.includes("EasyJet U2456"));
});

Deno.test("searchPastHistoryToolRaw - filters by event type", async () => {
  const history = createSampleHistory();
  const searchTool = searchPastHistoryToolRaw(() => Promise.resolve(history));

  const result = await searchTool.handler({
    types: ["participant_utterance"],
    order: "asc",
    limit: 20,
    offset: 0,
  });

  assert(
    typeof result === "string" && result.includes("Found 2 matching events"),
    `Expected 2 participant utterances, got: ${result}`,
  );
  assert(result.includes("Alice"));
  assert(!result.includes("AF123 ($450"));
});

Deno.test("searchPastHistoryToolRaw - filters by time range", async () => {
  const history = createSampleHistory();
  const searchTool = searchPastHistoryToolRaw(() => Promise.resolve(history));

  const baseTime = 1788500000000;
  const startTime = new Date(baseTime + 55000).toISOString();
  const endTime = new Date(baseTime + 70000).toISOString();

  const result = await searchTool.handler({
    start_time: startTime,
    end_time: endTime,
    order: "asc",
    limit: 20,
    offset: 0,
  });

  assert(
    typeof result === "string" && result.includes("Found 4 matching events"),
    `Expected 4 events in time window, got: ${result}`,
  );
  assert(result.includes("CONF-987654"));
  assert(!result.includes("Can you help me book a flight to Paris"));
});

Deno.test("searchPastHistoryToolRaw - handles pagination with limit and offset", async () => {
  const history = createSampleHistory();
  const searchTool = searchPastHistoryToolRaw(() => Promise.resolve(history));

  const page1 = await searchTool.handler({
    order: "asc",
    limit: 3,
    offset: 0,
  });

  assert(
    typeof page1 === "string" &&
      page1.includes(
        "Found 9 matching events (showing 3 events from offset 0)",
      ),
    `Expected page 1 of 3 events, got: ${page1}`,
  );
  assert(page1.includes("6 more matching events available"));

  const page2 = await searchTool.handler({
    order: "asc",
    limit: 3,
    offset: 3,
  });

  assert(
    typeof page2 === "string" &&
      page2.includes(
        "Found 9 matching events (showing 3 events from offset 3)",
      ),
    `Expected page 2 of 3 events, got: ${page2}`,
  );
});

Deno.test("searchPastHistoryToolRaw - returns clean message when no events match", async () => {
  const history = createSampleHistory();
  const searchTool = searchPastHistoryToolRaw(() => Promise.resolve(history));

  const result = await searchTool.handler({
    query: "NonExistentKeywordXYZ",
    order: "asc",
    limit: 20,
    offset: 0,
  });

  assertEquals(
    result,
    "No past conversation events matched your search criteria.",
  );
});

Deno.test("searchPastHistoryToolRaw - rejects invalid start_time cleanly", async () => {
  const history = createSampleHistory();
  const searchTool = searchPastHistoryToolRaw(() => Promise.resolve(history));

  const result = await searchTool.handler({
    start_time: "not-a-valid-date",
    order: "asc",
    limit: 20,
    offset: 0,
  });

  assert(
    typeof result === "string" &&
      result.includes("Invalid date format for start_time"),
    `Expected date validation error, got: ${result}`,
  );
});

Deno.test("searchPastHistoryTool - agent can call tool to retrieve past facts", async () => {
  const sampleHistory = createSampleHistory();
  const history: HistoryEvent[] = [
    ...sampleHistory,
    {
      ...participantUtteranceTurn({
        name: "Alice",
        text: "What was my flight confirmation code from earlier?",
      }),
      timestamp: 1788500000000 + 120000,
    },
  ];

  let toolWasCalled = false;

  const fakeCallModel = (received: HistoryEvent[]): Promise<HistoryEvent[]> => {
    const lastEvent = received[received.length - 1];
    if (lastEvent.type === "participant_utterance") {
      return Promise.resolve([
        toolUseTurn({
          name: searchPastHistoryToolName,
          args: { query: "confirmation" },
        }),
      ]);
    }
    if (lastEvent.type === "tool_result") {
      toolWasCalled = true;
      assert(
        typeof lastEvent.result === "string" &&
          lastEvent.result.includes("CONF-987654"),
        `Tool result did not contain expected confirmation code: ${lastEvent.result}`,
      );
      return Promise.resolve([
        ownUtteranceTurn(
          "Your flight confirmation code is CONF-987654 for flight AF123.",
        ),
      ]);
    }
    return Promise.resolve([]);
  };

  await injectCallModel(fakeCallModel)(async () => {
    await agentDeps(history)(runAgent)({
      provider: "anthropic",
      maxIterations: 3,
      tools: [],
      prompt: "You are a helpful assistant.",
      timezoneIANA: "UTC",
    });
  })();

  assert(toolWasCalled, "Expected search_past_history tool to be called");
  const finalUtterance = history[history.length - 1];
  assert(
    finalUtterance.type === "own_utterance" &&
      finalUtterance.text.includes("CONF-987654"),
    `Expected final utterance with confirmation code, got: ${
      JSON.stringify(finalUtterance)
    }`,
  );
});
