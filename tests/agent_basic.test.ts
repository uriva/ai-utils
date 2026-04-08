import { assert, assertEquals } from "@std/assert";
import {
  doNothingEvent,
  type HistoryEvent,
  ownUtteranceTurn,
  participantUtteranceTurn,
} from "../src/agent.ts";
import {
  agentDeps,
  noopRewriteHistory,
  runForBothProviders,
  someTool,
} from "../test_helpers.ts";

runForBothProviders(
  "agent can run when history starts with only a model message",
  async (runAgent) => {
    await agentDeps([ownUtteranceTurn("Priming without user turn")])(runAgent)({
      maxIterations: 1,
      onMaxIterationsReached: () => {},
      tools: [],
      prompt: "You are a helper.",
      lightModel: true,
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
  },
);

runForBothProviders(
  "agent can start an empty conversation",
  async (runAgent) => {
    await agentDeps([])(runAgent)({
      onMaxIterationsReached: () => {},
      tools: [],
      prompt: `You are the neighborhood friendly spiderman.`,
      maxIterations: 5,
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
  },
);

runForBothProviders(
  "maxOutputTokens limits output length",
  async (runAgent) => {
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Write the full alphabet from A to Z",
      }),
    ];
    await agentDeps(mockHistory)(runAgent)({
      onMaxIterationsReached: () => {},
      tools: [],
      prompt: "You are a helpful assistant that writes the full alphabet.",
      maxIterations: 1,
      maxOutputTokens: 2,
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
    const ownUtterance = mockHistory.find((e) => e.type === "own_utterance");
    assert(
      !ownUtterance?.text || ownUtterance.text.length <= 2,
      `Expected at most 2 characters but got: "${ownUtterance?.text}"`,
    );
  },
);

runForBothProviders(
  "agent with history starting with only tool doesn't trigger 400",
  async (runAgent) => {
    const mockHistory: HistoryEvent[] = [{
      type: "tool_result",
      isOwn: true,
      id: "test-id",
      timestamp: Date.now(),
      result: "some result",
    }];
    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 1,
      onMaxIterationsReached: () => {},
      tools: [],
      prompt: "You are a helper.",
      lightModel: true,
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
  },
  3,
  true, // Gemini-only: test uses orphaned tool_result which Kimi rejects
);

runForBothProviders(
  "tool_call with empty thoughtSignature is filtered along with its tool_result",
  async (runAgent) => {
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Please call the testTool.",
      }),
      {
        type: "tool_call",
        isOwn: true,
        id: "test-id",
        timestamp: Date.now(),
        name: "testTool",
        parameters: {},
        modelMetadata: {
          type: "gemini",
          thoughtSignature: "",
          responseId: "resp_id",
        },
      } as HistoryEvent,
      {
        type: "tool_result",
        isOwn: true,
        id: "result-id",
        timestamp: Date.now(),
        toolCallId: "test-id",
        result: "tool result",
      } as HistoryEvent,
    ];

    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 1,
      onMaxIterationsReached: () => {},
      tools: [someTool],
      prompt: "You are a helper.",
      lightModel: true,
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
  },
  3,
  true, // Gemini-only: test uses Gemini-specific thoughtSignature metadata
);

runForBothProviders(
  "tool_call with empty thoughtSignature filters out other events from the same responseId",
  async (runAgent) => {
    let rewriteReplacements: Record<string, HistoryEvent> = {};
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Please call the testTool and say something.",
      }),
      {
        type: "tool_call",
        isOwn: true,
        id: "test-id",
        timestamp: Date.now(),
        name: "testTool",
        parameters: {},
        modelMetadata: {
          type: "gemini",
          thoughtSignature: "",
          responseId: "resp_id",
        },
      } as HistoryEvent,
      {
        type: "own_utterance",
        isOwn: true,
        id: "utterance-id",
        timestamp: Date.now(),
        text: "I am going to call the tool.",
        modelMetadata: {
          type: "gemini",
          thoughtSignature: "",
          responseId: "resp_id",
        },
      } as HistoryEvent,
      {
        type: "tool_result",
        isOwn: true,
        id: "result-id",
        timestamp: Date.now(),
        toolCallId: "test-id",
        result: "tool result",
      } as HistoryEvent,
    ];

    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 1,
      onMaxIterationsReached: () => {},
      tools: [someTool],
      prompt: "You are a helper.",
      lightModel: true,
      rewriteHistory: (replacements) => {
        rewriteReplacements = replacements;
        return Promise.resolve();
      },
      timezoneIANA: "UTC",
    });

    assertEquals(Object.keys(rewriteReplacements).length, 3);
    assertEquals(rewriteReplacements["test-id"].type, "own_thought");
    assertEquals(rewriteReplacements["utterance-id"].type, "own_thought");
    assertEquals(rewriteReplacements["result-id"].type, "own_thought");
    assert(
      ("text" in rewriteReplacements["utterance-id"]
        ? rewriteReplacements["utterance-id"].text
        : "")?.includes(
          "Removed own_utterance from response containing invalid tool call: I am going to call the tool.",
        ),
    );
  },
  3,
  true, // Gemini-only: test uses Gemini-specific thoughtSignature metadata
);

runForBothProviders(
  "agent filters unsupported attachments before api call",
  async (runAgent) => {
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Describe the text message only.",
        attachments: [
          {
            kind: "inline",
            mimeType: "application/octet-stream",
            dataBase64: "dGVzdA==",
          },
        ],
      }),
    ];
    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 1,
      onMaxIterationsReached: () => {},
      tools: [],
      prompt: "You are a helper.",
      lightModel: true,
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
  },
);

runForBothProviders(
  "handles 403 file permission errors and replaces history items",
  async (runAgent) => {
    const replacedItems = new Map<string, HistoryEvent>();
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Here's an image",
        attachments: [
          {
            kind: "file",
            mimeType: "image/png",
            fileUri:
              "https://generativelanguage.googleapis.com/v1beta/files/2opdg5pjmw67",
            caption: "Test image",
          },
        ],
      }),
      participantUtteranceTurn({
        name: "user",
        text: "What do you see?",
      }),
    ];

    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 5,
      onMaxIterationsReached: () => {},
      tools: [],
      prompt: "You are a helpful assistant.",
      lightModel: true,
      rewriteHistory: (
        replacements: Record<string, HistoryEvent>,
      ) => {
        Object.entries(replacements).forEach(([id, newItem]) => {
          replacedItems.set(id, newItem);
          const index = mockHistory.findIndex((e) => e.id === id);
          if (index !== -1) {
            mockHistory[index] = newItem;
          }
        });
        return Promise.resolve();
      },
      timezoneIANA: "UTC",
    });

    assert(true, "Agent completed successfully");
  },
);

runForBothProviders(
  "handles unsupported MIME type by stripping attachment and rewriting history",
  async (runAgent) => {
    const replacedItems = new Map<string, HistoryEvent>();
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Summarize this spreadsheet.",
        attachments: [
          {
            kind: "inline",
            mimeType:
              "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            dataBase64: "dGVzdA==",
          },
        ],
      }),
    ];

    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 1,
      onMaxIterationsReached: () => {},
      tools: [],
      prompt: "You are a helpful assistant.",
      lightModel: true,
      rewriteHistory: (
        replacements: Record<string, HistoryEvent>,
      ) => {
        Object.entries(replacements).forEach(([id, newItem]) => {
          replacedItems.set(id, newItem);
          const index = mockHistory.findIndex((e) => e.id === id);
          if (index !== -1) {
            mockHistory[index] = newItem;
          }
        });
        return Promise.resolve();
      },
      timezoneIANA: "UTC",
    });

    assert(replacedItems.size > 0, "rewriteHistory should have been called");
    const rewritten = [...replacedItems.values()][0];
    assert(
      !("attachments" in rewritten) ||
        !rewritten.attachments?.some((a) =>
          a.mimeType ===
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ),
      "Unsupported attachment should have been stripped from rewritten history",
    );
  },
  3,
  true, // Gemini-only: test relies on Gemini-specific MIME type filtering behavior
);

runForBothProviders(
  "agent streams output chunk by chunk",
  async (runAgent) => {
    let streamedText = "";
    let chunkCount = 0;

    await agentDeps([
      participantUtteranceTurn({
        name: "user",
        text: "Please write a short 20-word story about a brave knight.",
      }),
    ])(runAgent)({
      maxIterations: 1,
      onMaxIterationsReached: () => {},
      tools: [],
      prompt: "You are a creative writer.",
      onStreamChunk: (chunk) => {
        streamedText += chunk;
        chunkCount++;
      },
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });

    assert(
      chunkCount > 1,
      `Should have received multiple stream chunks (got ${chunkCount})`,
    );
    assert(streamedText.length > 20, "Streamed text should be reasonably long");
  },
);

runForBothProviders(
  "agent outputs complete text in one chunk when disableStreaming is true",
  async (runAgent) => {
    let streamedText = "";
    let chunkCount = 0;

    await agentDeps([
      participantUtteranceTurn({
        name: "user",
        text: "Please write a short 20-word story about a brave knight.",
      }),
    ])(runAgent)({
      maxIterations: 1,
      onMaxIterationsReached: () => {},
      tools: [],
      prompt: "You are a creative writer.",
      onStreamChunk: (chunk) => {
        streamedText += chunk;
        chunkCount++;
      },
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
      disableStreaming: true,
    });

    assertEquals(
      chunkCount,
      1,
      `Should have received exactly one stream chunk (got ${chunkCount})`,
    );
    assert(streamedText.length > 20, "Streamed text should be reasonably long");
  },
);

runForBothProviders(
  "handles do_nothing events in history",
  async (runAgent) => {
    const history = [
      participantUtteranceTurn({ name: "user", text: "Hello" }),
      // do_nothing event should be filtered out
      doNothingEvent({ type: "gemini", responseId: "test-id" }),
      participantUtteranceTurn({ name: "user", text: "What's up?" }),
    ];

    await agentDeps(history)(runAgent)({
      maxIterations: 1,
      onMaxIterationsReached: () => {},
      tools: [],
      prompt: "You are a helpful assistant.",
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
  },
);

runForBothProviders(
  "agent stays silent when it has nothing to say",
  async (runAgent) => {
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "Alice",
        text: "Hey Bob, what time is our meeting tomorrow?",
      }),
      participantUtteranceTurn({
        name: "Bob",
        text: "It's at 3pm, same room as last week.",
      }),
      participantUtteranceTurn({
        name: "Alice",
        text: "Great, thanks Bob!",
      }),
    ];

    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 1,
      onMaxIterationsReached: () => {},
      tools: [],
      prompt:
        "You are a silent observer in a group chat. You must never respond to messages between other people. Only respond if someone explicitly addresses you by name.",
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });

    const botUtterances = mockHistory.filter((e) => e.type === "own_utterance");
    assertEquals(
      botUtterances.length,
      0,
      `Expected no own_utterance events but found ${botUtterances.length}: ${
        botUtterances.map((e) =>
          e.type === "own_utterance" ? `"${e.text.slice(0, 60)}"` : ""
        ).join(", ")
      }`,
    );
  },
);
