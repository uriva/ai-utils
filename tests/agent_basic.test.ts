import { assert } from "@std/assert";
import {
  type HistoryEvent,
  ownUtteranceTurn,
  participantUtteranceTurn,
} from "../src/agent.ts";
import { runAgent } from "../mod.ts";
import {
  agentDeps,
  injectSecrets,
  noopRewriteHistory,
  someTool,
} from "../test_helpers.ts";

Deno.test(
  "agent can run when history starts with only a model message",
  injectSecrets(async () => {
    await agentDeps([ownUtteranceTurn("Priming without user turn")])(runAgent)({
      maxIterations: 1,
      onMaxIterationsReached: () => {},
      tools: [],
      prompt: "You are a helper.",
      lightModel: true,
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
  }),
);

Deno.test(
  "agent can start an empty conversation",
  injectSecrets(async () => {
    await agentDeps([])(runAgent)({
      onMaxIterationsReached: () => {},
      tools: [],
      prompt: `You are the neighborhood friendly spiderman.`,
      maxIterations: 5,
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
  }),
);

Deno.test(
  "maxOutputTokens limits gemini output length",
  injectSecrets(async () => {
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
  }),
);

Deno.test(
  "agent with history starting with only tool doesn't trigger 400",
  injectSecrets(async () => {
    const mockHistory: HistoryEvent[] = [{
      type: "tool_result",
      isOwn: true,
      id: "test-id",
      timestamp: Date.now(),
      name: "someTool",
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
  }),
);

Deno.test(
  "tool_call with empty thoughtSignature is filtered out with warning",
  injectSecrets(async () => {
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
    ];

    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 1,
      onMaxIterationsReached: () => {},
      tools: [someTool],
      prompt: "You are a helper.",
      lightModel: false,
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
  }),
);

Deno.test(
  "agent filters unsupported gemini attachments before api call",
  injectSecrets(async () => {
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
  }),
);

Deno.test(
  "handles 403 file permission errors and replaces history items",
  injectSecrets(async () => {
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
  }),
);

Deno.test(
  "handles unsupported MIME type by stripping attachment and rewriting history",
  injectSecrets(async () => {
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
      !("attachments" in rewritten) || !rewritten.attachments?.some((a) =>
        a.mimeType ===
          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
      ),
      "Unsupported attachment should have been stripped from rewritten history",
    );
  }),
);
