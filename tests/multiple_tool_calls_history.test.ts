import { assert, assertEquals } from "@std/assert";
import type { HistoryEvent } from "../src/agent.ts";
import {
  agentDeps,
  noopRewriteHistory,
  runForAllProviders,
} from "../test_helpers.ts";

const createHistoryWithMultipleToolCalls = (): HistoryEvent[] => [
  {
    type: "participant_utterance",
    isOwn: false,
    name: "Uri",
    text: "What's the status?",
    id: "user-1",
    timestamp: 1776265180839,
  },
  {
    type: "tool_call",
    id: "tool-call-1",
    timestamp: 1776265190841,
    isOwn: true,
    name: "read_tasks",
    parameters: {},
  },
  {
    type: "own_utterance",
    isOwn: true,
    text: "Let me check the status for you.",
    id: "utterance-1",
    timestamp: 1776265190841,
  },
  {
    type: "tool_call",
    id: "tool-call-2",
    timestamp: 1776265190841,
    isOwn: true,
    name: "read_messages",
    parameters: {
      conversationId: "test",
    },
  },
  {
    type: "tool_result",
    isOwn: true,
    id: "tool-result-1",
    timestamp: 1776265192000,
    result: "Tasks: pending 3 items",
    toolCallId: "tool-call-1",
  },
  {
    type: "tool_result",
    isOwn: true,
    id: "tool-result-2",
    timestamp: 1776265192500,
    result: "Messages: 5 unread",
    toolCallId: "tool-call-2",
  },
  {
    type: "own_utterance",
    isOwn: true,
    text:
      "Oops something went wrong. Please try again, contact support or try /reset to reset bot memory.",
    id: "error-utterance",
    timestamp: 1776265193054,
  },
];

runForAllProviders(
  "agent handles history with multiple tool_calls from same response",
  async (runAgent) => {
    const inMemoryHistory = createHistoryWithMultipleToolCalls();

    const initialErrorCount = inMemoryHistory.filter(
      (event: unknown) =>
        typeof event === "object" &&
        event !== null &&
        "type" in event &&
        event.type === "own_utterance" &&
        "text" in event &&
        typeof event.text === "string" &&
        event.text.includes("Oops something went wrong"),
    ).length;

    await agentDeps(inMemoryHistory)(runAgent)({
      maxIterations: 1,
      onMaxIterationsReached: () => {},
      tools: [],
      prompt:
        "You are a helpful assistant. Respond to the user with a natural, helpful message.",
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });

    const finalErrorCount = inMemoryHistory.filter(
      (event: unknown) =>
        typeof event === "object" &&
        event !== null &&
        "type" in event &&
        event.type === "own_utterance" &&
        "text" in event &&
        typeof event.text === "string" &&
        event.text.includes("Oops something went wrong"),
    ).length;

    assertEquals(
      finalErrorCount,
      initialErrorCount,
      `Agent should not produce "Oops something went wrong" errors. Got ${
        finalErrorCount - initialErrorCount
      } new error(s).`,
    );

    const lastEvent = inMemoryHistory[inMemoryHistory.length - 1];
    assert(
      lastEvent.type === "own_utterance" || lastEvent.type === "do_nothing",
      `Last event should be own_utterance or do_nothing, got: ${lastEvent.type}`,
    );
    if (lastEvent.type === "own_utterance") {
      assert(
        !lastEvent.text.includes("Oops something went wrong"),
        "Last utterance should not be an error message",
      );
    }
  },
);
