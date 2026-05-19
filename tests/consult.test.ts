import { assert } from "@std/assert";
import { type HistoryEvent, participantUtteranceTurn } from "../src/agent.ts";
import { consultToolName } from "../src/consultTool.ts";
import {
  agentDeps,
  noopRewriteHistory,
  runForAllProviders,
} from "../test_helpers.ts";

runForAllProviders(
  "lightModel agent can call consult to ask the strong model",
  async (runAgentWithProvider) => {
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text:
          "I have a tricky architectural question. Use the consult tool to ask the stronger model whether I should pick monolith or microservices for a 3-engineer startup, then summarize its advice in one sentence.",
      }),
    ];
    await agentDeps(mockHistory)(runAgentWithProvider)({
      maxIterations: 6,
      onMaxIterationsReached: () => {},
      tools: [],
      prompt:
        "You are a junior assistant. When uncertain about hard reasoning, call the consult tool.",
      lightModel: true,
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
    assert(
      mockHistory.some((event) =>
        event.type === "tool_call" && event.name === consultToolName
      ),
      `expected agent to call ${consultToolName}. history: ${
        JSON.stringify(mockHistory, null, 2)
      }`,
    );
    assert(
      mockHistory.some((event) =>
        event.type === "tool_result" &&
        event.result.length > 0 &&
        !event.result.startsWith("[stronger model returned no text]")
      ),
      `expected consult tool to return a non-empty result. history: ${
        JSON.stringify(mockHistory, null, 2)
      }`,
    );
  },
);

runForAllProviders(
  "consult tool is absent when not on lightModel",
  async (runAgentWithProvider) => {
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text:
          "List the names of all tools currently available to you, one per line. Do not call any tool.",
      }),
    ];
    await agentDeps(mockHistory)(runAgentWithProvider)({
      maxIterations: 2,
      onMaxIterationsReached: () => {},
      tools: [],
      prompt: "You are a helpful assistant.",
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
    const reply = mockHistory.find((e) => e.type === "own_utterance");
    assert(reply, "expected a reply");
    assert(
      reply.type === "own_utterance" &&
        !reply.text.toLowerCase().includes(consultToolName),
      `expected reply not to mention ${consultToolName}. got: ${
        reply.type === "own_utterance" ? reply.text : ""
      }`,
    );
    assert(
      !mockHistory.some((event) =>
        event.type === "tool_call" && event.name === consultToolName
      ),
      `expected no ${consultToolName} tool call. history: ${
        JSON.stringify(mockHistory, null, 2)
      }`,
    );
  },
);
