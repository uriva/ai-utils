import { assert } from "@std/assert";
import { z } from "zod/v4";
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

const getOfficialRecord = {
  name: "get_official_record",
  description:
    "Returns the authoritative current status of official record R-204. This is the ONLY source of truth for the record; its real status cannot be known or guessed any other way.",
  parameters: z.object({}),
  handler: () =>
    Promise.resolve("Record R-204: status=DECOMMISSIONED (as of 2031-02-11)."),
};

// Regression: the strong model consulted by `consult` must NOT inherit the
// agent's tools. When it did, a verification question ("are you sure?") made it
// lead with a tool_call (here: get_official_record) to check the fact itself and
// emit no text, so consult returned "[stronger model returned no text]".
// Stripped of tools it must answer with text. Asserts on the consult result
// specifically (via toolCallId) so the agent's own tool call can't mask an
// empty consult reply.
runForAllProviders(
  "consult returns strong-model text even when the agent has tools",
  async (runAgentWithProvider) => {
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text:
          "I'm certain official record R-204 is currently ACTIVE. Before you reply, be absolutely sure: use the consult tool to have the stronger model check record R-204 itself and confirm its real current status.",
      }),
    ];
    await agentDeps(mockHistory)(runAgentWithProvider)({
      maxIterations: 6,
      tools: [getOfficialRecord],
      prompt:
        "You are a junior assistant and you must never guess or trust unverified claims. When the user asks you to confirm something, call the consult tool and ask the stronger model to verify it using whatever tools are available, then report back.",
      lightModel: true,
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
    const consultCallId = mockHistory.find(
      (e): e is Extract<HistoryEvent, { type: "tool_call" }> =>
        e.type === "tool_call" && e.name === consultToolName,
    )?.id;
    assert(
      consultCallId,
      `expected agent to call ${consultToolName}. history: ${
        JSON.stringify(mockHistory, null, 2)
      }`,
    );
    const consultReply = mockHistory.find(
      (e): e is Extract<HistoryEvent, { type: "tool_result" }> =>
        e.type === "tool_result" && e.toolCallId === consultCallId,
    );
    assert(
      consultReply && consultReply.result.length > 0 &&
        !consultReply.result.startsWith("[stronger model returned no text]"),
      `expected ${consultToolName} to return non-empty strong-model text; the strong model must not inherit the agent's tools. history: ${
        JSON.stringify(mockHistory, null, 2)
      }`,
    );
  },
);

runForAllProviders(
  "consult does not return empty when the agent prompt contains irrelevant message instructions",
  async (runAgentWithProvider) => {
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "is it sunny over there today?",
      }),
    ];
    await agentDeps(mockHistory)(runAgentWithProvider)({
      maxIterations: 6,
      tools: [],
      prompt:
        "You are a soccer bot. Respond ONLY to registrations. If a message is not related to registration, do not respond at all. When the user sends an irrelevant message, you MUST use the consult tool to ask the stronger model for advice on what to do. Be extra gentle.",
      lightModel: true,
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
    const consultCallId = mockHistory.find(
      (e): e is Extract<HistoryEvent, { type: "tool_call" }> =>
        e.type === "tool_call" && e.name === consultToolName,
    )?.id;
    assert(
      consultCallId,
      `expected agent to call ${consultToolName}. history: ${
        JSON.stringify(mockHistory, null, 2)
      }`,
    );
    const consultReply = mockHistory.find(
      (e): e is Extract<HistoryEvent, { type: "tool_result" }> =>
        e.type === "tool_result" && e.toolCallId === consultCallId,
    );
    assert(
      consultReply && consultReply.result.length > 0 &&
        !consultReply.result.startsWith("[stronger model returned no text]"),
      `expected ${consultToolName} to return non-empty strong-model text; the strong model must not return no-text. history: ${
        JSON.stringify(mockHistory, null, 2)
      }`,
    );
  },
);
