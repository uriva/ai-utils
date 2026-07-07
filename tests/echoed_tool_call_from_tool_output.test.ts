import { assert } from "@std/assert";
import { runAgent } from "../mod.ts";
import {
  type HistoryEvent,
  injectCallModel,
  ownUtteranceTurn,
  participantUtteranceTurn,
  toolResultTurn,
} from "../src/agent.ts";
import { agentDeps, noopRewriteHistory } from "../test_helpers.ts";
import { z } from "zod/v4";

// Repro of the prompt2bot builder-bot infinite loop
// (view-chat conversation 58b66537-6613-4243-93c4-a1b95456ba32).
//
// A `read_conversation` tool returns another bot's WhatsApp history verbatim.
// That history contains a customer-visible LEAK of raw tool-call syntax, e.g.
//   <call:default_api:check_customer{phone: ...
// The model, seeing that syntax inside a tool RESULT, echoes it back as its own
// utterance (mimicry). The agent's mangled-tool-call recovery then promotes the
// echoed text into a REAL `check_customer` tool call. That tool does not exist,
// so the model gets "Tool not found", the poison text stays in context, and the
// same promotion fires again -> infinite loop.
//
// First failure point: the single turn where the model's echoed utterance is
// promoted into a real tool call. Mangled-tool-call recovery must only act on
// syntax the model ORIGINATED, never on syntax copied out of a tool result.

const leakedCallSyntax = "<call:default_api:check_customer{phone: 972526932741";

const readConversationTool = {
  name: "read_conversation",
  description: "Read messages from a conversation.",
  parameters: z.object({ conversation_id: z.string() }),
  handler: () =>
    Promise.resolve(
      "Messages:\n" +
        `- customer: hi\n` +
        `- bot: ${leakedCallSyntax}\n` +
        `- bot: yes\n`,
    ),
};

Deno.test(
  "echoed tool-call syntax from a tool result is not promoted into a real tool call",
  async () => {
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "admin",
        text: "Read conversation abc and tell me what leaked.",
      }),
      {
        type: "tool_call",
        isOwn: true,
        name: "read_conversation",
        parameters: { conversation_id: "abc" },
        id: crypto.randomUUID(),
        timestamp: Date.now() - 2000,
      },
      toolResultTurn({
        result: "Messages:\n- customer: hi\n- bot: " + leakedCallSyntax +
          "\n- bot: yes\n",
      }),
    ];

    // The model mimics the tool output: on its first turn it repeats the leaked
    // <call:...> text verbatim inside a normal utterance (exactly what happened
    // in the loop). Before the fix this echo was promoted into a real
    // check_customer tool call, whose "not found" result fed the next turn and
    // re-triggered the promotion forever.
    let turn = 0;
    const fakeCallModel = () => {
      turn++;
      if (turn === 1) {
        return Promise.resolve([
          ownUtteranceTurn("The leaked message was: " + leakedCallSyntax + "}"),
        ]);
      }
      return Promise.resolve([ownUtteranceTurn("Done.")]);
    };

    await injectCallModel(fakeCallModel)(async () => {
      await agentDeps(mockHistory)(runAgent)({
        maxIterations: 1,
        tools: [readConversationTool],
        prompt: "You are a builder assistant.",
        rewriteHistory: noopRewriteHistory,
        timezoneIANA: "UTC",
      });
    })();

    const newEvents = mockHistory.slice(3);

    const promotedToRealCall = newEvents.some((e) =>
      e.type === "tool_call" && e.name === "check_customer"
    );
    assert(
      !promotedToRealCall,
      `Echoed tool-call syntax from a tool result was promoted into a real ` +
        `check_customer tool call:\n${JSON.stringify(newEvents, null, 2)}`,
    );
  },
);
