import { assert, assertEquals } from "@std/assert";
import { z } from "zod/v4";
import { runAgent } from "../mod.ts";
import {
  type HistoryEvent,
  injectCallModel,
  ownUtteranceTurn,
  participantUtteranceTurn,
} from "../src/agent.ts";
import { agentDeps, noopRewriteHistory } from "../test_helpers.ts";

// The model sometimes renders a tool call as visible text, and the provider can
// deliver that text split across several text parts (thought-signature
// boundaries break one logical message into separate parts). Each part becomes
// its own utterance event, and the mangled tool-call recovery then classifies
// every event in isolation: the part carrying the preamble is promoted with
// truncated/empty arguments, while the remaining parts — which no longer match
// the mangled-call pattern on their own — leak to the user as visible
// messages. Recovery must consider the response's visible text as a whole:
// join the fragments, recover the full call, and never leak pieces of it.

const toolName = "read_file";
const filePath = "/tmp/notes.txt";
const spinnerText = "Reading the file...";
const fileContents = "milk, eggs, flour";
const finalAnswer = "The file contains a short shopping list.";

const preambleFragment = `startcall:default_api:${toolName}{path: `;
const bodyFragment = `${filePath}\n\n,spinnerText:\n\n${spinnerText}\n\n}`;

const isRecord = (x: unknown): x is Record<string, unknown> =>
  !!x && typeof x === "object" && !Array.isArray(x);

const readFileTool = {
  name: toolName,
  description: "Read a file from disk.",
  parameters: z.object({ path: z.string(), spinnerText: z.string() }),
  handler: () => Promise.resolve(fileContents),
};

Deno.test(
  "mangled tool call split across multiple text parts does not leak fragments and recovers the full call",
  async () => {
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: `Read ${filePath} and tell me what it says.`,
      }),
    ];

    let turn = 0;
    const fakeCallModel = () => {
      turn++;
      if (turn === 1) {
        return Promise.resolve([
          ownUtteranceTurn(preambleFragment),
          ownUtteranceTurn(bodyFragment),
        ]);
      }
      return Promise.resolve([ownUtteranceTurn(finalAnswer)]);
    };

    await injectCallModel(fakeCallModel)(async () => {
      await agentDeps(mockHistory)(runAgent)({
        maxIterations: 10,
        tools: [readFileTool],
        prompt: "You are a helpful assistant.",
        rewriteHistory: noopRewriteHistory,
        timezoneIANA: "UTC",
      });
    })();

    const newEvents = mockHistory.slice(1);

    const leakedFragments = newEvents.filter(
      (e) => e.type === "own_utterance" && e.text !== finalAnswer,
    );
    assertEquals(
      leakedFragments,
      [],
      "fragments of a mangled tool call leaked as visible messages",
    );

    const calls = newEvents.filter(
      (e) => e.type === "tool_call" && e.name === toolName,
    );
    assertEquals(calls.length, 1, "expected exactly one recovered tool call");
    const [call] = calls;
    assert(
      call.type === "tool_call" && isRecord(call.parameters) &&
        call.parameters.path === filePath &&
        call.parameters.spinnerText === spinnerText,
      `recovered tool call has truncated arguments: ${JSON.stringify(call)}`,
    );
  },
);
