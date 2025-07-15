import { pipe } from "gamla";
import { assert, assertEquals } from "jsr:@std/assert";
import type { ChatCompletionMessageParam } from "openai/resources/index.mjs";
import { z } from "zod/v4";
import {
  geminiGenJsonFromConvo,
  injectCacher,
  injectDebugger,
  injectGeminiToken,
  injectOpenAiToken,
  openAiGenJsonFromConvo,
  runBot,
} from "./mod.ts";
import {
  type HistoryEvent,
  injectInMemoryHistory,
  participantUtteranceTurn,
  toolResultTurn,
  toolUseTurn,
} from "./src/geminiAgent.ts";

const injectSecrets = pipe(
  injectCacher(() => (f) => f),
  injectOpenAiToken(
    Deno.env.get("OPENAI_API_KEY") ?? "",
  ),
  injectGeminiToken(Deno.env.get("GEMINI_API_KEY") ?? ""),
);

Deno.test(
  "returns valid result for hello schema",
  injectSecrets(async () => {
    const schema = z.object({ hello: z.string() });
    const messages: ChatCompletionMessageParam[] = [
      { role: "system", content: "Say hello as JSON." },
      { role: "user", content: "hello" },
    ];
    for (
      const service of [openAiGenJsonFromConvo, geminiGenJsonFromConvo]
    ) {
      for (
        const [thinking, mini] of [
          [false, false],
          [false, true],
          [true, false],
          [true, true],
        ]
      ) {
        const result = await service({ thinking, mini }, messages, schema);
        console.log(result);
        assertEquals(result, { hello: result.hello });
      }
    }
  }),
);

const agentDeps = (mutableHistory: HistoryEvent[]) =>
  pipe(injectInMemoryHistory(mutableHistory), injectDebugger(() => {}));

const toolResult = "43212e8e-4c29-4a3c-aba2-723e668b5537";

const someTool = {
  name: "doSomethingUnique",
  description: "Returns a unique string so we know the tool was called.",
  parameters: z.object({}),
  handler: () => Promise.resolve(toolResult),
};

Deno.test(
  "runBot calls the tool and replies with its output",
  injectSecrets(async () => {
    const mockHistory: HistoryEvent[] = [participantUtteranceTurn({
      name: "user",
      text:
        `Please call the doSomethingUnique tool now and only reply with its output.`,
    })];
    await agentDeps(mockHistory)(runBot)({
      maxIterations: 5,
      actions: [someTool],
      prompt: `You are an AI assistant.`,
    });
    assert(mockHistory.some((event) => (event.type === "own_utterance" &&
      event.text?.includes(toolResult))
    ));
  }),
);

Deno.test(
  "agent can start an empty conversation",
  injectSecrets(async () => {
    await agentDeps([])(runBot)({
      actions: [],
      prompt: `You are the neighborhood friendly spiderman.`,
      maxIterations: 5,
    });
  }),
);

Deno.test(
  "conversation can start with a tool call",
  injectSecrets(async () => {
    await agentDeps([
      toolUseTurn({ name: someTool.name, args: {} }),
      toolResultTurn({
        name: someTool.name,
        response: { result: toolResult },
      }),
    ])(
      runBot,
    )({
      actions: [someTool],
      prompt: `You are the neighborhood friendly spiderman.`,
      maxIterations: 5,
    });
  }),
);
