import type { Content } from "@google/generative-ai";
import { pipe } from "gamla";
import { assert, assertEquals } from "jsr:@std/assert";
import type { ChatCompletionMessageParam } from "openai/resources/index.mjs";
import { z } from "zod/v4";
import {
  geminiGenJsonFromConvo,
  injectAccessHistory,
  injectCacher,
  injectedDebugLogs,
  injectGeminiToken,
  injectOpenAiToken,
  injectOutputEvent,
  openAiGenJsonFromConvo,
  runBot,
} from "./mod.ts";
import { functionCallTurn, functionResultTurn } from "./src/geminiAgent.ts";

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

const agentDeps = (mutableHistory: Content[]) =>
  pipe(
    injectOutputEvent((event: Content) => {
      mutableHistory.push(event);
      return Promise.resolve();
    }),
    injectAccessHistory(() => Promise.resolve(mutableHistory)),
    injectedDebugLogs(() => {}),
  );

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
    const mockHistory: Content[] = [{
      role: "user",
      parts: [{ text: "please use the tool" }],
    }];
    await agentDeps(mockHistory)(runBot)({
      maxIterations: 5,
      actions: [someTool],
      prompt:
        `Always use ${someTool.name} tool to answer the user. Include in your answer the unique string you got.`,
    });
    assert(
      mockHistory.some((event) =>
        event.parts.some((part) => part.text?.includes(toolResult))
      ),
    );
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
      functionCallTurn({ name: someTool.name, args: {} }),
      functionResultTurn({
        functionResponse: {
          name: someTool.name,
          response: { result: toolResult },
        },
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
