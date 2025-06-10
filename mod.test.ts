import { pipe } from "gamla";
import { assert, assertEquals } from "jsr:@std/assert";
import type { ChatCompletionMessageParam } from "openai/resources/index.mjs";
import { waitAllWrites } from "rmmbr";
import { z } from "zod/v4";
import {
  geminiGenJsonFromConvo,
  injectAccessHistory,
  injectAgentSystemLog,
  injectedDebugLogs,
  injectGeminiToken,
  injectOpenAiToken,
  injectReply,
  injectRmmbrToken,
  openAiGenJsonFromConvo,
  runBot,
} from "./mod.ts";

const injectSecrets = pipe(
  injectRmmbrToken(Deno.env.get("RMMBR_TOKEN") ?? ""),
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
      const service of [
        openAiGenJsonFromConvo,
        geminiGenJsonFromConvo,
      ]
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
        assertEquals(result, { hello: result.hello });
      }
    }
    await waitAllWrites();
  }),
);

Deno.test(
  "runBot calls the tool and replies with its output",
  injectSecrets(async () => {
    let replyText = "";
    const mockHistory = [
      { text: "please use the tool", from: "user", time: Date.now() },
    ];
    const deps = pipe(
      injectAccessHistory(() => Promise.resolve(mockHistory)),
      injectReply((text: string) => {
        replyText = text;
        return Promise.resolve();
      }),
      injectedDebugLogs(() => {}),
      injectAgentSystemLog(() => {}),
    );
    const toolResult = "43212e8e-4c29-4a3c-aba2-723e668b5537";
    const toolName = "doSomethingUnique";
    await deps(runBot)({
      actions: [{
        name: toolName,
        description: "Returns a unique string so we know the tool was called.",
        parameters: z.object({}),
        handler: () => Promise.resolve(toolResult),
      }],
      prompt: `Always use ${toolName} tool to answer the user.`,
      botNameInHistory: "bot",
    });
    assert(replyText.includes(toolResult));
  }),
);
