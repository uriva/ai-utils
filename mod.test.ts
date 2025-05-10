import { coerce, pipe } from "gamla";
import { assertEquals } from "jsr:@std/assert";
import type { ChatCompletionMessageParam } from "openai/resources/index.mjs";
import { waitAllWrites } from "rmmbr";
import z from "zod";
import {
  injectOpenAiToken,
  injectRmmbrToken,
  openAiGenJsonFromConvo,
} from "./mod.ts";

Deno.test(
  "openAiGenJsonFromConvo returns valid result for hello schema",
  pipe(
    injectRmmbrToken(coerce(Deno.env.get("RMMBR_API_KEY"))),
    injectOpenAiToken(coerce(Deno.env.get("OPENAI_API_KEY"))),
  )(async () => {
    const schema = z.object({ hello: z.string() });
    const messages: ChatCompletionMessageParam[] = [
      { role: "system", content: "Say hello as JSON." },
      { role: "user", content: "hello" },
    ];
    for (
      const [thinking, mini] of [
        [false, false],
        [false, true],
        [true, false],
        [true, true],
      ]
    ) {
      const result = await openAiGenJsonFromConvo(
        { thinking, mini },
        messages,
        schema,
      );
      assertEquals(result, { hello: result.hello });
    }
    await waitAllWrites();
  }),
);
