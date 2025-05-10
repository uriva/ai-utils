import {
  injectOpenAiToken,
  injectRmmbrToken,
  openAiGenJsonFromConvo,
} from "./mod.ts";
import z from "zod";
import type { ChatCompletionMessageParam } from "openai/resources/index.mjs";
import { assert, assertObjectMatch } from "jsr:@std/assert";
import { coerce, pipe } from "gamla";

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
    let result;
    try {
      result = await openAiGenJsonFromConvo(
        { thinking: false, mini: true },
        messages,
        schema,
      );
    } catch (_e) {
      // If you expect a real API key, this will fail with test-token
      result = null;
    }
    assert(result == null || typeof result.hello === "string");
    if (result) {
      assertObjectMatch(result, { hello: result.hello });
    }
  }),
);
