import { assertRejects } from "@std/assert";
import type { Injector } from "@uri/inject";
import {
  genJson,
  injectCacher,
  injectGeminiToken,
  invalidGenJsonMessage,
  z,
} from "../mod.ts";

// Reproduces the production incident where Gemini returned an empty candidate.
// The SDK call was coerced to the literal "{}", which passed through genJson as
// a schema-violating empty object and was cached, so every subsequent call for
// the same input returned {} forever — later crashing a downstream consumer
// that read a required field off the result. genJson must instead reject a
// result that does not satisfy the requested schema, so nothing invalid is
// returned or cached.
const poisonedTextCacher = (
  poison: string,
) => ((_cacheId: string): Injector =>
  ((_f: (...args: unknown[]) => Promise<unknown>) => () =>
    Promise.resolve(poison)) as Injector);

const schema = z.object({
  intro: z.string(),
  sections: z.array(z.object({ title: z.string(), body: z.string() })),
});

Deno.test("genJson throws instead of returning a schema-violating empty object", async () => {
  await injectGeminiToken("unused-because-cache-hits")(
    injectCacher(poisonedTextCacher("{}"))(async () => {
      await assertRejects(
        () =>
          genJson({ provider: "google", mini: false }, "split it", schema)(
            "some long prompt",
          ),
        Error,
        invalidGenJsonMessage,
      );
    }),
  )();
});
