import { assertEquals, assertRejects } from "@std/assert";
import type { GenerateContentResponse } from "@google/genai";
import type { Injector } from "@uri/inject";
import {
  genJson,
  injectCacher,
  injectGeminiToken,
  invalidGenJsonMessage,
  z,
} from "../mod.ts";
import { injectGeminiGenerateContent } from "../src/gemini.ts";

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

const memoryCacher =
  (store: Record<string, unknown>) => (_cacheId: string): Injector =>
    ((f: (...args: unknown[]) => Promise<unknown>) =>
    async (...args: unknown[]) => {
      const key = JSON.stringify(args);
      if (key in store) return store[key];
      const result = await f(...args);
      store[key] = result;
      return result;
    }) as Injector;

const fakeResponse = (text: string) =>
  ({
    candidates: [{ finishReason: "STOP" }],
    text,
  }) as GenerateContentResponse;

const schema = z.object({
  intro: z.string(),
  sections: z.array(z.object({ title: z.string(), body: z.string() })),
});

Deno.test("genJson throws instead of returning a schema-violating empty object", async () => {
  await injectGeminiToken("unused-because-cache-hits")(
    injectCacher(poisonedTextCacher("{}"))(async () => {
      await assertRejects(
        () =>
          genJson({ provider: "google", tier: "pro" }, "split it", schema)(
            "some long prompt",
          ),
        Error,
        invalidGenJsonMessage,
      );
    }),
  )();
});

const simpleSchema = z.object({ answer: z.string() });

Deno.test("genJson recovers from a one-off empty candidate response via retry", async () => {
  const validJson = JSON.stringify({ answer: "ok" });
  let calls = 0;
  const emptyThenValid = () => {
    calls++;
    return Promise.resolve(
      fakeResponse(calls === 1 ? "" : validJson),
    );
  };
  await injectGeminiToken("unused-because-generate-is-faked")(
    injectCacher(memoryCacher({}))(
      injectGeminiGenerateContent(emptyThenValid)(async () => {
        const result = await genJson(
          { provider: "google", tier: "pro" },
          "sys",
          simpleSchema,
        )("user");
        assertEquals(result, { answer: "ok" });
        assertEquals(
          calls > 1,
          true,
          "empty candidate must be retried against the API, not throw immediately",
        );
      }),
    ),
  )();
});

Deno.test("genJson falls back to alternate model when primary model persistently returns empty candidate", async () => {
  const validJson = JSON.stringify({ answer: "fallback-ok" });
  const modelsCalled: string[] = [];
  const emptyOnPrimaryThenValidOnFallback = (req: { model: string }) => {
    modelsCalled.push(req.model);
    if (req.model === "gemini-3.5-flash-lite") {
      return Promise.resolve(fakeResponse(validJson));
    }
    return Promise.resolve(fakeResponse(""));
  };
  await injectGeminiToken("unused-because-generate-is-faked")(
    injectCacher(memoryCacher({}))(
      // deno-lint-ignore no-explicit-any
      injectGeminiGenerateContent(emptyOnPrimaryThenValidOnFallback as any)(
        async () => {
          const result = await genJson(
            { provider: "google", tier: "pro" },
            "sys",
            simpleSchema,
          )("user");
          assertEquals(result, { answer: "fallback-ok" });
          assertEquals(modelsCalled.includes("gemini-3.5-flash-lite"), true);
        },
      ),
    ),
  )();
});
