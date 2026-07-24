import { assertEquals, assertRejects } from "@std/assert";
import type { GenerateContentResponse } from "@google/genai";
import type { Injector } from "@uri/inject";
import { genJson, injectCacher, injectGeminiToken, z } from "../mod.ts";
import { injectGeminiGenerateContent } from "../src/gemini.ts";

// Reproduces a cache-poisoning incident: the model returned a malformed
// (truncated) JSON body once, that raw text was written to the response
// cache, and every later call for the same input replayed the poisoned
// payload, so JSON.parse failed at the exact same byte position forever.
// Parsing must happen inside the cached function, so a malformed body is
// never cached and the next attempt re-hits the API instead of the poison.
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

const schema = z.object({ answer: z.string() });

const runGenJson = () =>
  genJson({ provider: "google", mini: false }, "sys", schema)("user");

Deno.test("genJson recovers from a one-off malformed response and never caches it", async () => {
  const validJson = JSON.stringify({ answer: "ok" });
  let calls = 0;
  const malformedThenValid = () => {
    calls++;
    return Promise.resolve(
      fakeResponse(calls === 1 ? "{malformed" : validJson),
    );
  };
  await injectGeminiToken("unused-because-generate-is-faked")(
    injectCacher(memoryCacher({}))(
      injectGeminiGenerateContent(malformedThenValid)(async () => {
        const result = await runGenJson();
        assertEquals(result, { answer: "ok" });
        assertEquals(
          calls > 1,
          true,
          "malformed response must be retried against the API, not served from cache",
        );
        const callsAfterFirst = calls;
        await runGenJson();
        assertEquals(
          calls,
          callsAfterFirst,
          "successful result must be served from cache",
        );
      }),
    ),
  )();
});

Deno.test("genJson does not cache a persistently malformed response", async () => {
  let calls = 0;
  const alwaysMalformed = () => {
    calls++;
    return Promise.resolve(fakeResponse("{malformed"));
  };
  await injectGeminiToken("unused-because-generate-is-faked")(
    injectCacher(memoryCacher({}))(
      injectGeminiGenerateContent(alwaysMalformed)(async () => {
        await assertRejects(runGenJson, SyntaxError);
        const callsAfterFirst = calls;
        await assertRejects(runGenJson, SyntaxError);
        assertEquals(
          calls > callsAfterFirst,
          true,
          "failed parse must not be cached; each call must re-hit the API",
        );
      }),
    ),
  )();
});
