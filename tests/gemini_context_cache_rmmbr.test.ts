import { assertEquals } from "@std/assert";
import { GoogleGenAI } from "@google/genai";
import {
  clearGeminiContextCacheMap,
  geminiContextCacheClientTtlSeconds,
  geminiContextCacheId,
  geminiContextCacheTtlSeconds,
  getOrCreateGeminiContextCache,
} from "../src/geminiContextCache.ts";
import { injectCacher } from "../src/cacher.ts";
import { injectGeminiToken } from "../src/gemini.ts";

Deno.test(
  "gemini context cache reuses rmmbr across isolate memory resets",
  async () => {
    const mockCache = new Map<string, unknown>();
    let cacherCalls = 0;
    const fakeCacher = (cacheId: string, ttl?: number) =>
      ((fn: (...args: unknown[]) => Promise<unknown>) =>
      async (...args: unknown[]) => {
        cacherCalls++;
        assertEquals(cacheId, geminiContextCacheId);
        assertEquals(ttl, geminiContextCacheClientTtlSeconds);
        const key = JSON.stringify(args);
        if (mockCache.has(key)) return mockCache.get(key);
        const result = await fn(...args);
        mockCache.set(key, result);
        return result;
      }) as never;

    const apiKey = Deno.env.get("GEMINI_API_KEY");
    if (!apiKey) return;

    await injectGeminiToken(apiKey)(
      injectCacher(fakeCacher)(async () => {
        clearGeminiContextCacheMap();

        assertEquals(geminiContextCacheTtlSeconds, 3600);
        assertEquals(geminiContextCacheClientTtlSeconds, 3480);
        assertEquals(geminiContextCacheId, "gemini-context-cache-v1");

        const sdk = new GoogleGenAI({ apiKey });
        const longPrompt =
          "Astronomical catalog test prompt for planetary science and orbital physics. "
            .repeat(400);

        const res1 = await getOrCreateGeminiContextCache(
          sdk,
          "gemini-3.8-flash",
          longPrompt,
        );
        assertEquals(typeof res1, "string");
        assertEquals(mockCache.size, 1);
        assertEquals(cacherCalls, 1);

        clearGeminiContextCacheMap();

        const res2 = await getOrCreateGeminiContextCache(
          sdk,
          "gemini-3.8-flash",
          longPrompt,
        );
        assertEquals(res2, res1);
        assertEquals(cacherCalls, 2);
        assertEquals(mockCache.size, 1);
      }),
    )();
  },
);
