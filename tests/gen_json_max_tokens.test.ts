import { assertEquals, assertRejects } from "@std/assert";
import type {
  GenerateContentParameters,
  GenerateContentResponse,
} from "@google/genai";
import type { Injector } from "@uri/inject";
import { genJson, injectCacher, injectGeminiToken, z } from "../mod.ts";
import { injectGeminiGenerateContent } from "../src/gemini.ts";
import { injectSecrets, llmTest } from "../test_helpers.ts";

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

const schema = z.object({
  intro: z.string(),
  sections: z.array(z.object({ title: z.string(), body: z.string() })),
});

const simpleSchema = z.object({ answer: z.string() });

llmTest(
  "genJson throws a clear error on MAX_TOKENS truncation",
  injectSecrets(async () => {
    await assertRejects(
      () =>
        genJson(
          { provider: "google", tier: "pro", maxOutputTokens: 5 },
          "Please output an extremely long and detailed response conforming to the schema. V2",
          schema,
        )("Write a 500-word introduction about the history of computing."),
      Error,
      "truncated due to MAX_TOKENS limit",
    );
  }),
);

Deno.test(
  "genJson falls back to alternate model when primary model returns MAX_TOKENS",
  async () => {
    const validJson = JSON.stringify({ answer: "fallback-success" });
    const modelsCalled: string[] = [];
    const maxTokensOnPrimaryThenValidOnFallback = (
      req: GenerateContentParameters,
    ): Promise<GenerateContentResponse> => {
      modelsCalled.push(req.model);
      if (req.model === "gemini-3.5-flash-lite") {
        return Promise.resolve({
          candidates: [{ finishReason: "STOP" }],
          text: validJson,
        } as unknown as GenerateContentResponse);
      }
      return Promise.resolve({
        candidates: [{ finishReason: "MAX_TOKENS" }],
        text: "",
      } as unknown as GenerateContentResponse);
    };

    await injectGeminiToken("unused-because-generate-is-faked")(
      injectCacher(memoryCacher({}))(
        injectGeminiGenerateContent(maxTokensOnPrimaryThenValidOnFallback)(
          async () => {
            const result = await genJson(
              { provider: "google", tier: "pro" },
              "sys",
              simpleSchema,
            )("user");
            assertEquals(result, { answer: "fallback-success" });
            assertEquals(modelsCalled.includes("gemini-3.5-flash-lite"), true);
          },
        ),
      ),
    )();
  },
);
