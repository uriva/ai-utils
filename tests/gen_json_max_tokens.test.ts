import { assertRejects } from "@std/assert";
import { genJson, z } from "../mod.ts";
import { injectSecrets, llmTest } from "../test_helpers.ts";

const schema = z.object({
  intro: z.string(),
  sections: z.array(z.object({ title: z.string(), body: z.string() })),
});

llmTest(
  "genJson throws a clear error on MAX_TOKENS truncation",
  injectSecrets(async () => {
    await assertRejects(
      () =>
        genJson(
          { provider: "google", mini: false, maxOutputTokens: 5 },
          "Please output an extremely long and detailed response conforming to the schema. V2",
          schema,
        )("Write a 500-word introduction about the history of computing."),
      Error,
      "truncated due to MAX_TOKENS limit",
    );
  }),
);
