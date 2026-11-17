import { assertEquals } from "@std/assert";
import z from "zod/v4";
import { genJson, injectOpenAiToken } from "../mod.ts";
import {
  restoreOptionalNullsToUndefined,
  toOpenAiStrictSchema,
} from "../src/openai.ts";

Deno.test("toOpenAiStrictSchema and restoreOptionalNullsToUndefined handle optional, nullable, nullish, and nested array/object fields", () => {
  const schema = z.object({
    hasIntent: z.boolean(),
    eventReference: z.string().optional().describe("Optional event URL/name"),
    nullableNote: z.string().nullable(),
    nullishTag: z.string().nullish(),
    items: z.array(
      z.object({
        title: z.string(),
        venue: z.string().optional(),
      }),
    ),
  });

  const rawJsonSchema = z.toJSONSchema(schema) as Record<string, unknown>;
  const strict = toOpenAiStrictSchema(rawJsonSchema);
  assertEquals(strict.required, [
    "hasIntent",
    "eventReference",
    "nullableNote",
    "nullishTag",
    "items",
  ]);
  assertEquals(strict.additionalProperties, false);

  const restored = restoreOptionalNullsToUndefined(
    {
      hasIntent: false,
      eventReference: null,
      nullableNote: null,
      nullishTag: null,
      items: [{ title: "Jazz Night", venue: null }],
    },
    rawJsonSchema,
  );

  const parsed = schema.parse(restored);
  assertEquals(parsed, {
    hasIntent: false,
    nullableNote: null,
    nullishTag: null,
    items: [{ title: "Jazz Night" }],
  });
  assertEquals("eventReference" in parsed, false);
});

Deno.test("genJson with openai provider succeeds on schema with optional eventReference field", async () => {
  const openAiKey = Deno.env.get("OPENAI_API_KEY");
  if (!openAiKey || openAiKey === "ci-placeholder") return;

  const intentSchema = z.object({
    hasAttendanceIntent: z.boolean(),
    eventReference: z.string().optional().describe(
      "Specific event name or URL if mentioned",
    ),
  });

  const res = await injectOpenAiToken(openAiKey)(() =>
    genJson(
      { provider: "openai", tier: "flash" },
      "Classify whether the user expresses intent to attend a specific event.",
      intentSchema,
    )("Good morning everyone, how is traffic today?")
  )();

  assertEquals(res.hasAttendanceIntent, false);
  assertEquals(res.eventReference, undefined);
});
