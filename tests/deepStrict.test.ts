import { assertEquals, assert } from "@std/assert";
import z from "zod/v4";
import { deepStrict } from "../src/deepStrict.ts";

Deno.test("deepStrict makes nested objects strict", () => {
  const schema = z.object({
    outer: z.object({
      inner: z.string(),
    }),
  });

  const strictSchema = deepStrict(schema);

  const result = strictSchema.safeParse({
    outer: {
      inner: "hello",
      extra: "world",
    },
  });

  assertEquals(result.success, false);
});

Deno.test("deepStrict handles ZodDefault properly", () => {
  const searchToolParams = z.object({
    prefix: z.string(),
    skip: z.number().default(0),
  });

  const strictSchema = deepStrict(searchToolParams);
  
  const result = strictSchema.safeParse({ prefix: "abc" });
  assert(result.success);
  assertEquals(result.data.skip, 0);
  
  const resultWithSkip = strictSchema.safeParse({ prefix: "abc", skip: 5 });
  assert(resultWithSkip.success);
  assertEquals(resultWithSkip.data.skip, 5);
  
  const resultWithExtra = strictSchema.safeParse({ prefix: "abc", extra: true });
  assertEquals(resultWithExtra.success, false);
});
