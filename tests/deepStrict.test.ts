import { assertEquals } from "@std/assert";
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
