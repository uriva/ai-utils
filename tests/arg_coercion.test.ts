import { assertEquals } from "@std/assert";
import { z } from "zod/v4";
import { coerceArgs } from "../src/argCoercion.ts";

const schemaOf = (s: z.ZodType) => z.toJSONSchema(s);

Deno.test("coerceArgs no-op when args already match", () => {
  const schema = schemaOf(
    z.object({
      query: z.object({
        name: z.string(),
        episode: z.number(),
        season: z.number(),
      }),
      timeout: z.number().optional(),
    }),
  );
  const input = {
    query: { name: "x", episode: 1, season: 2 },
    timeout: 5,
  };
  const { args, corrections } = coerceArgs(schema, input);
  assertEquals(args, input);
  assertEquals(corrections, []);
});

Deno.test("coerceArgs wraps flat args into nested object when unambiguous", () => {
  const schema = schemaOf(
    z.object({
      query: z.object({
        name: z.string(),
        episode: z.number(),
        season: z.number(),
      }),
      timeout: z.number().optional(),
    }),
  );
  const { args, corrections } = coerceArgs(schema, {
    name: "x",
    episode: 1,
    season: 2,
  });
  assertEquals(args, { query: { name: "x", episode: 1, season: 2 } });
  assertEquals(corrections.length, 3);
});

Deno.test("coerceArgs corrects case mismatch on key", () => {
  const schema = schemaOf(z.object({ name: z.string() }));
  const { args, corrections } = coerceArgs(schema, { Name: "x" });
  assertEquals(args, { name: "x" });
  assertEquals(corrections.length, 1);
});

Deno.test("coerceArgs corrects snake_case to camelCase", () => {
  const schema = schemaOf(z.object({ episodeNumber: z.number() }));
  const { args, corrections } = coerceArgs(schema, { episode_number: 3 });
  assertEquals(args, { episodeNumber: 3 });
  assertEquals(corrections.length, 1);
});

Deno.test("coerceArgs corrects PascalCase to camelCase", () => {
  const schema = schemaOf(z.object({ episodeNumber: z.number() }));
  const { args, corrections } = coerceArgs(schema, { EpisodeNumber: 3 });
  assertEquals(args, { episodeNumber: 3 });
  assertEquals(corrections.length, 1);
});

Deno.test("coerceArgs corrects kebab-case to camelCase", () => {
  const schema = schemaOf(z.object({ episodeNumber: z.number() }));
  const { args, corrections } = coerceArgs(schema, { "episode-number": 3 });
  assertEquals(args, { episodeNumber: 3 });
  assertEquals(corrections.length, 1);
});

Deno.test("coerceArgs leaves args alone when ambiguous", () => {
  const schema = schemaOf(
    z.object({
      a: z.object({ shared: z.string() }),
      b: z.object({ shared: z.string() }),
    }),
  );
  const input = { shared: "x" };
  const { args, corrections } = coerceArgs(schema, input);
  assertEquals(args, input);
  assertEquals(corrections, []);
});

Deno.test("coerceArgs relocates a single wrong key to deep path when unique", () => {
  const schema = schemaOf(
    z.object({
      query: z.object({
        filters: z.object({
          season: z.number(),
        }),
      }),
    }),
  );
  const { args, corrections } = coerceArgs(schema, {
    query: { filters: {}, season: 2 },
  });
  assertEquals(args, { query: { filters: { season: 2 } } });
  assertEquals(corrections.length, 1);
});

Deno.test("coerceArgs handles multiple wrong keys one at a time", () => {
  const schema = schemaOf(
    z.object({
      query: z.object({
        name: z.string(),
        episode: z.number(),
      }),
    }),
  );
  const { args, corrections } = coerceArgs(schema, {
    Name: "x",
    Episode: 1,
  });
  assertEquals(args, { query: { name: "x", episode: 1 } });
  assertEquals(corrections.length, 2);
});

Deno.test("coerceArgs preserves correctly placed keys at top level", () => {
  const schema = schemaOf(
    z.object({
      query: z.object({ name: z.string() }),
      timeout: z.number().optional(),
    }),
  );
  const { args, corrections } = coerceArgs(schema, {
    query: { name: "x" },
    Timeout: 5,
  });
  assertEquals(args, { query: { name: "x" }, timeout: 5 });
  assertEquals(corrections.length, 1);
});

Deno.test("coerceArgs nests flat title under query for video source schema", () => {
  const schema = schemaOf(
    z.object({
      query: z.object({
        title: z.string().nullable().optional(),
        year: z.number().nullable().optional(),
        season: z.number().nullable().optional(),
      }),
    }),
  );
  const { args, corrections } = coerceArgs(schema, {
    title: "Never Let Me Go",
  });
  assertEquals(args, { query: { title: "Never Let Me Go" } });
  assertEquals(corrections.length, 1);
});
