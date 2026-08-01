import { assertEquals } from "@std/assert";
import { z } from "zod/v4";
import { arrayWrapCorrection, coerceArgs } from "../src/argCoercion.ts";

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

Deno.test("coerceArgs treats undefined as empty object when schema expects an object", () => {
  const schema = schemaOf(z.object({}));
  const { args, corrections } = coerceArgs(schema, undefined);
  assertEquals(args, {});
  assertEquals(corrections, []);
});

Deno.test("coerceArgs wraps a scalar into a single-element array when schema expects an array", () => {
  const schema = schemaOf(
    z.object({ query: z.array(z.string()).optional(), mapId: z.string() }),
  );
  const { args, corrections } = coerceArgs(schema, {
    mapId: "m1",
    query: "ramen",
  });
  assertEquals(args, { mapId: "m1", query: ["ramen"] });
  assertEquals(corrections, [arrayWrapCorrection("query")]);
});

Deno.test("coerceArgs wraps a single object into an array when schema expects an array of objects", () => {
  const schema = schemaOf(
    z.object({ filters: z.array(z.object({ season: z.number() })) }),
  );
  const { args, corrections } = coerceArgs(schema, { filters: { season: 2 } });
  assertEquals(args, { filters: [{ season: 2 }] });
  assertEquals(corrections, [arrayWrapCorrection("filters")]);
});

Deno.test("coerceArgs wraps scalars nested deep in the args tree", () => {
  const schema = schemaOf(
    z.object({
      query: z.object({ tags: z.array(z.string()), name: z.string() }),
    }),
  );
  const { args, corrections } = coerceArgs(schema, {
    query: { tags: "a", name: "x" },
  });
  assertEquals(args, { query: { tags: ["a"], name: "x" } });
  assertEquals(corrections, [arrayWrapCorrection("query.tags")]);
});

Deno.test("coerceArgs leaves existing arrays untouched", () => {
  const schema = schemaOf(z.object({ query: z.array(z.string()).optional() }));
  const input = { query: ["a", "b"] };
  const { args, corrections } = coerceArgs(schema, input);
  assertEquals(args, input);
  assertEquals(corrections, []);
});

Deno.test("coerceArgs does not unwrap an array when schema expects a string", () => {
  const schema = schemaOf(z.object({ query: z.string().optional() }));
  const input = { query: ["a", "b"] };
  const { args, corrections } = coerceArgs(schema, input);
  assertEquals(args, input);
  assertEquals(corrections, []);
});

Deno.test("coerceArgs wraps after renaming when both are needed", () => {
  const schema = schemaOf(z.object({ query: z.array(z.string()) }));
  const { args, corrections } = coerceArgs(schema, { Query: "x" });
  assertEquals(args, { query: ["x"] });
  assertEquals(corrections.length, 2);
  assertEquals(corrections[1], arrayWrapCorrection("query"));
});

Deno.test("coerceArgs wraps scalar items inside array-of-object elements", () => {
  const schema = schemaOf(
    z.object({
      groups: z.array(z.object({ tags: z.array(z.string()) })),
    }),
  );
  const { args, corrections } = coerceArgs(schema, {
    groups: [{ tags: "a" }],
  });
  assertEquals(args, { groups: [{ tags: ["a"] }] });
  assertEquals(corrections, [arrayWrapCorrection("groups.0.tags")]);
});
