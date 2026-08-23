import { assert, assertEquals } from "@std/assert";
import { z } from "zod/v4";
import { callToResult, type Tool } from "../src/agent.ts";

// Pins strict-key rejection of model-supplied tool arguments: unknown keys
// must produce actionable "Unrecognized key" errors instead of being silently
// stripped by Zod. Covers nesting, arrays-of-objects, unions (unchecked), and
// catchall passthrough (allowed).

const profileParameters = z.object({
  userId: z.string(),
  prefs: z
    .object({ theme: z.string(), lang: z.string().optional() })
    .optional(),
  tags: z.array(z.object({ key: z.string() })).optional(),
});

const profileTool: Tool<typeof profileParameters> = {
  name: "update_profile",
  description: "Updates the user profile",
  parameters: profileParameters,
  handler: (params) => Promise.resolve(JSON.stringify(params)),
};

const flexibleParameters = z.object({ name: z.string() }).catchall(z.unknown());

const flexibleTool: Tool<typeof flexibleParameters> = {
  name: "run_flexible",
  description: "Accepts arbitrary extra keys",
  parameters: flexibleParameters,
  handler: (params) => Promise.resolve(JSON.stringify(params)),
};

const run = async (toolName: string, args: unknown) =>
  await callToResult([profileTool, flexibleTool])({
    name: toolName,
    args: args as Record<string, unknown>,
    id: "call-1",
  });

Deno.test("unknown top-level key is rejected with expected-key list", async () => {
  const out = await run("update_profile", { userId: "u1", oops: true });
  assertEquals(
    out?.result,
    "Invalid arguments: oops: Unrecognized key. Expected keys: userId, prefs, tags",
  );
});

Deno.test("unknown nested key reports its path", async () => {
  const out = await run("update_profile", {
    userId: "u1",
    prefs: { theme: "dark", extra: 1 },
  });
  assertEquals(
    out?.result,
    "Invalid arguments: prefs.extra: Unrecognized key. Expected keys: theme, lang",
  );
});

Deno.test("unknown key inside array element reports index path", async () => {
  const out = await run("update_profile", {
    userId: "u1",
    tags: [{ key: "a" }, { key: "b", junk: 2 }],
  });
  assertEquals(
    out?.result,
    "Invalid arguments: tags.1.junk: Unrecognized key. Expected keys: key",
  );
});

Deno.test("union-typed values are not strict-key checked", async () => {
  const unionParameters = z.object({
    query: z.union([z.object({ text: z.string() }), z.string()]),
  });
  const unionTool: Tool<typeof unionParameters> = {
    name: "search",
    description: "Searches",
    parameters: unionParameters,
    handler: (params) => Promise.resolve(JSON.stringify(params)),
  };
  const out = await callToResult([unionTool])({
    name: "search",
    args: { query: { text: "hi", arbitrary: 1 } },
    id: "call-2",
  });
  assert(
    out?.result?.includes('"text":"hi"'),
    "union branch content should pass through without unknown-key rejection",
  );
});

Deno.test("catchall objects allow unknown keys", async () => {
  const out = await run("run_flexible", {
    name: "n",
    anything: { deep: true },
  });
  assert(out?.result?.includes('"anything"'), "catchall extras must survive");
});

Deno.test("valid arguments execute without corrections", async () => {
  const out = await run("update_profile", {
    userId: "u1",
    prefs: { theme: "dark" },
    tags: [{ key: "a" }],
  });
  assertEquals(
    out?.result,
    JSON.stringify({
      userId: "u1",
      prefs: { theme: "dark" },
      tags: [{ key: "a" }],
    }),
  );
});
