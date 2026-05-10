import { assertEquals } from "@std/assert";
import { z } from "zod/v4";
import { callToResult, createSkillTools, tool } from "../src/agent.ts";

const findEpisode = tool({
  name: "find_episode",
  description: "find an episode",
  parameters: z.object({
    query: z.object({
      name: z.string(),
      episode: z.number(),
      season: z.number(),
    }),
    timeout: z.number().optional(),
  }),
  handler: ({ query, timeout }) =>
    Promise.resolve(
      `found ${query.name} s${query.season}e${query.episode} timeout=${
        timeout ?? "n/a"
      }`,
    ),
});

Deno.test("callToResult coerces flat args into nested shape and prefixes the result", async () => {
  const out = await callToResult([findEpisode])({
    name: "find_episode",
    args: { name: "Lost", episode: 1, season: 2 },
    id: "call-1",
  });
  assertEquals(out?.toolCallId, "call-1");
  if (!out) throw new Error("expected a result");
  // Prefix mentions the auto-correction.
  if (!out.result.startsWith("[arguments auto-corrected:")) {
    throw new Error(`expected prefix, got: ${out.result}`);
  }
  // Underlying tool ran with correctly shaped args.
  if (!out.result.includes("found Lost s2e1")) {
    throw new Error(`expected handler result, got: ${out.result}`);
  }
});

Deno.test("callToResult does not add prefix when args are already canonical", async () => {
  const out = await callToResult([findEpisode])({
    name: "find_episode",
    args: { query: { name: "Lost", episode: 1, season: 2 } },
    id: "call-2",
  });
  if (!out) throw new Error("expected a result");
  assertEquals(out.result.startsWith("[arguments auto-corrected"), false);
});

Deno.test("callToResult prefixes error when coerced args still fail validation", async () => {
  const out = await callToResult([findEpisode])({
    name: "find_episode",
    args: { Name: "Lost" }, // case-corrected to `name`, then validation fails (missing episode/season)
    id: "call-3",
  });
  if (!out) throw new Error("expected a result");
  if (!out.result.startsWith("[arguments auto-corrected:")) {
    throw new Error(`expected correction prefix, got: ${out.result}`);
  }
  if (!out.result.includes("Invalid arguments")) {
    throw new Error(`expected validation error, got: ${out.result}`);
  }
});

Deno.test("run_command coerces inner skill-tool params and prefixes the result", async () => {
  const skillTools = createSkillTools([
    {
      name: "video",
      description: "video skill",
      instructions: "x",
      tools: [findEpisode],
    },
  ]);
  const runCommand = skillTools.find((t) => t.name === "run_command");
  if (!runCommand) throw new Error("run_command missing");
  const out = await runCommand.handler(
    {
      command: "video/find_episode",
      params: { Name: "Lost", Episode: 1, Season: 2 },
    },
    "call-id",
  );
  if (typeof out !== "string") throw new Error("expected string result");
  if (!out.startsWith("[arguments auto-corrected:")) {
    throw new Error(`expected prefix, got: ${out}`);
  }
  if (!out.includes("found Lost s2e1")) {
    throw new Error(`expected handler output, got: ${out}`);
  }
});

Deno.test("run_command leaves canonical params alone", async () => {
  const skillTools = createSkillTools([
    {
      name: "video",
      description: "video skill",
      instructions: "x",
      tools: [findEpisode],
    },
  ]);
  const runCommand = skillTools.find((t) => t.name === "run_command");
  if (!runCommand) throw new Error("run_command missing");
  const out = await runCommand.handler(
    {
      command: "video/find_episode",
      params: { query: { name: "Lost", episode: 1, season: 2 } },
    },
    "call-id",
  );
  if (typeof out !== "string") throw new Error("expected string result");
  assertEquals(out.startsWith("[arguments auto-corrected"), false);
  if (!out.includes("found Lost s2e1")) {
    throw new Error(`expected handler output, got: ${out}`);
  }
});
