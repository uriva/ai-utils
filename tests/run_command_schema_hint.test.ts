import { assertStringIncludes } from "@std/assert";
import { z } from "zod/v4";
import { createSkillTools, type Tool, tool } from "../src/agent.ts";

// Mirrors find-scene's find_by_scene_description: flat params where models
// blindly guess conventional-but-wrong names like `query`/`movie_title`.
const findBySceneDescription = tool({
  name: "find_by_scene_description",
  description: "Search scene by non dialog description.",
  parameters: z.object({
    description: z.string(),
    title: z.string().optional(),
    season: z.number().optional(),
    nSkip: z.number().default(0),
  }),
  handler: () => Promise.resolve("results"),
});

// Mirrors find-scene's flattenQueryField-wrapped schemas.
const flattenQueryField = (input: unknown): unknown => {
  if (input && typeof input === "object" && !Array.isArray(input)) {
    const { query, ...rest } = input as Record<string, unknown>;
    if (query && typeof query === "object" && !Array.isArray(query)) {
      return { ...rest, ...query };
    }
  }
  return input;
};

// deno-lint-ignore no-explicit-any
const getBestVideoSource: Tool<any> = {
  name: "get_best_video_source",
  description: "Get the best video source for a given video.",
  parameters: z.preprocess(
    flattenQueryField,
    z.object({
      title: z.string().describe("The movie or TV show title"),
      season: z.number().optional(),
      episode: z.number().optional(),
    }),
  ),
  handler: () => Promise.resolve("source"),
};

const sceneSearchSkill = {
  name: "scene_search",
  description: "scene search skill",
  instructions: "x",
  tools: [findBySceneDescription],
};

const videoSourcesSkill = {
  name: "video_sources",
  description: "video sources skill",
  instructions: "x",
  tools: [getBestVideoSource],
};

// deno-lint-ignore no-explicit-any
const runCommandTool = (skills: any[]) => {
  const runCommand = createSkillTools(skills).find((t) =>
    t.name === "run_command"
  );
  if (!runCommand) throw new Error("run_command missing");
  return runCommand;
};

Deno.test("run_command invalid-params error includes the tool's expected parameter typing", async () => {
  const runCommand = runCommandTool([sceneSearchSkill]);
  const out = await runCommand.handler(
    {
      command: "scene_search/find_by_scene_description",
      params: { query: "woman holding blanket wrapped around chest" },
    },
    "call-1",
  );
  if (typeof out !== "string") throw new Error("expected string result");
  assertStringIncludes(out, "Invalid parameters");
  // The error must teach the exact expected shape, so a blind retry is informed.
  assertStringIncludes(out, "description: string");
  assertStringIncludes(out, "title?: string");
});

Deno.test("run_command invalid-params error includes typing for preprocess-wrapped schemas", async () => {
  const runCommand = runCommandTool([videoSourcesSkill]);
  const out = await runCommand.handler(
    {
      command: "video_sources/get_best_video_source",
      params: { movie_title: "Fight Club" },
    },
    "call-2",
  );
  if (typeof out !== "string") throw new Error("expected string result");
  assertStringIncludes(out, "Invalid parameters");
  assertStringIncludes(out, "title: string");
  assertStringIncludes(out, "season?: number");
});
