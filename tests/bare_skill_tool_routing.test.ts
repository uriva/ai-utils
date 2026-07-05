import { assertEquals } from "@std/assert";
import { z } from "zod/v4";
import {
  callToResult,
  createSkillTools,
  formatSkillsPrompt,
  getSpecForTurn,
  resolveToolDescription,
  tool,
} from "../src/agent.ts";
import type { AgentSpec, Skill } from "../src/agent.ts";

const todoWrite = tool({
  name: "todo_write",
  description: "write todos",
  parameters: z.object({ todos: z.array(z.string()) }),
  handler: ({ todos }) => Promise.resolve(`wrote ${todos.length} todos`),
});

const sharedDelete = tool({
  name: "delete",
  description: "delete something",
  parameters: z.object({ id: z.string() }),
  handler: ({ id }) => Promise.resolve(`deleted ${id}`),
});

const todoSkill: Skill = {
  name: "todo",
  description: "todo skill",
  instructions: "x",
  tools: [todoWrite],
};

const fileSkill: Skill = {
  name: "file",
  description: "file skill",
  instructions: "x",
  tools: [sharedDelete],
};

const dbSkill: Skill = {
  name: "db",
  description: "db skill",
  instructions: "x",
  tools: [sharedDelete],
};

Deno.test("bare unambiguous skill tool name is auto-routed via run_command", async () => {
  const skillTools = createSkillTools([todoSkill]);
  const out = await callToResult(skillTools, [todoSkill])({
    name: "todo_write",
    args: { todos: ["a", "b"] },
    id: "call-1",
  });
  if (!out) throw new Error("expected a result");
  if (!out.result.includes("wrote 2 todos")) {
    throw new Error(`expected handler output, got: ${out.result}`);
  }
});

Deno.test("bare ambiguous skill tool name still returns not-found", async () => {
  const skillTools = createSkillTools([fileSkill, dbSkill]);
  const out = await callToResult(skillTools, [fileSkill, dbSkill])({
    name: "delete",
    args: { id: "x" },
    id: "call-2",
  });
  if (!out) throw new Error("expected a result");
  assertEquals(out.result.startsWith(`Tool "delete" not found.`), true);
});

Deno.test("bare unknown tool name returns not-found", async () => {
  const skillTools = createSkillTools([todoSkill]);
  const out = await callToResult(skillTools, [todoSkill])({
    name: "nonexistent_tool",
    args: {},
    id: "call-3",
  });
  if (!out) throw new Error("expected a result");
  assertEquals(
    out.result.startsWith(`Tool "nonexistent_tool" not found.`),
    true,
  );
});

Deno.test("prefixed or malformed learn_skill calls are normalized and routed correctly", async () => {
  const skillTools = createSkillTools([todoSkill]);
  const out1 = await callToResult(skillTools, [todoSkill])({
    name: "todo/learn_skill",
    args: { spinnerText: "Learning todo skill" },
    id: "call-4",
  });
  if (!out1) throw new Error("expected a result");
  assertEquals(out1.result.includes("learned successfully"), true);

  const out2 = await callToResult(skillTools, [todoSkill])({
    name: "learn_skill",
    args: { skill: "todo", spinnerText: "Learning todo skill" },
    id: "call-5",
  });
  if (!out2) throw new Error("expected a result");
  assertEquals(out2.result.includes("learned successfully"), true);
});

Deno.test("auto-routed skill tool name successfully resolves description", () => {
  const skillTools = createSkillTools([todoSkill]);
  const desc = resolveToolDescription(
    skillTools,
    "todo_write",
    { todos: ["a", "b"] },
    [todoSkill],
  );
  assertEquals(desc, undefined);
});

Deno.test("run_command with bare tool name auto-corrects and routes correctly", async () => {
  const skillTools = createSkillTools([todoSkill]);
  const runCommand = skillTools.find((t) => t.name === "run_command");
  if (!runCommand) throw new Error("run_command missing");
  const out = await runCommand.handler(
    {
      command: "todo_write",
      params: { todos: ["a", "b"] },
      spinnerText: "writing",
    },
    "call-id",
  );
  if (typeof out !== "string") throw new Error("expected string result");
  assertEquals(out.includes("wrote 2 todos"), true);
});

Deno.test("run_command with incorrect prefix auto-corrects and routes correctly", async () => {
  const skillTools = createSkillTools([todoSkill]);
  const runCommand = skillTools.find((t) => t.name === "run_command");
  if (!runCommand) throw new Error("run_command missing");
  const out = await runCommand.handler(
    {
      command: "default_api/todo_write",
      params: { todos: ["a", "b"] },
      spinnerText: "writing",
    },
    "call-id",
  );
  if (typeof out !== "string") throw new Error("expected string result");
  assertEquals(out.includes("wrote 2 todos"), true);
});

Deno.test("formatSkillsPrompt outputs fully-qualified tool names", () => {
  const prompt = formatSkillsPrompt([todoSkill]);
  assertEquals(prompt.includes("- todo/todo_write:"), true);
  assertEquals(prompt.includes("- todo_write:"), false);
});

Deno.test("active skills prompt includes tool names and descriptions", () => {
  const spec: AgentSpec = {
    tools: [],
    skills: [todoSkill],
    prompt: "Help.",
  } as unknown as AgentSpec;

  const history = [
    {
      id: "call-1",
      type: "tool_call" as const,
      isOwn: true as const,
      name: "learn_skill",
      parameters: { skillName: "todo" },
      timestamp: 1000,
    },
    {
      id: "result-1",
      type: "tool_result" as const,
      isOwn: true as const,
      toolCallId: "call-1",
      result: "Skill learned successfully.",
      timestamp: 2000,
    },
  ];

  const specTurn2 = getSpecForTurn(spec, history);
  assertEquals(specTurn2.prompt.includes("todo/todo_write(params:"), true);
});
