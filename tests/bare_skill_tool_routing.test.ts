import { assertEquals } from "@std/assert";
import { z } from "zod/v4";
import { callToResult, createSkillTools, tool } from "../src/agent.ts";
import type { Skill } from "../src/agent.ts";

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
    args: {},
    id: "call-4",
  });
  if (!out1) throw new Error("expected a result");
  assertEquals(out1.result.includes("todo skill"), true);

  const out2 = await callToResult(skillTools, [todoSkill])({
    name: "learn_skill",
    args: { skill: "todo" },
    id: "call-5",
  });
  if (!out2) throw new Error("expected a result");
  assertEquals(out2.result.includes("todo skill"), true);
});
