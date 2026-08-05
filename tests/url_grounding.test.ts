import { assert, assertFalse } from "@std/assert";
import { z } from "zod/v4";
import { runAgent, tool } from "../mod.ts";
import {
  type AgentSpec,
  type HistoryEvent,
  injectCallModel,
  ownThoughtTurn,
  ownUtteranceTurn,
  participantUtteranceTurn,
  toolResultTurn,
  toolUseTurn,
} from "../src/agent.ts";
import { ungroundedHostBlockedNotice } from "../src/urlGrounding.ts";
import { agentDeps, noopRewriteHistory } from "../test_helpers.ts";

const unseenHost = "api.neverseen.example";
const honestFallbackReply = "I don't have a documented way to do that.";

const probeTool = (name: string, description: string, paramName: string) => {
  let executed = false;
  return {
    wasExecuted: () => executed,
    tool: tool({
      name,
      description,
      parameters: z.object({ [paramName]: z.string() }),
      handler: () => {
        executed = true;
        return Promise.resolve("probe result");
      },
    }),
  };
};

type ScriptedRun = {
  prompt?: string;
  tools: AgentSpec["tools"];
  history: HistoryEvent[];
  extraSpec?: Record<string, unknown>;
};

const runWithScriptedModel = async (
  { prompt = "You are a helpful assistant.", tools, history, extraSpec = {} }:
    ScriptedRun,
  scripted: (callCount: number, events: HistoryEvent[]) => HistoryEvent[],
): Promise<HistoryEvent[][]> => {
  const seenByModel: HistoryEvent[][] = [];
  let callCount = 0;
  await injectCallModel((events: HistoryEvent[]) => {
    callCount++;
    seenByModel.push(events);
    return Promise.resolve(scripted(callCount, events));
  })(async () => {
    await agentDeps(history)(runAgent)({
      maxIterations: 4,
      prompt,
      tools,
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
      ...extraSpec,
    });
  })();
  return seenByModel;
};

// Keeps re-issuing the same tool call until the harness feeds back a thought
// mentioning the host (the correction notice), so a missing guard shows up as
// the tool executing rather than as an endless loop.
const callsHostUntilNoticed =
  (toolName: string, args: Record<string, unknown>, host: string) =>
  (callCount: number, events: HistoryEvent[]): HistoryEvent[] =>
    events.some((e) => e.type === "own_thought" && e.text.includes(host)) ||
      callCount >= 3
      ? [ownUtteranceTurn(honestFallbackReply)]
      : [toolUseTurn({ name: toolName, args })];

const callOnceThenReply =
  (toolName: string, args: Record<string, unknown>) =>
  (callCount: number): HistoryEvent[] =>
    callCount === 1
      ? [toolUseTurn({ name: toolName, args })]
      : [ownUtteranceTurn("done")];

const assertBlockedWithNotice = (
  history: HistoryEvent[],
  seenByModel: HistoryEvent[][],
  wasExecuted: () => boolean,
  host: string,
) => {
  assertFalse(
    wasExecuted(),
    "tool call targeting an ungrounded host must not execute",
  );
  assert(
    seenByModel.flat().some((e) =>
      e.type === "own_thought" &&
      e.text.includes(ungroundedHostBlockedNotice([host]))
    ),
    "model should receive the ungrounded-host correction thought",
  );
  assertFalse(
    history.some((e) => e.type === "tool_call"),
    "blocked tool call must not enter history",
  );
  assert(
    history.some((e) => e.type === "own_utterance"),
    "agent should conclude with an honest reply after the block",
  );
};

const assertExecuted = (
  history: HistoryEvent[],
  wasExecuted: () => boolean,
) => {
  assert(wasExecuted(), "tool call targeting a grounded host must execute");
  assert(
    history.some((e) => e.type === "tool_result" && e.result.includes("probe")),
    "tool result should be recorded",
  );
};

Deno.test("url grounding gate blocks tool call to a host absent from instructions and history", async () => {
  const probe = probeTool("fetch_data", "Fetch data from a URL", "url");
  const history: HistoryEvent[] = [
    participantUtteranceTurn({
      name: "user",
      text: "get me the latest train times",
    }),
  ];
  const seenByModel = await runWithScriptedModel(
    { tools: [probe.tool], history },
    callsHostUntilNoticed("fetch_data", {
      url: `https://${unseenHost}/api/trains/search`,
    }, unseenHost),
  );
  assertBlockedWithNotice(history, seenByModel, probe.wasExecuted, unseenHost);
});

Deno.test("url grounding gate ignores URLs embedded in generated content", async () => {
  const probe = probeTool("write_page", "Write a web page", "html");
  const history: HistoryEvent[] = [
    participantUtteranceTurn({
      name: "user",
      text: "build me a landing page",
    }),
  ];
  await runWithScriptedModel(
    { tools: [probe.tool], history },
    callOnceThenReply("write_page", {
      html:
        `<html><head><link href="https://${unseenHost}/fonts.css" rel="stylesheet"></head><body><img src="https://${unseenHost}/hero.jpg"></body></html>`,
    }),
  );
  assertExecuted(history, probe.wasExecuted);
});

Deno.test("url grounding gate blocks host literal inside code parameters", async () => {
  const probe = probeTool("run_script", "Execute a script", "code");
  const history: HistoryEvent[] = [
    participantUtteranceTurn({
      name: "user",
      text: "check the backend logs for errors",
    }),
  ];
  const seenByModel = await runWithScriptedModel(
    { tools: [probe.tool], history },
    callsHostUntilNoticed("run_script", {
      code:
        `check = (): string => {\n  r = httpRequest({ host: "${unseenHost}", path: "/api/logs", method: "GET" })\n  return r.body\n}`,
    }, unseenHost),
  );
  assertBlockedWithNotice(history, seenByModel, probe.wasExecuted, unseenHost);
});

Deno.test("url grounding gate does not treat the model's own thoughts as ground truth", async () => {
  const probe = probeTool("fetch_data", "Fetch data from a URL", "url");
  const history: HistoryEvent[] = [
    participantUtteranceTurn({
      name: "user",
      text: "check the backend logs for errors",
    }),
    ownThoughtTurn(
      `Maybe there is a logs endpoint at https://${unseenHost}/api/logs I can try`,
    ),
  ];
  const seenByModel = await runWithScriptedModel(
    { tools: [probe.tool], history },
    (callCount, events) =>
      events.some((e) =>
          e.type === "own_thought" &&
          e.text.includes(ungroundedHostBlockedNotice([unseenHost]))
        ) || callCount >= 3
        ? [ownUtteranceTurn(honestFallbackReply)]
        : [toolUseTurn({
          name: "fetch_data",
          args: { url: `https://${unseenHost}/api/logs` },
        })],
  );
  assertBlockedWithNotice(history, seenByModel, probe.wasExecuted, unseenHost);
});

Deno.test("url grounding gate allows a host the user provided", async () => {
  const userHost = "user-given.example";
  const probe = probeTool("fetch_data", "Fetch data from a URL", "url");
  const history: HistoryEvent[] = [
    participantUtteranceTurn({
      name: "user",
      text: `please fetch https://${userHost}/page and summarize it`,
    }),
  ];
  await runWithScriptedModel(
    { tools: [probe.tool], history },
    callOnceThenReply("fetch_data", { url: `https://${userHost}/page` }),
  );
  assertExecuted(history, probe.wasExecuted);
});

Deno.test("url grounding gate allows a host documented in tool descriptions despite www mismatch", async () => {
  const documentedHost = "docs-api.example";
  const probe = probeTool(
    "fetch_data",
    `Fetch data. Backed by the API at https://${documentedHost}`,
    "url",
  );
  const history: HistoryEvent[] = [
    participantUtteranceTurn({ name: "user", text: "get the data" }),
  ];
  await runWithScriptedModel(
    { tools: [probe.tool], history },
    callOnceThenReply("fetch_data", {
      url: `https://www.${documentedHost}/v1/data`,
    }),
  );
  assertExecuted(history, probe.wasExecuted);
});

Deno.test("url grounding gate allows a host that appeared in a prior tool result", async () => {
  const resultHost = "result-host.example";
  const probe = probeTool("open_link", "Open a link", "url");
  const searchCall = toolUseTurn({
    name: "search",
    args: { q: "train schedules" },
  });
  const history: HistoryEvent[] = [
    participantUtteranceTurn({
      name: "user",
      text: "find train schedules and open the top result",
    }),
    searchCall,
    toolResultTurn({
      toolCallId: searchCall.id,
      result: `Top result: https://${resultHost}/item/123`,
    }),
  ];
  await runWithScriptedModel(
    { tools: [probe.tool], history },
    callOnceThenReply("open_link", { url: `https://${resultHost}/item/123` }),
  );
  assertExecuted(history, probe.wasExecuted);
});

Deno.test("url grounding gate allows a host mentioned in the system prompt", async () => {
  const promptHost = "prompt-host.example";
  const probe = probeTool("fetch_data", "Fetch data from a URL", "url");
  const history: HistoryEvent[] = [
    participantUtteranceTurn({ name: "user", text: "check the status page" }),
  ];
  await runWithScriptedModel(
    {
      prompt:
        `You are a helpful assistant. The company status page is https://${promptHost}/status.`,
      tools: [probe.tool],
      history,
    },
    callOnceThenReply("fetch_data", { url: `https://${promptHost}/status` }),
  );
  assertExecuted(history, probe.wasExecuted);
});

Deno.test("url grounding gate skips tools exempted by the consumer", async () => {
  const probe = probeTool(
    "run_code",
    "Execute arbitrary code with network access",
    "code",
  );
  const history: HistoryEvent[] = [
    participantUtteranceTurn({
      name: "user",
      text: "check if that package exists",
    }),
  ];
  await runWithScriptedModel(
    {
      tools: [probe.tool],
      history,
      extraSpec: { urlGroundingExemptToolNames: ["run_code"] },
    },
    callOnceThenReply("run_code", {
      code: `await fetch("https://${unseenHost}/meta.json")`,
    }),
  );
  assertExecuted(history, probe.wasExecuted);
});

Deno.test("url grounding gate skips routed calls whose inner command is exempted", async () => {
  let executed = false;
  const runnerTool = tool({
    name: "skill_runner",
    description: "Run a skill command",
    parameters: z.object({
      command: z.string(),
      params: z.record(z.string(), z.unknown()),
    }),
    handler: () => {
      executed = true;
      return Promise.resolve("probe result");
    },
  });
  const history: HistoryEvent[] = [
    participantUtteranceTurn({
      name: "user",
      text: "run a shell command to check the registry",
    }),
  ];
  await runWithScriptedModel(
    {
      tools: [runnerTool],
      history,
      extraSpec: {
        urlGroundingExemptToolNames: ["code_execution/run_shell_command"],
      },
    },
    callOnceThenReply("skill_runner", {
      command: "code_execution/run_shell_command",
      params: { script: `curl https://${unseenHost}/pkg` },
    }),
  );
  assertExecuted(history, () => executed);
});

Deno.test("url grounding gate ignores prose that merely looks like a domain", async () => {
  const probe = probeTool("send_note", "Send a note to the user", "text");
  const history: HistoryEvent[] = [
    participantUtteranceTurn({
      name: "user",
      text: "tell the team where the docs are",
    }),
  ];
  await runWithScriptedModel(
    { tools: [probe.tool], history },
    callOnceThenReply("send_note", {
      text: "The docs are in README.md on the shared drive.",
    }),
  );
  assertExecuted(history, probe.wasExecuted);
});
