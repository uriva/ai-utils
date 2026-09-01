import { assert, assertEquals, assertFalse } from "@std/assert";
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
import {
  findUngroundedUtteranceArtifacts,
  isComplexUrl,
  isLikelyPhoneNumber,
  ungroundedHostBlockedNotice,
} from "../src/urlGrounding.ts";
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

Deno.test("isComplexUrl correctly classifies trivial vs complex URLs", () => {
  assertFalse(isComplexUrl("https://gmail.com"));
  assertFalse(isComplexUrl("http://google.com/"));
  assertFalse(isComplexUrl("https://www.youtube.com"));
  assertFalse(isComplexUrl("https://www.google.co.uk"));

  assert(isComplexUrl("https://tokenharbor.ai/v1"));
  assert(isComplexUrl("https://form.claude.com"));
  assert(isComplexUrl("https://docs.google.com/forms/d/123/viewform"));
  assert(isComplexUrl("https://cinema-events.org/pretty-woman"));
  assert(isComplexUrl("https://api.openai.com/v1/chat"));
  assert(isComplexUrl("https://x.com/user/status/123456789"));
  assert(isComplexUrl("https://service-host.org:8080"));
});

Deno.test("isLikelyPhoneNumber distinguishes phone numbers from dates, prices, and timestamps", () => {
  assert(isLikelyPhoneNumber("+380 67 352 2777"));
  assert(isLikelyPhoneNumber("+380975015774"));
  assert(isLikelyPhoneNumber("+1 (800) 123-4567"));
  assert(isLikelyPhoneNumber("050-1234567"));
  assert(isLikelyPhoneNumber("03-5252144"));
  assert(isLikelyPhoneNumber("tel:+972526966032"));

  assertFalse(isLikelyPhoneNumber("2026-08-20"));
  assertFalse(isLikelyPhoneNumber("20/08/2026"));
  assertFalse(isLikelyPhoneNumber("10:00"));
  assertFalse(isLikelyPhoneNumber("01:23:45"));
  assertFalse(isLikelyPhoneNumber("192.168.1.1"));
  assertFalse(isLikelyPhoneNumber("100"));
  assertFalse(isLikelyPhoneNumber("1866"));
});

Deno.test("findUngroundedUtteranceArtifacts extracts ungrounded complex URLs and phones", () => {
  const groundTruth = [
    "You are an event guide. Check events at https://example.com/events or call support at 03-5000000.",
    "User query: Tell me about screenings in Tel Aviv",
    "Tool result: Found event Movie Night at https://example.com/movie-night with contact +972 50 111 2222",
  ];

  const cleanReply = [
    "Here is the event https://example.com/movie-night or you can call +972 50 111 2222. You can also visit https://google.com for more info.",
  ];
  const cleanResult = findUngroundedUtteranceArtifacts(groundTruth, cleanReply);
  assertEquals(cleanResult.ungroundedUrls, []);
  assertEquals(cleanResult.ungroundedPhones, []);

  const hallucinatedReply = [
    "I booked your ticket! Details at https://tokenharbor.ai/v1/tickets/999 and notary Bodnarchuk Oksana at +380 67 352 2777.",
  ];
  const hallucinatedResult = findUngroundedUtteranceArtifacts(
    groundTruth,
    hallucinatedReply,
  );
  assertEquals(hallucinatedResult.ungroundedUrls, [
    "https://tokenharbor.ai/v1/tickets/999",
  ]);
  assertEquals(hallucinatedResult.ungroundedPhones, ["+380 67 352 2777"]);
});

Deno.test("utterance grounding gate blocks ungrounded complex URL in concluding reply", async () => {
  const history: HistoryEvent[] = [
    participantUtteranceTurn({ name: "user", text: "where is the form?" }),
  ];
  let callCount = 0;
  const seenThoughts: string[] = [];

  await injectCallModel((events: HistoryEvent[]) => {
    callCount++;
    const thoughts = events
      .filter((e) => e.type === "own_thought")
      .map((e) => ("text" in e ? e.text : ""));
    seenThoughts.push(...thoughts);
    if (callCount === 1) {
      return Promise.resolve([
        ownUtteranceTurn(
          "Here is the form: https://form.claude.com/survey/123",
        ),
      ]);
    }
    return Promise.resolve([
      ownUtteranceTurn("I do not have the direct form link yet."),
    ]);
  })(async () => {
    await agentDeps(history)(runAgent)({
      maxIterations: 3,
      prompt: "You are a helpful assistant.",
      tools: [],
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
  })();

  assert(
    seenThoughts.some((t) =>
      t.includes('unverified URL(s): "https://form.claude.com/survey/123"')
    ),
    "model should receive ungrounded URL correction thought",
  );
  const finalUtterance = history[history.length - 1];
  assert(
    finalUtterance.type === "own_utterance" &&
      finalUtterance.text === "I do not have the direct form link yet.",
    "final emitted message should be the corrected reply",
  );
});

Deno.test("utterance grounding gate blocks ungrounded phone number in concluding reply", async () => {
  const history: HistoryEvent[] = [
    participantUtteranceTurn({
      name: "user",
      text: "what is the notary's number?",
    }),
  ];
  let callCount = 0;
  const seenThoughts: string[] = [];

  await injectCallModel((events: HistoryEvent[]) => {
    callCount++;
    const thoughts = events
      .filter((e) => e.type === "own_thought")
      .map((e) => ("text" in e ? e.text : ""));
    seenThoughts.push(...thoughts);
    if (callCount === 1) {
      return Promise.resolve([
        ownUtteranceTurn(
          "The notary's direct phone is +380 67 352 2777.",
        ),
      ]);
    }
    return Promise.resolve([
      ownUtteranceTurn(
        "I do not have the verified phone number for the notary.",
      ),
    ]);
  })(async () => {
    await agentDeps(history)(runAgent)({
      maxIterations: 3,
      prompt: "You are a helpful assistant.",
      tools: [],
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
  })();

  assert(
    seenThoughts.some((t) =>
      t.includes('unverified phone number(s): "+380 67 352 2777"')
    ),
    "model should receive ungrounded phone number correction thought",
  );
  const finalUtterance = history[history.length - 1];
  assert(
    finalUtterance.type === "own_utterance" &&
      finalUtterance.text ===
        "I do not have the verified phone number for the notary.",
    "final emitted message should be the corrected reply",
  );
});

Deno.test("utterance grounding gate does not block example/placeholder domains", async () => {
  const history: HistoryEvent[] = [
    participantUtteranceTurn({
      name: "user",
      text: "how do I configure webhooks?",
    }),
  ];
  let callCount = 0;

  await injectCallModel((_events: HistoryEvent[]) => {
    callCount++;
    return Promise.resolve([
      ownUtteranceTurn(
        "You can configure your webhook endpoint at https://api.example.com/v1/webhook or https://your-server.example.org/events.",
      ),
    ]);
  })(async () => {
    await agentDeps(history)(runAgent)({
      maxIterations: 2,
      prompt: "You are a developer assistant.",
      tools: [],
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
  })();

  assertEquals(callCount, 1);
  const finalUtterance = history[history.length - 1];
  assert(
    finalUtterance.type === "own_utterance" &&
      finalUtterance.text.includes("https://api.example.com/v1/webhook"),
    "example domains must be delivered without blocking",
  );
});

Deno.test("utterance grounding gate does not block code snippets containing URLs or sample phones", async () => {
  const history: HistoryEvent[] = [
    participantUtteranceTurn({
      name: "user",
      text: "write python code to call an API",
    }),
  ];
  let callCount = 0;

  await injectCallModel((_events: HistoryEvent[]) => {
    callCount++;
    return Promise.resolve([
      ownUtteranceTurn(
        "Here is the sample code:\n```python\nimport requests\nresponse = requests.post('https://api.stripe.com/v1/charges', data={'phone': '+1 (555) 019-2834'})\n```",
      ),
    ]);
  })(async () => {
    await agentDeps(history)(runAgent)({
      maxIterations: 2,
      prompt: "You are a coding assistant.",
      tools: [],
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
  })();

  assertEquals(callCount, 1);
  const finalUtterance = history[history.length - 1];
  assert(
    finalUtterance.type === "own_utterance" &&
      finalUtterance.text.includes("https://api.stripe.com/v1/charges"),
    "code blocks must be delivered without blocking",
  );
});

Deno.test("utterance grounding gate respects model objection in internal thought for illustrative examples", async () => {
  const history: HistoryEvent[] = [
    participantUtteranceTurn({
      name: "user",
      text: "What is an example of a REST endpoint?",
    }),
  ];
  let callCount = 0;

  await injectCallModel((_events: HistoryEvent[]) => {
    callCount++;
    return Promise.resolve([
      ownThoughtTurn(
        "The user is asking for an educational illustrative example of a REST endpoint. I will cite the public GitHub API as an example, not as a ground-truth factual assertion about our system.",
      ),
      ownUtteranceTurn(
        "For instance, you can query public repositories via https://api.github.com/v3/repos/octocat/hello-world.",
      ),
    ]);
  })(async () => {
    await agentDeps(history)(runAgent)({
      maxIterations: 2,
      prompt: "You are a helpful programming tutor.",
      tools: [],
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
  })();

  assertEquals(callCount, 1);
  const finalUtterance = history[history.length - 1];
  assert(
    finalUtterance.type === "own_utterance" &&
      finalUtterance.text.includes("https://api.github.com/v3/repos"),
    "thought-justified illustrative examples must be delivered without blocking",
  );
});

Deno.test("utterance grounding gate allows URLs provided in proactive task notification thought", async () => {
  const taskUrl =
    "https://app.dashboard.example.org/threads?id=461a0a99-3377-49b3-90fd-13d7145c4e45";
  const history: HistoryEvent[] = [
    ownThoughtTurn(
      `PROACTIVE TASK: You have a task to complete: New thread matched\n\nLink: ${taskUrl}`,
    ),
  ];
  let callCount = 0;

  await injectCallModel((_events: HistoryEvent[]) => {
    callCount++;
    return Promise.resolve([
      ownUtteranceTurn(
        `New match detected! Review it here: ${taskUrl}`,
      ),
    ]);
  })(async () => {
    await agentDeps(history)(runAgent)({
      maxIterations: 2,
      prompt: "You are a helpful notification bot.",
      tools: [],
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
  })();

  assertEquals(callCount, 1);
  const finalUtterance = history[history.length - 1];
  assert(
    finalUtterance.type === "own_utterance" &&
      finalUtterance.text.includes(taskUrl),
    "URLs from proactive task notification must be delivered without blocking",
  );
});

Deno.test("utterance grounding gate allows URLs from compacted conversation summary", async () => {
  const summaryUrl = "https://dashboard.service.net/records/12345";
  const history: HistoryEvent[] = [
    ownThoughtTurn(
      `History compacted: 80 events → 14 summaries\n\nSummary 1:\n[This summary covers the period from May 1 to May 10]\nUser requested review for ${summaryUrl}.`,
    ),
    participantUtteranceTurn({
      name: "user",
      text: "what was the record link from earlier?",
    }),
  ];
  let callCount = 0;

  await injectCallModel((_events: HistoryEvent[]) => {
    callCount++;
    return Promise.resolve([
      ownUtteranceTurn(`Here is the link from earlier: ${summaryUrl}`),
    ]);
  })(async () => {
    await agentDeps(history)(runAgent)({
      maxIterations: 2,
      prompt: "You are a helpful assistant.",
      tools: [],
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
  })();

  assertEquals(callCount, 1);
  const finalUtterance = history[history.length - 1];
  assert(
    finalUtterance.type === "own_utterance" &&
      finalUtterance.text.includes(summaryUrl),
    "URLs from compacted summary must be delivered without blocking",
  );
});

Deno.test("url grounding gate allows tool calls targeting a host provided in proactive task notification", async () => {
  const taskHost = "dashboard.target-service.net";
  const probe = probeTool("fetch_data", "Fetch data from a URL", "url");
  const history: HistoryEvent[] = [
    ownThoughtTurn(
      `PROACTIVE TASK: You have a task to complete: New item at https://${taskHost}/items/123`,
    ),
  ];
  await runWithScriptedModel(
    { tools: [probe.tool], history },
    callOnceThenReply("fetch_data", { url: `https://${taskHost}/items/123` }),
  );
  assertExecuted(history, probe.wasExecuted);
});
