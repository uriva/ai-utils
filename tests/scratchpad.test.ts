import { assert } from "@std/assert";
import { z } from "zod/v4";
import { runAgent } from "../mod.ts";
import {
  compileGrepPattern,
  type HistoryEvent,
  injectCallModel,
  participantUtteranceTurn,
  readScratchFileToolName,
  type ToolOutputScratchPad,
} from "../src/agent.ts";
import {
  agentDeps,
  noopRewriteHistory,
  runForAllProviders,
} from "../test_helpers.ts";

const needle = "SECRET_TOKEN_a7f3b9c2_v2";

const makeBigBlob = () => {
  const filler = Array.from(
    { length: 800 },
    (_, i) => `line ${i + 1}: lorem ipsum dolor sit amet consectetur ${i}`,
  );
  filler[423] = `line 424: ${needle}`;
  return filler.join("\n");
};

const makeScratchPad = (store: Map<string, string>): ToolOutputScratchPad => ({
  set: (id, content) => {
    store.set(id, content);
    return Promise.resolve();
  },
  get: (id) => Promise.resolve(store.get(id)),
  threshold: 2000,
});

const dumpLogsTool = {
  name: "dump_logs",
  description:
    "Returns a large server log dump. Use read_scratch_file to inspect.",
  parameters: z.object({}),
  handler: () => Promise.resolve(makeBigBlob()),
};

runForAllProviders(
  "large tool outputs spill to scratch pad and are readable via read_scratch_file",
  async (runAgentWithProvider) => {
    const mockHistory: HistoryEvent[] = [participantUtteranceTurn({
      name: "user",
      text:
        `Call dump_logs, then find the SECRET_TOKEN value in its output and reply with just that token.`,
    })];
    const store = new Map<string, string>();
    const scratchPad = makeScratchPad(store);
    await agentDeps(mockHistory)(runAgentWithProvider)({
      maxIterations: 8,
      onMaxIterationsReached: () => {},
      tools: [dumpLogsTool],
      prompt:
        `You are an AI assistant. When a tool output is spilled to a scratch pad, use the ${readScratchFileToolName} tool (with the grep argument for regex search) to find what you need.`,
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
      toolOutputScratchPad: scratchPad,
    });

    const dumpResult = mockHistory.find(
      (e): e is Extract<HistoryEvent, { type: "tool_result" }> =>
        e.type === "tool_result" && e.result.includes("scratch pad with id"),
    );
    assert(
      dumpResult,
      `dump_logs result should be a spill notice. History: ${
        JSON.stringify(mockHistory, null, 2)
      }`,
    );
    assert(
      !dumpResult.result.includes(needle),
      "spill notice must not contain the needle (the blob was spilled, not inlined)",
    );
    assert(
      dumpResult.result.includes(`${readScratchFileToolName}({id:`),
      `spill notice should include a concrete example call. Got: ${dumpResult.result}`,
    );

    const readCall = mockHistory.find(
      (e): e is Extract<HistoryEvent, { type: "tool_call" }> =>
        e.type === "tool_call" && e.name === readScratchFileToolName,
    );
    assert(
      readCall,
      `agent should call ${readScratchFileToolName}. History: ${
        JSON.stringify(mockHistory, null, 2)
      }`,
    );

    const readResult = mockHistory.find(
      (e): e is Extract<HistoryEvent, { type: "tool_result" }> =>
        e.type === "tool_result" && e.toolCallId === readCall.id,
    );
    assert(
      readResult,
      `${readScratchFileToolName} should have a tool_result.`,
    );
    assert(
      /\[Scratch pad ".*": \d+ lines, \d+ chars total\.\]/.test(
        readResult.result,
      ),
      `read_scratch_file output should start with a size header. Got: ${
        readResult.result.slice(0, 200)
      }`,
    );

    const finalAnswer = [...mockHistory].reverse().find(
      (e): e is Extract<HistoryEvent, { type: "own_utterance" }> =>
        e.type === "own_utterance" && e.text.includes(needle),
    );
    assert(
      finalAnswer,
      `agent should reply with the needle ${needle}. History: ${
        JSON.stringify(mockHistory, null, 2)
      }`,
    );
  },
);

Deno.test("compileGrepPattern translates leading PCRE inline flags", () => {
  const r = compileGrepPattern("(?i)Shinjuku|Ramen");
  assert(r.ok, `expected compile success, got: ${JSON.stringify(r)}`);
  assert(r.re.flags.includes("i"));
  assert(r.re.test("shinjuku station"));
  assert(r.re.test("RAMEN house"));
});

Deno.test("compileGrepPattern handles combined PCRE flags (?ims)", () => {
  const r = compileGrepPattern("(?ims)^foo");
  assert(r.ok);
  ["i", "m", "s"].forEach((f) => assert(r.re.flags.includes(f)));
});

Deno.test("compileGrepPattern silently drops unsupported PCRE flag x", () => {
  const r = compileGrepPattern("(?ix)foo");
  assert(r.ok);
  assert(r.re.flags.includes("i"));
  assert(!r.re.flags.includes("x"));
});

Deno.test("compileGrepPattern returns error on invalid regex", () => {
  const r = compileGrepPattern("(unclosed");
  assert(!r.ok);
  assert(r.error.length > 0);
});

const fakeReadScratchCall = (
  id: string,
  grep: string,
): HistoryEvent => ({
  type: "tool_call",
  isOwn: true,
  name: readScratchFileToolName,
  parameters: { id, grep },
  id: "tc-grep",
  timestamp: Date.now(),
});

const runFakeGrepAgent = async (
  grep: string,
  content: string,
): Promise<HistoryEvent[]> => {
  const scratchId = "fake-scratch-id";
  const store = new Map<string, string>([[scratchId, content]]);
  const scratchPad = makeScratchPad(store);
  const mockHistory: HistoryEvent[] = [participantUtteranceTurn({
    name: "user",
    text: "look up the file",
  })];
  let n = 0;
  const fakeCallModel = () => {
    n++;
    if (n === 1) return Promise.resolve([fakeReadScratchCall(scratchId, grep)]);
    return Promise.resolve([{
      type: "own_utterance" as const,
      isOwn: true as const,
      text: "done",
      id: "u-done",
      timestamp: Date.now(),
    }]);
  };
  await injectCallModel(fakeCallModel)(async () => {
    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 4,
      onMaxIterationsReached: () => {},
      tools: [],
      prompt: "test",
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
      toolOutputScratchPad: scratchPad,
    });
  })();
  return mockHistory;
};

const findToolResult = (history: HistoryEvent[]) =>
  history.find((
    e,
  ): e is Extract<HistoryEvent, { type: "tool_result" }> =>
    e.type === "tool_result" && e.toolCallId === "tc-grep"
  );

Deno.test(
  "read_scratch_file with PCRE-style (?i) flag matches case-insensitively",
  async () => {
    const history = await runFakeGrepAgent(
      "(?i)Shinjuku|Ramen",
      "Tokyo notes\nshinjuku district\nRAMEN shop\nunrelated",
    );
    const result = findToolResult(history);
    assert(result, `expected tool_result. History: ${JSON.stringify(history)}`);
    assert(
      result.result.includes("shinjuku district") &&
        result.result.includes("RAMEN shop"),
      `expected case-insensitive matches. Got: ${result.result}`,
    );
  },
);

Deno.test(
  "read_scratch_file with invalid regex returns error to model instead of throwing",
  async () => {
    const history = await runFakeGrepAgent(
      "(unclosed",
      "anything\ngoes here",
    );
    const result = findToolResult(history);
    assert(result, `expected tool_result. History: ${JSON.stringify(history)}`);
    assert(
      result.result.toLowerCase().includes("invalid grep regex"),
      `expected error message in tool_result. Got: ${result.result}`,
    );
  },
);
