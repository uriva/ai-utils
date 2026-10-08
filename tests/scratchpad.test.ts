import { assert, assertEquals } from "@std/assert";
import { z } from "zod/v4";
import { runAgent } from "../mod.ts";
import {
  compileGrepPattern,
  createReadScratchFileTool,
  type HistoryEvent,
  injectCallModel,
  maxToolOutputChars,
  participantUtteranceTurn,
  readScratchFileToolName,
  type ToolOutputScratchPad,
  truncateToolOutput,
} from "../src/agent.ts";
import { agentDeps, runForAllProviders } from "../test_helpers.ts";

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
        `Call dump_logs, then find the SECRET_TOKEN value in its output and reply with the exact token name including its prefix (e.g. starting with SECRET_TOKEN_).`,
    })];
    const store = new Map<string, string>();
    const scratchPad = makeScratchPad(store);
    await agentDeps(mockHistory)(runAgentWithProvider)({
      maxIterations: 8,
      tools: [dumpLogsTool],
      prompt:
        `You are an AI assistant. When a tool output is spilled to a scratch pad, use the ${readScratchFileToolName} tool (with the grep argument for regex search) to find what you need.`,
      timezoneIANA: "UTC",
      toolOutputScratchPad: scratchPad,
    });

    const dumpResult = mockHistory.find(
      (e): e is Extract<HistoryEvent, { type: "tool_result" }> =>
        e.type === "tool_result" &&
        e.result.includes("Tool output was truncated"),
    );
    assert(
      dumpResult,
      `dump_logs result should be a spill notice. History: ${
        JSON.stringify(mockHistory, null, 2)
      }`,
    );
    assert(
      !dumpResult.result.includes(needle),
      "needle lives past the preview window, so it must not appear in the spill result (would defeat the spill test)",
    );
    assert(
      dumpResult.result.includes("line 1: lorem ipsum"),
      `spill result should inline the first chunk as a preview. Got: ${
        dumpResult.result.slice(0, 300)
      }`,
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

    const readCallIndex = mockHistory.indexOf(readCall);
    const readResult = mockHistory.slice(readCallIndex + 1).find(
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
  startLine?: number,
  limit?: number,
): HistoryEvent => ({
  type: "tool_call",
  isOwn: true,
  name: readScratchFileToolName,
  parameters: {
    id,
    grep,
    ...(startLine ? { startLine } : {}),
    ...(limit ? { limit } : {}),
  },
  id: "tc-grep",
  timestamp: Date.now(),
});

const runFakeGrepAgent = async (
  grep: string,
  content: string,
  startLine?: number,
  limit?: number,
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
    if (n === 1) {
      return Promise.resolve([
        fakeReadScratchCall(scratchId, grep, startLine, limit),
      ]);
    }
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
      tools: [],
      prompt: "test",
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
  "read_scratch_file grep windows very long lines around the match instead of returning them whole",
  async () => {
    const longLine = `${"P".repeat(10000)}the-needle-here${"S".repeat(10000)}`;
    const history = await runFakeGrepAgent("needle", longLine);
    const result = findToolResult(history);
    assert(result, `expected tool_result. History: ${JSON.stringify(history)}`);
    assert(
      result.result.includes("the-needle-here"),
      `expected the match to appear in the window. Got: ${
        result.result.slice(0, 300)
      }`,
    );
    assert(
      !result.result.includes("P".repeat(2000)),
      `a 20k-char matching line must not be returned whole (window it around the match). Got ${result.result.length} chars`,
    );
    assert(
      result.result.includes("S".repeat(400)),
      `window should include context after the match. Got: ${
        result.result.slice(0, 300)
      }`,
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

Deno.test(
  "read_scratch_file with grep and startLine paginates lines from startLine",
  async () => {
    const store = new Map<string, string>();
    const scratchPad = makeScratchPad(store);
    const lines = Array.from(
      { length: 100 },
      (_, i) => `row ${i + 1}: data`,
    ).join("\n");
    store.set("test-paginate-scratch", lines);
    const tool = createReadScratchFileTool(scratchPad);

    const page1 = await tool.handler({
      id: "test-paginate-scratch",
      grep: ".*",
      startLine: 1,
      limit: 40,
    }, "tc-p1");
    if (typeof page1 !== "string") throw new Error("expected string result");
    assert(
      page1.includes("1: row 1: data"),
      `page 1 should start at line 1. Got: ${page1.slice(0, 200)}`,
    );
    assert(page1.includes("40: row 40: data"), "page 1 should end at line 40");
    assert(
      !page1.includes("41: row 41: data"),
      "page 1 must not include line 41",
    );
    assert(
      page1.includes("startLine=41"),
      `page 1 continuation must recommend startLine=41. Got: ${
        page1.slice(-150)
      }`,
    );

    const page2 = await tool.handler({
      id: "test-paginate-scratch",
      grep: ".*",
      startLine: 41,
      limit: 40,
    }, "tc-p2");
    if (typeof page2 !== "string") throw new Error("expected string result");
    assert(
      page2.includes("41: row 41: data"),
      `page 2 should start at line 41. Got: ${page2.slice(0, 200)}`,
    );
    assert(page2.includes("80: row 80: data"), "page 2 should end at line 80");
    assert(!page2.includes("1: row 1: data"), "page 2 must not repeat line 1");
    assert(
      page2.includes("startLine=81"),
      `page 2 continuation must recommend startLine=81. Got: ${
        page2.slice(-150)
      }`,
    );

    const page3 = await tool.handler({
      id: "test-paginate-scratch",
      grep: ".*",
      startLine: 81,
      limit: 40,
    }, "tc-p3");
    if (typeof page3 !== "string") throw new Error("expected string result");
    assert(
      page3.includes("81: row 81: data"),
      `page 3 should start at line 81. Got: ${page3.slice(0, 200)}`,
    );
    assert(
      page3.includes("100: row 100: data"),
      "page 3 should end at line 100",
    );
    assert(
      !page3.includes("startLine="),
      "page 3 is end of file, must not have continuation prompt",
    );
  },
);

Deno.test(
  "read_scratch_file with specific pattern filters only matching lines at or after startLine",
  async () => {
    const store = new Map<string, string>();
    const scratchPad = makeScratchPad(store);
    const content = [
      "line 1: ignore",
      "line 10: target alpha",
      "line 25: ignore",
      "line 50: target beta",
      "line 75: ignore",
      "line 90: target gamma",
    ].join("\n");
    store.set("test-pattern-scratch", content);
    const tool = createReadScratchFileTool(scratchPad);

    const result = await tool.handler({
      id: "test-pattern-scratch",
      grep: "target",
      startLine: 3,
      limit: 10,
    }, "tc-target");
    if (typeof result !== "string") throw new Error("expected string result");
    assert(
      !result.includes("target alpha"),
      "should not include match before line 3",
    );
    assert(
      result.includes("4: line 50: target beta"),
      "should include match at line 4 (line 50 in content)",
    );
    assert(
      result.includes("6: line 90: target gamma"),
      "should include match at line 6 (line 90 in content)",
    );
  },
);

Deno.test(
  "agent-level: read_scratch_file paginates with grep and startLine",
  async () => {
    const history = await runFakeGrepAgent(
      ".*",
      Array.from({ length: 150 }, (_, i) => `item ${i + 1}`).join("\n"),
      81,
      50,
    );
    const result = findToolResult(history);
    assert(result, `expected tool_result. History: ${JSON.stringify(history)}`);
    assert(
      result.result.includes("\n81: item 81\n"),
      `expected result to start from line 81. Got: ${
        result.result.slice(0, 300)
      }`,
    );
    assert(
      !result.result.includes("\n1: item 1\n"),
      "expected result to not include line 1",
    );
    assert(
      result.result.includes("130: item 130"),
      "expected result to show 50 lines ending at 130",
    );
    assert(
      result.result.includes("startLine=131"),
      "expected continuation prompt for startLine=131",
    );
  },
);

Deno.test("truncateToolOutput trims the middle of very large texts and places a nice marker", () => {
  const shortText = "Hello World";
  assert(truncateToolOutput(shortText) === shortText);

  const marker = "\n\n<content trimmed due to length>\n\n";
  const prefix = "A".repeat(15000);
  const suffix = "B".repeat(15000);
  const largeText = prefix + suffix;
  const truncated = truncateToolOutput(largeText);

  assert(truncated.length === maxToolOutputChars);
  assert(truncated.includes(marker));

  const expectedKeepStart = Math.ceil((maxToolOutputChars - marker.length) / 2);
  const expectedKeepEnd = Math.floor((maxToolOutputChars - marker.length) / 2);
  assert(truncated.startsWith("A".repeat(expectedKeepStart)));
  assert(truncated.endsWith("B".repeat(expectedKeepEnd)));
});

Deno.test(
  "universal SCRATCH: resolution replaces prefix with full content before tool handler runs",
  {
    sanitizeResources: false,
  },
  async () => {
    const store = new Map<string, string>();
    const scratchPad = makeScratchPad(store);

    const scratchId = "test-nested-scratch-id-123";
    const testContent = "This is fully resolved content from the scratchpad!";
    store.set(scratchId, testContent);

    // deno-lint-ignore no-explicit-any
    let capturedParams: any = null;

    const testTool = {
      name: "test_resolution",
      description: "A test tool",
      parameters: z.object({
        nested: z.object({
          content: z.string(),
        }),
      }),
      // deno-lint-ignore no-explicit-any
      handler: (params: any) => {
        capturedParams = params;
        return Promise.resolve("Success");
      },
    };

    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Call test_resolution",
      }),
    ];

    let callCount = 0;
    await injectCallModel((events) => {
      callCount += 1;
      if (callCount > 1) {
        return Promise.resolve([
          ...events,
          {
            type: "own_utterance" as const,
            isOwn: true as const,
            text: "Done",
            id: "done-msg",
            timestamp: Date.now(),
          },
        ]);
      }
      return Promise.resolve([
        ...events,
        {
          type: "tool_call" as const,
          isOwn: true as const,
          name: "test_resolution",
          parameters: {
            nested: {
              content: `SCRATCH:${scratchId}`,
            },
          },
          id: "call-1",
          timestamp: Date.now(),
        },
      ]);
    })(async () => {
      await agentDeps(mockHistory)(runAgent)({
        maxIterations: 1,
        tools: [testTool],
        prompt: "Test assistant",
        timezoneIANA: "UTC",
        toolOutputScratchPad: scratchPad,
      });
    })();

    assertEquals(capturedParams, {
      nested: {
        content: testContent,
      },
    });
  },
);

Deno.test(
  "scratchpad unpacks JSON/github file content wrappers so line counting, preview, grep, and SCRATCH: resolution use real file text",
  async () => {
    const store = new Map<string, string>();
    const scratchPad = makeScratchPad(store);

    const codeBlocks = Array.from(
      { length: 100 },
      (_, i) =>
        [
          `// Module section ${i + 1} definition`,
          `export function computeValue${i + 1}(input: number): number {`,
          `  const multiplier = ${i * 7 + 3};`,
          `  const offset = ${i * 13 + 5};`,
          `  return input * multiplier + offset;`,
          `}`,
        ].join("\n"),
    );
    codeBlocks[50] = [
      `// Target section`,
      `export function targetFunction(arg: string): string {`,
      `  return "FOUND_TARGET_HERES_THE_CODE";`,
      `}`,
    ].join("\n");
    const realCode = codeBlocks.join("\n");

    const githubTool = {
      name: "github_get_file_contents",
      description: "Get file contents from github",
      parameters: z.object({ path: z.string() }),
      handler: () =>
        Promise.resolve(JSON.stringify({
          success: true,
          result: realCode,
        })),
    };

    let capturedInOtherTool: string | null = null;
    const saveTool = {
      name: "save_file",
      description: "Save file",
      parameters: z.object({ content: z.string() }),
      handler: (params: { content: string }) => {
        capturedInOtherTool = params.content;
        return Promise.resolve("saved");
      },
    };

    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Fetch file and inspect it",
      }),
    ];

    let callCount = 0;
    let spilledId = "";

    await injectCallModel((events) => {
      callCount++;
      if (callCount === 1) {
        return Promise.resolve([
          {
            type: "tool_call" as const,
            isOwn: true as const,
            name: "github_get_file_contents",
            parameters: { path: "src/server.ts" },
            id: "call-gh-1",
            timestamp: Date.now(),
          },
        ]);
      }
      if (callCount === 2) {
        const ghResult = events.find((e) =>
          e.type === "tool_result" && e.toolCallId === "call-gh-1"
        ) as Extract<HistoryEvent, { type: "tool_result" }> | undefined;
        assert(ghResult, "Expected ghResult");
        assert(
          ghResult.result.includes("598 lines total"),
          `Expected spill notice to report 598 lines total, got: ${ghResult.result}`,
        );
        assert(
          ghResult.result.includes("export function computeValue1"),
          `Expected preview to contain real code, got: ${
            ghResult.result.slice(0, 200)
          }`,
        );
        assert(
          !ghResult.result.includes('"success": true'),
          "Preview should not contain JSON wrapper syntax",
        );
        spilledId = "call-gh-1";

        return Promise.resolve([
          {
            type: "tool_call" as const,
            isOwn: true as const,
            name: readScratchFileToolName,
            parameters: { id: spilledId, offset: 300, limit: 10 },
            id: "call-read-offset",
            timestamp: Date.now(),
          },
        ]);
      }
      if (callCount === 3) {
        const readResult = events.find((e) =>
          e.type === "tool_result" && e.toolCallId === "call-read-offset"
        ) as Extract<HistoryEvent, { type: "tool_result" }> | undefined;
        assert(readResult, "Expected readResult");
        assert(
          readResult.result.includes("targetFunction"),
          "Expected targetFunction in offset slice",
        );

        return Promise.resolve([
          {
            type: "tool_call" as const,
            isOwn: true as const,
            name: readScratchFileToolName,
            parameters: { id: spilledId, grep: "targetFunction" },
            id: "call-read-grep",
            timestamp: Date.now(),
          },
        ]);
      }
      if (callCount === 4) {
        const grepResult = events.find((e) =>
          e.type === "tool_result" && e.toolCallId === "call-read-grep"
        ) as Extract<HistoryEvent, { type: "tool_result" }> | undefined;
        assert(grepResult, "Expected grepResult");
        assert(
          grepResult.result.includes("302: export function targetFunction"),
          `Expected grep to match line 302 with real line number, got: ${grepResult.result}`,
        );

        return Promise.resolve([
          {
            type: "tool_call" as const,
            isOwn: true as const,
            name: "save_file",
            parameters: { content: `SCRATCH:${spilledId}` },
            id: "call-save",
            timestamp: Date.now(),
          },
        ]);
      }
      return Promise.resolve([
        {
          type: "own_utterance" as const,
          isOwn: true as const,
          text: "All done",
          id: "done-msg",
          timestamp: Date.now(),
        },
      ]);
    })(async () => {
      await agentDeps(mockHistory)(runAgent)({
        provider: "moonshot",
        maxIterations: 5,
        tools: [githubTool, saveTool],
        prompt: "Test assistant",
        timezoneIANA: "UTC",
        toolOutputScratchPad: scratchPad,
      });
    })();

    assertEquals(capturedInOtherTool, realCode);
  },
);

Deno.test(
  "scratchpad automatically decodes base64-encoded GitHub API content payloads",
  async () => {
    const store = new Map<string, string>();
    const scratchPad = makeScratchPad(store);

    const lines = Array.from(
      { length: 250 },
      (_, i) =>
        `line ${i + 1}: code content for testing base64 decode ${i + 1}`,
    );
    const code = lines.join("\n");
    const base64Content = btoa(code);

    const githubApiTool = {
      name: "github_api_get_content",
      description: "Direct GitHub API contents endpoint",
      parameters: z.object({ path: z.string() }),
      handler: () =>
        Promise.resolve(JSON.stringify({
          name: "index.ts",
          path: "src/index.ts",
          sha: "abc12345",
          size: code.length,
          encoding: "base64",
          content: base64Content,
        })),
    };

    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Fetch github content",
      }),
    ];

    let callCount = 0;
    await injectCallModel(() => {
      callCount++;
      if (callCount === 1) {
        return Promise.resolve([
          {
            type: "tool_call" as const,
            isOwn: true as const,
            name: "github_api_get_content",
            parameters: { path: "src/index.ts" },
            id: "call-gh-b64",
            timestamp: Date.now(),
          },
        ]);
      }
      return Promise.resolve([
        {
          type: "own_utterance" as const,
          isOwn: true as const,
          text: "Done",
          id: "done-msg",
          timestamp: Date.now(),
        },
      ]);
    })(async () => {
      await agentDeps(mockHistory)(runAgent)({
        provider: "moonshot",
        maxIterations: 2,
        tools: [githubApiTool],
        prompt: "Test assistant",
        timezoneIANA: "UTC",
        toolOutputScratchPad: scratchPad,
      });
    })();

    const stored = await scratchPad.get("call-gh-b64");
    assertEquals(stored, code);
  },
);
