import {
  type AgentSpec,
  type CallModel,
  generateId,
  type HistoryEvent,
  injectAccessHistory,
  injectCallModel,
  injectOutputEvent,
  injectShouldAbort,
  ownThoughtTurn,
  ownUtteranceTurn,
  participantUtteranceTurn,
  runAgent,
  tool,
  toolUseTurn,
  z,
} from "../mod.ts";
import { OpenAI } from "openai";
import { parseArgs } from "@std/cli/parse-args";
import { resolve } from "@std/path";

// Token and cost tracking
export type UsageStats = {
  promptTokens: number;
  completionTokens: number;
  cachedTokens: number;
  reasoningTokens: number;
  totalCalls: number;
};

const usageStats: UsageStats = {
  promptTokens: 0,
  completionTokens: 0,
  cachedTokens: 0,
  reasoningTokens: 0,
  totalCalls: 0,
};

// Pricing for GLM-5.3-Flash (USD per million tokens)
// Prompt / Miss: $0.10, Cache hit: $0.02, Output: $0.20
export const calculateGlmFlashCost = (stats: UsageStats): number => {
  const uncachedInput = Math.max(0, stats.promptTokens - stats.cachedTokens);
  const cost = (uncachedInput * 0.10 +
    stats.cachedTokens * 0.02 +
    stats.completionTokens * 0.20) /
    1_000_000;
  return cost;
};

// Factory for OpenAI-compatible CallModel
export const createGlmCallModel = (
  apiKey: string,
  model = "glm-5.3-flash",
  baseURL = "https://open.bigmodel.cn/api/paas/v4",
  stats: UsageStats = usageStats,
  onStreamThinking?: (chunk: string) => void,
) => {
  const client = new OpenAI({ apiKey, baseURL });

  return (spec: AgentSpec): CallModel => {
    return async (events: HistoryEvent[]): Promise<HistoryEvent[]> => {
      const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
        { role: "system", content: spec.prompt },
      ];

      const toolResultsById = new Map<string, string>();
      for (const e of events) {
        if (e.type === "tool_result") {
          const id = e.toolCallId || e.id;
          toolResultsById.set(
            id,
            typeof e.result === "string" ? e.result : JSON.stringify(e.result),
          );
        }
      }

      let i = 0;
      while (i < events.length) {
        const e = events[i];
        if (e.type === "participant_utterance") {
          messages.push({ role: "user", content: e.text });
          i++;
        } else if (e.type === "own_utterance") {
          messages.push({ role: "assistant", content: e.text });
          i++;
        } else if (e.type === "own_thought") {
          i++;
        } else if (e.type === "tool_call") {
          const currentToolCalls:
            OpenAI.Chat.Completions.ChatCompletionMessageToolCall[] = [];
          while (i < events.length && events[i].type === "tool_call") {
            const tc = events[i] as Extract<
              HistoryEvent,
              { type: "tool_call" }
            >;
            currentToolCalls.push({
              id: tc.id,
              type: "function",
              function: {
                name: tc.name,
                arguments: JSON.stringify(tc.parameters),
              },
            });
            i++;
          }
          messages.push({
            role: "assistant",
            content: null,
            tool_calls: currentToolCalls,
          });
          for (const tc of currentToolCalls) {
            const res = toolResultsById.get(tc.id) ?? "[no result]";
            messages.push({
              role: "tool",
              tool_call_id: tc.id,
              content: res,
            });
          }
        } else if (e.type === "tool_result") {
          i++;
        } else {
          i++;
        }
      }

      const tools: OpenAI.Chat.Completions.ChatCompletionTool[] = spec.tools
        .map(
          (t) => ({
            type: "function",
            function: {
              name: t.name,
              description: t.description,
              parameters: z.toJSONSchema(t.parameters) as Record<
                string,
                unknown
              >,
            },
          }),
        );

      const stream = await client.chat.completions.create({
        model,
        messages,
        tools: tools.length > 0 ? tools : undefined,
        stream: true,
        stream_options: { include_usage: true },
      });

      let reasoningText = "";
      let contentText = "";
      const toolCalls = new Map<
        number,
        { id: string; name: string; args: string }
      >();

      for await (const chunk of stream) {
        if (chunk.usage) {
          const u = chunk.usage;
          stats.totalCalls++;
          stats.promptTokens += u.prompt_tokens || 0;
          stats.completionTokens += u.completion_tokens || 0;
          const details = u as unknown as {
            prompt_tokens_details?: { cached_tokens?: number };
            completion_tokens_details?: { reasoning_tokens?: number };
          };
          if (details.prompt_tokens_details?.cached_tokens) {
            stats.cachedTokens += details.prompt_tokens_details.cached_tokens;
          }
          if (details.completion_tokens_details?.reasoning_tokens) {
            stats.reasoningTokens +=
              details.completion_tokens_details.reasoning_tokens;
          }
        }

        const delta = chunk.choices[0]?.delta as unknown as {
          content?: string;
          reasoning_content?: string;
          tool_calls?: Array<{
            index: number;
            id?: string;
            function?: { name?: string; arguments?: string };
          }>;
        };

        if (!delta) continue;

        if (delta.reasoning_content) {
          reasoningText += delta.reasoning_content;
          if (onStreamThinking) {
            onStreamThinking(delta.reasoning_content);
          }
        }
        if (delta.content) {
          contentText += delta.content;
        }
        if (delta.tool_calls) {
          for (const tc of delta.tool_calls) {
            const existing = toolCalls.get(tc.index) || {
              id: tc.id || generateId(),
              name: tc.function?.name || "",
              args: "",
            };
            if (tc.id) existing.id = tc.id;
            if (tc.function?.name) existing.name = tc.function.name;
            if (tc.function?.arguments) existing.args += tc.function.arguments;
            toolCalls.set(tc.index, existing);
          }
        }
      }

      const outputEvents: HistoryEvent[] = [];
      if (reasoningText) {
        outputEvents.push(ownThoughtTurn(reasoningText));
      }

      if (toolCalls.size > 0) {
        for (const tc of toolCalls.values()) {
          let args = {};
          try {
            args = JSON.parse(tc.args || "{}");
          } catch {
            args = { raw: tc.args };
          }
          outputEvents.push(
            toolUseTurn({
              id: tc.id,
              name: tc.name,
              args,
            }),
          );
        }
      } else if (contentText) {
        outputEvents.push(ownUtteranceTurn(contentText));
      }

      return outputEvents;
    };
  };
};

// SWE Agent tools
export const createSweTools = (
  workdir: string,
  onFinish: (message: string) => void,
) => {
  const bashTool = tool({
    name: "bash",
    description:
      "Run a bash command in the workspace directory. Returns stdout, stderr, and exit code. Long output is truncated.",
    parameters: z.object({
      command: z.string().describe("Bash command to execute"),
      timeout_seconds: z.number().optional().describe(
        "Timeout in seconds (default: 120)",
      ),
    }),
    handler: async ({ command, timeout_seconds = 120 }) => {
      try {
        const p = new Deno.Command("bash", {
          args: ["-c", command],
          cwd: workdir,
          stdout: "piped",
          stderr: "piped",
        });

        const child = p.spawn();

        const timeoutPromise = new Promise<
          { code: number; stdout: Uint8Array; stderr: Uint8Array }
        >(
          (_, reject) =>
            setTimeout(
              () =>
                reject(
                  new Error(`Command timed out after ${timeout_seconds}s`),
                ),
              timeout_seconds * 1000,
            ),
        );

        const result = await Promise.race([child.output(), timeoutPromise]);
        const stdoutStr = new TextDecoder().decode(result.stdout);
        const stderrStr = new TextDecoder().decode(result.stderr);

        let combined = stdoutStr;
        if (stderrStr) {
          combined += (combined ? "\nSTDERR:\n" : "") + stderrStr;
        }

        const maxChars = 25000;
        let truncated = false;
        if (combined.length > maxChars) {
          combined = combined.slice(0, maxChars) +
            `\n... [Output truncated. Total chars: ${combined.length}]`;
          truncated = true;
        }

        return JSON.stringify({
          exit_code: result.code,
          output: combined || "(no output)",
          truncated,
        });
      } catch (err) {
        return JSON.stringify({
          exit_code: -1,
          output: `Command failed: ${
            err instanceof Error ? err.message : String(err)
          }`,
        });
      }
    },
  });

  const readFileTool = tool({
    name: "read_file",
    description:
      "Read file contents with line numbers. Use offset and limit to read sections of large files.",
    parameters: z.object({
      file_path: z.string().describe("File path relative to the workspace"),
      offset: z.number().int().optional().describe(
        "1-indexed line number to start reading from",
      ),
      limit: z.number().int().optional().describe(
        "Number of lines to read (default: 2000)",
      ),
    }),
    handler: async ({ file_path, offset = 1, limit = 2000 }) => {
      try {
        const fullPath = resolve(workdir, file_path);
        const content = await Deno.readTextFile(fullPath);
        const lines = content.split("\n");
        const start = Math.max(0, offset - 1);
        const end = Math.min(lines.length, start + limit);
        const sliced = lines.slice(start, end);
        const numbered = sliced.map((line, idx) =>
          `${start + idx + 1}: ${line}`
        ).join("\n");
        return `File: ${file_path} (lines ${
          start + 1
        }-${end} of ${lines.length})\n${numbered}`;
      } catch (err) {
        return `Error reading ${file_path}: ${
          err instanceof Error ? err.message : String(err)
        }`;
      }
    },
  });

  const writeFileTool = tool({
    name: "write_file",
    description:
      "Write content to a file. Creates parent directories if needed. Overwrites existing file.",
    parameters: z.object({
      file_path: z.string().describe("File path relative to the workspace"),
      content: z.string().describe("Content to write into the file"),
    }),
    handler: async ({ file_path, content }) => {
      try {
        const fullPath = resolve(workdir, file_path);
        const dir = fullPath.substring(0, fullPath.lastIndexOf("/"));
        if (dir) {
          await Deno.mkdir(dir, { recursive: true });
        }
        await Deno.writeTextFile(fullPath, content);
        return `Successfully wrote ${content.length} characters to ${file_path}`;
      } catch (err) {
        return `Error writing to ${file_path}: ${
          err instanceof Error ? err.message : String(err)
        }`;
      }
    },
  });

  const editFileTool = tool({
    name: "edit_file",
    description:
      "Replace an exact string in a file with new text. The old_string must appear EXACTLY ONCE in the file.",
    parameters: z.object({
      file_path: z.string().describe("File path relative to the workspace"),
      old_string: z.string().describe(
        "Exact string to replace (must be unique)",
      ),
      new_string: z.string().describe("New replacement text"),
    }),
    handler: async ({ file_path, old_string, new_string }) => {
      try {
        const fullPath = resolve(workdir, file_path);
        const content = await Deno.readTextFile(fullPath);

        const count = content.split(old_string).length - 1;
        if (count === 0) {
          return `Error: old_string not found in ${file_path}. Make sure whitespace and line breaks match exactly.`;
        }
        if (count > 1) {
          return `Error: old_string found ${count} times in ${file_path}. Provide more surrounding context to make it unique.`;
        }

        const updated = content.replace(old_string, new_string);
        await Deno.writeTextFile(fullPath, updated);
        return `Successfully replaced occurrence in ${file_path}`;
      } catch (err) {
        return `Error editing ${file_path}: ${
          err instanceof Error ? err.message : String(err)
        }`;
      }
    },
  });

  const finishTool = tool({
    name: "finish",
    description:
      "Call this tool when you have completely solved and verified the task. Describe your solution.",
    parameters: z.object({
      message: z.string().describe(
        "Summary of the changes made and tests run to verify",
      ),
    }),
    handler: ({ message }) => {
      onFinish(message);
      return Promise.resolve("Task marked as finished.");
    },
  });

  return [bashTool, readFileTool, writeFileTool, editFileTool, finishTool];
};

const SWE_SYSTEM_PROMPT =
  `You are an expert autonomous software engineer solving technical issues and tasks in a Linux environment.

Guidelines:
1. First, explore the repository and understand the directory structure, build system, and tests.
2. If there is a bug or problem to solve, reproduce it or inspect failing tests first.
3. Use read_file to inspect code before editing.
4. Use edit_file or write_file to apply precise fixes.
5. After making changes, run relevant tests or build scripts using bash to verify your fix.
6. When the task is completely finished and verified, call the "finish" tool with a summary or provide your final conclusion.
7. Always check command exit codes and error messages. If a command fails, diagnose the root cause and fix it.`;

export const runSweAgent = async ({
  task,
  workdir = Deno.cwd(),
  maxIterations = 30,
  apiKey,
  model = "glm-5.3-flash",
  onEvent,
}: {
  task: string;
  workdir?: string;
  maxIterations?: number;
  apiKey?: string;
  model?: string;
  onEvent?: (event: HistoryEvent) => void;
}) => {
  const resolvedKey = apiKey || Deno.env.get("ZAI_API_KEY") ||
    Deno.env.get("ZHIPU_API_KEY");
  if (!resolvedKey) {
    throw new Error(
      "ZAI_API_KEY or ZHIPU_API_KEY environment variable required",
    );
  }

  let isDone = false;
  let finishMessage = "";

  const onFinish = (msg: string) => {
    isDone = true;
    finishMessage = msg;
  };

  const tools = createSweTools(workdir, onFinish);

  const history: HistoryEvent[] = [
    participantUtteranceTurn({
      name: "user",
      text: task,
    }),
  ];

  const spec: AgentSpec = {
    provider: "moonshot", // Use moonshot provider mode so history is preserved without Gemini signatures
    prompt: SWE_SYSTEM_PROMPT,
    tools,
    maxIterations,
    timezoneIANA: "UTC",
  };

  const caller = createGlmCallModel(
    resolvedKey,
    model,
    undefined,
    usageStats,
    (chunk) => {
      // Print dot or short progress for thinking
      Deno.stdout.writeSync(new TextEncoder().encode(chunk));
    },
  )(spec);

  await injectAccessHistory(() => Promise.resolve(history))(
    injectOutputEvent((event) => {
      if (onEvent) onEvent(event);
      if (event.type === "own_utterance" && event.text && event.text.trim()) {
        isDone = true;
        if (!finishMessage) finishMessage = event.text;
      }
      history.push(event);
      return Promise.resolve();
    })(
      injectShouldAbort(() => Promise.resolve(isDone))(
        injectCallModel(caller)(() => runAgent(spec)),
      ),
    ),
  )();

  const cost = calculateGlmFlashCost(usageStats);

  return {
    history,
    isDone,
    finishMessage,
    usage: usageStats,
    costUsd: cost,
  };
};

if (import.meta.main) {
  const args = parseArgs(Deno.args, {
    string: ["task", "workdir", "output", "model"],
    boolean: ["verbose"],
    alias: { t: "task", w: "workdir", o: "output", m: "model", v: "verbose" },
  });

  const task = args.task || args._[0];
  if (!task || typeof task !== "string") {
    console.error('Usage: swe_agent --task "<instruction>" [--workdir <dir>]');
    Deno.exit(1);
  }

  const workdir = args.workdir ? resolve(args.workdir) : Deno.cwd();
  console.log(`Starting SWE Agent in ${workdir}`);
  console.log(`Task: ${task}\n`);

  const startTime = Date.now();

  const result = await runSweAgent({
    task,
    workdir,
    model: args.model || "glm-5.3-flash",
    onEvent: (event) => {
      if (args.verbose) {
        if (event.type === "own_thought") {
          console.log(`\n💭 [Thought]:\n${event.text}`);
        } else if (event.type === "tool_call") {
          console.log(
            `\n🛠️  [Action]: ${event.name}(${
              JSON.stringify(event.parameters)
            })`,
          );
        } else if (event.type === "tool_result") {
          console.log(`📋 [Observation]: ${event.result.slice(0, 300)}...`);
        } else if (event.type === "own_utterance") {
          console.log(`\n🤖 [Response]:\n${event.text}`);
        }
      } else {
        if (event.type === "tool_call") {
          console.log(`-> Tool call: ${event.name}`);
        }
      }
    },
  });

  const durationSec = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log(`\n=== Run Complete (${durationSec}s) ===`);
  console.log(`Finished: ${result.isDone}`);
  if (result.finishMessage) {
    console.log(`Summary: ${result.finishMessage}`);
  }
  console.log(`Token Usage: ${JSON.stringify(result.usage, null, 2)}`);
  console.log(`Estimated Cost: $${result.costUsd.toFixed(6)} USD`);

  if (args.output) {
    await Deno.writeTextFile(args.output, JSON.stringify(result, null, 2));
    console.log(`Wrote trajectory to ${args.output}`);
  }

  Deno.exit(0);
}
