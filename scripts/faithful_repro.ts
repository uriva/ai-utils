import "@std/dotenv/load";
import { GoogleGenAI } from "@google/genai";
import { z } from "zod/v4";
import {
  buildReq,
  filterAndRewriteInvalidToolCalls,
} from "../src/geminiAgent.ts";

const report = JSON.parse(await Deno.readTextFile("/tmp/report.json"));
const a = report.agentRunInput;

// deno-lint-ignore no-explicit-any
const stubTool = (name: string): any => ({
  name,
  description: `Tool ${name}`,
  parameters: z.object({}).passthrough(),
  handler: () => Promise.resolve(""),
});

const skillToolNames = a.skills.flatMap((s: { toolNames: string[] }) =>
  s.toolNames
);
const allToolNames = [
  "run_command",
  "learn_skill",
  ...a.toolNames,
  ...skillToolNames,
];
const tools = allToolNames.map(stubTool);

const noopRewrite = () => Promise.resolve();
const filteredHistory = filterAndRewriteInvalidToolCalls(noopRewrite)(
  a.history,
);

const countToolCalls = (h: typeof a.history) =>
  h.filter((e: { kind: string }) => e.kind === "tool_call").length;
const countOwnThoughts = (h: typeof a.history) =>
  h.filter((e: { kind: string }) => e.kind === "own_thought").length;
const countRemovedPlaceholders = (h: typeof a.history) =>
  h.filter((e: { kind: string; content?: string }) =>
    e.kind === "own_thought" &&
    (e.content ?? "").startsWith("[Removed tool call")
  ).length;

console.log(
  `History before filter: events=${a.history.length} tool_calls=${
    countToolCalls(a.history)
  } own_thoughts=${countOwnThoughts(a.history)}`,
);
console.log(
  `History after  filter: events=${filteredHistory.length} tool_calls=${
    countToolCalls(filteredHistory)
  } own_thoughts=${countOwnThoughts(filteredHistory)} removedPlaceholders=${
    countRemovedPlaceholders(filteredHistory)
  }`,
);

const req = buildReq(
  false,
  a.model.lightModel,
  a.prompt,
  tools,
  a.model.timezoneIANA,
  a.model.maxOutputTokens,
)(filteredHistory);

console.log(
  `Built request: model=${req.model} systemInstructionLen=${
    (typeof req.config?.systemInstruction === "string"
      ? req.config.systemInstruction
      : "").length
  }`,
);

const sdk = new GoogleGenAI({ apiKey: Deno.env.get("GEMINI_API_KEY")! });

const runStream = async (label: string) => {
  const t0 = performance.now();
  const stream = await sdk.models.generateContentStream(req);
  let chunks = 0;
  let tFirst = 0;
  let thoughtChars = 0;
  let textChars = 0;
  let fnCalls = 0;
  let finish: string | undefined;
  let usage: unknown;
  for await (const chunk of stream) {
    chunks++;
    if (chunks === 1) tFirst = performance.now();
    const parts = chunk.candidates?.[0]?.content?.parts ?? [];
    for (const p of parts) {
      if (p.functionCall) fnCalls++;
      if (p.thought) thoughtChars += p.text?.length ?? 0;
      else if (typeof p.text === "string") textChars += p.text.length;
    }
    if (chunk.candidates?.[0]?.finishReason) {
      finish = chunk.candidates[0].finishReason;
    }
    if (chunk.usageMetadata) usage = chunk.usageMetadata;
  }
  const e = ((performance.now() - t0) / 1000).toFixed(1);
  const t = ((tFirst - t0) / 1000).toFixed(1);
  console.log(
    `[${label}] stream  total=${e}s ttfb=${t}s chunks=${chunks} thoughtChars=${thoughtChars} textChars=${textChars} fnCalls=${fnCalls} finish=${finish} usage=${
      JSON.stringify(usage)
    }`,
  );
};

const runBuffered = async (label: string) => {
  const t0 = performance.now();
  const res = await sdk.models.generateContent(req);
  const e = ((performance.now() - t0) / 1000).toFixed(1);
  const parts = res.candidates?.[0]?.content?.parts ?? [];
  const fnCalls = parts.filter((p) => p.functionCall).length;
  const thoughtChars = parts.filter((p) => p.thought).reduce(
    (s, p) => s + (p.text?.length ?? 0),
    0,
  );
  const textChars = parts.filter((p) =>
    !p.thought && typeof p.text === "string"
  ).reduce((s, p) => s + (p.text?.length ?? 0), 0);
  console.log(
    `[${label}] buffered total=${e}s thoughtChars=${thoughtChars} textChars=${textChars} fnCalls=${fnCalls} finish=${
      res.candidates?.[0]?.finishReason
    } usage=${JSON.stringify(res.usageMetadata)}`,
  );
};

const safe = (fn: () => Promise<void>, label: string) =>
  fn().catch((e) =>
    console.log(`[${label}] threw: ${(e as Error).message?.slice(0, 200)}`)
  );

await safe(() => runBuffered("buffered-1"), "buffered-1");
await safe(() => runBuffered("buffered-2"), "buffered-2");
await safe(() => runStream("stream-1"), "stream-1");
await safe(() => runStream("stream-2"), "stream-2");
