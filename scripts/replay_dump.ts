// Standalone replay — no ai-utils, no Zod. Posts the exact captured req to Gemini.
// Run: deno run --allow-net --allow-read --allow-env scripts/replay_dump.ts [N]
import "@std/dotenv/load";

const N = Number(Deno.args[0] ?? "5");
const apiKey = Deno.env.get("GEMINI_API_KEY");
if (!apiKey) {
  console.error("GEMINI_API_KEY missing");
  Deno.exit(1);
}

const req = JSON.parse(await Deno.readTextFile("/tmp/req.json"));
// strip non-serializable abortSignal placeholder if present
delete req.config?.abortSignal;

const model = req.model;
// Build googleapis payload shape (matches @google/genai generateContent)
const payload = {
  contents: req.contents,
  systemInstruction: { parts: [{ text: req.config.systemInstruction }] },
  tools: req.config.tools,
  toolConfig: req.config.toolConfig,
  generationConfig: {
    maxOutputTokens: req.config.maxOutputTokens,
    thinkingConfig: req.config.thinkingConfig,
  },
};

const url =
  `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;

const results: {
  i: number;
  ttfb: number;
  total: number;
  thoughtTok?: number;
  finishReason?: string;
  status: number;
}[] = [];

for (let i = 1; i <= N; i++) {
  const t0 = performance.now();
  const res = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  const ttfb = performance.now() - t0;
  const text = await res.text();
  const total = performance.now() - t0;
  let thoughtTok: number | undefined;
  let finishReason: string | undefined;
  if (res.ok) {
    const j = JSON.parse(text);
    thoughtTok = j.usageMetadata?.thoughtsTokenCount;
    finishReason = j.candidates?.[0]?.finishReason;
  } else {
    console.error(`iter ${i} HTTP ${res.status}: ${text.slice(0, 300)}`);
  }
  results.push({
    i,
    ttfb,
    total,
    thoughtTok,
    finishReason,
    status: res.status,
  });
  console.log(
    `iter=${i} status=${res.status} ttfb=${ttfb.toFixed(0)}ms total=${
      total.toFixed(0)
    }ms thoughtTok=${thoughtTok} finishReason=${finishReason}`,
  );
}

console.log("\nsummary:");
console.log(
  results.map((r) =>
    `  i=${r.i} total=${r.total.toFixed(0)}ms thoughtTok=${r.thoughtTok}`
  ).join("\n"),
);
