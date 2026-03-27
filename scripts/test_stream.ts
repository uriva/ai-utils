import { injectGeminiToken, runAgent } from "../mod.ts";
import { injectAccessHistory } from "../src/agent.ts";
import "@std/dotenv/load";

const runner = injectGeminiToken(Deno.env.get("GEMINI_API_KEY")!)(
  injectAccessHistory(() => Promise.resolve([]))(() =>
    runAgent({
      prompt: "Say 'Hello, how are you today?' slowly.",
      tools: [],
      skills: [],
      onStreamChunk: (chunk) => console.log("CHUNK:", chunk),
      onOutputEvent: (ev) => {
        console.log("OUTPUT:", ev.type);
        return Promise.resolve();
      },
      onMaxIterationsReached: () => console.log("MAX ITERS"),
      maxIterations: 1,
      rewriteHistory: () => Promise.resolve(),
      timezoneIANA: "UTC",
    })
  ),
);

runner().catch(console.error);
