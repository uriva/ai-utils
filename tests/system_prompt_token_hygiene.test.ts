import { assertEquals } from "@std/assert";
import type { GenerateContentParameters } from "@google/genai";
import { runAgent } from "../mod.ts";
import {
  type AgentSpec,
  type HistoryEvent,
  injectAccessHistory,
  injectOutputEvent,
  participantUtteranceTurn,
} from "../src/agent.ts";
import { learnedSkillCall, weatherSkill } from "../test_helpers.ts";
import { injectGeminiSdkExchange } from "../src/geminiAgent.ts";
import { pipe } from "gamla";

// Gemini-specific seam: the system prompt is assembled inside each provider
// caller, and the Gemini exchange seam lets us capture the exact request
// without hitting the API.

type CapturedReq = { req: GenerateContentParameters };

const runCapturingSystemInstruction = async (
  specOverrides: Partial<AgentSpec>,
): Promise<string> => {
  const captured: CapturedReq[] = [];
  const history: HistoryEvent[] = [
    participantUtteranceTurn({ name: "user", text: "hello" }),
    // After the user message and freshly timestamped so the deferred-call
    // normalization keeps it in the model view (a dangling call older than
    // the latest user message gets substituted away).
    { ...learnedSkillCall(weatherSkill.name), timestamp: Date.now() + 1000 },
  ];
  await pipe(
    injectGeminiSdkExchange((_signal, args) => {
      captured.push({ req: args.req });
      return Promise.resolve({
        parts: [{ text: "Hi there!" }],
        finishReason: "STOP",
      });
    }),
    injectAccessHistory(() => Promise.resolve(history)),
    injectOutputEvent((event) => {
      history.push(event);
      return Promise.resolve();
    }),
  )(runAgent)({
    maxIterations: 5,
    tools: [],
    prompt: "You are a helpful assistant.",
    timezoneIANA: "UTC",
    skills: [weatherSkill],
    ...specOverrides,
  });
  assertEquals(captured.length, 1, "expected exactly one model call");
  const systemInstruction = captured[0].req.config?.systemInstruction;
  assertEquals(
    typeof systemInstruction,
    "string",
    "system instruction must be a string",
  );
  return systemInstruction as string;
};

const countOccurrences = (haystack: string, needle: string): number =>
  haystack.split(needle).length - 1;

Deno.test("active skill tools are described once in the system prompt", async () => {
  const systemInstruction = await runCapturingSystemInstruction({});
  assertEquals(
    countOccurrences(systemInstruction, "Always ask for location"),
    1,
    "active skill instructions must appear exactly once",
  );
  assertEquals(
    countOccurrences(systemInstruction, "Get weather forecast for a location"),
    1,
    "active skill tool descriptions must appear exactly once (they are also sent as function declarations)",
  );
});

Deno.test("scratch pad instructions only enter the system prompt when a scratch pad is configured", async () => {
  const withoutScratchPad = await runCapturingSystemInstruction({});
  assertEquals(
    withoutScratchPad.includes("SCRATCH PAD RULE"),
    false,
    "scratch pad instructions must be absent when no scratch pad is configured",
  );

  const withScratchPad = await runCapturingSystemInstruction({
    toolOutputScratchPad: {
      set: () => Promise.resolve(),
      get: () => Promise.resolve(undefined),
    },
  });
  assertEquals(
    withScratchPad.includes("SCRATCH PAD RULE"),
    true,
    "scratch pad instructions must be present when a scratch pad is configured",
  );
});
