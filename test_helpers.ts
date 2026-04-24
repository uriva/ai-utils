import type { Injector } from "@uri/inject";
import { pipe } from "gamla";
import { cache, waitAllWrites } from "rmmbr";
import { z } from "zod/v4";
import {
  injectAnthropicToken,
  injectGeminiToken,
  injectKimiToken,
  injectOpenAiToken,
  overrideIdGenerator,
  runAgent,
  tool,
} from "./mod.ts";
import {
  type AgentSpec,
  type CallModelWrapper,
  type HistoryEvent,
  injectAccessHistory,
  injectCallModelWrapper,
  injectMetadataStore,
  injectOutputEvent,
  type ToolReturn,
} from "./src/agent.ts";

// Deterministic id generation so rmmbr cache keys are stable across runs.
// We do NOT override timestamp here — tests that care about time should
// override it explicitly (see `pipe(injectSecrets, overrideTime(...))`).
// Non-overriding tests will have wall-clock timestamps, which means their
// cache keys churn and they effectively run uncached — acceptable for now.
const makeCounter = (prefix: string) => {
  let n = 0;
  return () => `${prefix}-${++n}`;
};

const requireEnv = (name: string) => {
  const v = Deno.env.get(name);
  if (!v) throw new Error(`Missing required env var: ${name}`);
  return v;
};

const rmmbrToken = requireEnv("RMMBR_TOKEN");

// Forces every cache site to pass an explicit customKeyFn. rmmbr's default
// key function hashes the raw args, which silently includes non-deterministic
// fields (timestamps, random ids) — causing cache-miss storms that waste API
// dollars on every CI run. Crashing loudly when no customKeyFn is given is
// cheaper than learning about it from a bill.
const rmmbrCacheWithKey = <Args extends unknown[], R>(
  cacheId: string,
  customKeyFn: (...args: Args) => unknown,
  fn: (...args: Args) => Promise<R>,
): (...args: Args) => Promise<R> => {
  if (!customKeyFn) {
    throw new Error(
      `rmmbrCacheWithKey(${cacheId}) requires an explicit customKeyFn; ` +
        "without one, rmmbr would hash all args (including any timestamps " +
        "or random ids) and every call would miss the cache.",
    );
  }
  return cache({
    cacheId,
    ttl: 60 * 60 * 24 * 30,
    url: "https://rmmbr.net",
    token: rmmbrToken,
    customKeyFn: customKeyFn as (...args: unknown[]) => unknown,
  })(fn as (...args: unknown[]) => Promise<R>) as (
    ...args: Args
  ) => Promise<R>;
};

// Strip timestamps and (randomly-generated) ids before hashing so stable
// content → stable key. Only the content that semantically identifies a
// model request is kept.
const eventsCacheKey = (events: HistoryEvent[]) =>
  events.map(({ id: _id, timestamp: _ts, ...rest }) => rest);

const cachingCallModelWrapper: CallModelWrapper = ({ provider, inner }) => {
  const cached = rmmbrCacheWithKey(
    `callModel-${provider ?? "google"}-v4`,
    (events: HistoryEvent[]) => eventsCacheKey(events),
    (events: HistoryEvent[]) => inner(events),
  );
  return (events) => cached(events);
};

const flushRmmbr = (f: () => Promise<void>) => async () => {
  try {
    await f();
  } finally {
    await waitAllWrites();
  }
};

// Fresh id counter per test so each test produces the same sequence of ids
// regardless of ordering with other tests.
const injectDeterministic = (f: () => Promise<void>) => () =>
  overrideIdGenerator(makeCounter("id"))(f)();

export const injectSecrets = pipe(
  flushRmmbr,
  injectDeterministic,
  injectCallModelWrapper(cachingCallModelWrapper),
  injectMetadataStore(() => ({
    get: () => Promise.resolve(null),
    set: () => Promise.resolve(),
    mget: () => Promise.resolve([]),
  })),
  injectOpenAiToken(requireEnv("OPENAI_API_KEY")),
  injectGeminiToken(requireEnv("GEMINI_API_KEY")),
  injectKimiToken(requireEnv("KIMI_API_KEY")),
  injectAnthropicToken(requireEnv("ANTHROPIC_API_KEY")),
);

export const agentDeps = (inMemoryHistory: HistoryEvent[]): Injector =>
  pipe(
    injectAccessHistory(() => Promise.resolve(inMemoryHistory)),
    injectOutputEvent((event) => {
      inMemoryHistory.push(event);
      return Promise.resolve();
    }),
  );

// Run agent with a specific provider. Caching (via injectSecrets) applies
// regardless of which path is used; the wrapper reads provider from spec.
export const runWithProvider =
  (provider: "google" | "moonshot" | "anthropic" | undefined) =>
  (spec: AgentSpec): Promise<void> => runAgent({ ...spec, provider });

// Run the same test with all providers (Google, Moonshot, Anthropic).
// Set geminiOnly=true for tests that use Gemini-specific features/mock data
export const runForAllProviders = (
  testName: string,
  testFn: (
    runAgentWithProvider: (spec: AgentSpec) => Promise<void>,
  ) => Promise<void>,
  retries = 3,
  geminiOnly = false,
  sanitizeResources = true,
): void => {
  const requestedProvider = Deno.env.get("TEST_PROVIDER");
  if (!requestedProvider) return;

  const normalizedProvider = requestedProvider === "gemini"
    ? "google"
    : requestedProvider;
  if (
    normalizedProvider !== "google" && normalizedProvider !== "moonshot" &&
    normalizedProvider !== "anthropic"
  ) {
    throw new Error(
      `Invalid TEST_PROVIDER: ${requestedProvider}. Expected one of: google, moonshot, anthropic, gemini.`,
    );
  }

  if (normalizedProvider === "moonshot" && geminiOnly) return;
  if (normalizedProvider === "anthropic" && geminiOnly) return;

  const runWithSelectedProvider = normalizedProvider === "google"
    ? runWithProvider(undefined)
    : runWithProvider(normalizedProvider);

  Deno.test(
    {
      name: `${testName} [${normalizedProvider}]`,
      sanitizeResources,
      fn: injectSecrets(withRetries(retries, async () => {
        await testFn(runWithSelectedProvider);
      })),
    },
  );
};

export const noopRewriteHistory = async () => {};

export const toolResult = "43212e8e";

export const someTool = {
  name: "doSomethingUnique",
  description: "Returns a unique string so we know the tool was called.",
  parameters: z.object({}),
  handler: () => Promise.resolve(toolResult),
};

const toBase64 = (u8: Uint8Array): string => {
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < u8.length; i += chunk) {
    binary += String.fromCharCode(...u8.subarray(i, i + chunk));
  }
  return btoa(binary);
};

const bytes = await Deno.readFile("./dog.jpg");

export const b64 = toBase64(bytes);

export const mediaTool = {
  name: "returnMedia",
  description: "Returns media via attachments",
  parameters: z.object({}),
  handler: () => {
    const ret: ToolReturn = {
      result: "image attached",
      attachments: [
        { kind: "inline" as const, mimeType: "image/jpeg", dataBase64: b64 },
      ],
    };
    return Promise.resolve(ret);
  },
};

export const mediaToolWithCaption = {
  name: "returnMediaWithCaption",
  description: "Returns media with caption via attachments",
  parameters: z.object({}),
  handler: () => {
    const ret: ToolReturn = {
      result: "image with caption attached",
      attachments: [
        {
          kind: "inline" as const,
          mimeType: "image/jpeg",
          dataBase64: b64,
          caption: "A friendly golden retriever sitting in the grass",
        },
      ],
    };
    return Promise.resolve(ret);
  },
};

export const recognizedTheDog = (e: HistoryEvent) =>
  e.type === "own_utterance" &&
  (e.text.toLowerCase().includes("dog") ||
    e.text.toLowerCase().includes("retriever") ||
    e.text.toLowerCase().includes("puppy"));

export const findTextualAnswer = (events: HistoryEvent[]) =>
  events.find((event): event is Extract<HistoryEvent, {
    type: "own_utterance";
    text: string;
  }> =>
    event.type === "own_utterance" && typeof event.text === "string" &&
    event.text.length > 0
  );

export const collectAttachment = (events: HistoryEvent[]) =>
  [...events].reverse().find((
    event,
  ): event is Extract<
    HistoryEvent,
    { type: "tool_result" | "own_utterance" }
  > =>
    (event.type === "tool_result" || event.type === "own_utterance") &&
    !!event.attachments?.length
  )?.attachments?.[0];

export const addition = tool({
  name: "add",
  description: "Add two numbers",
  parameters: z.object({ a: z.number(), b: z.number() }),
  handler: ({ a, b }) => Promise.resolve(`${a + b}`),
});

export const multiplication = tool({
  name: "multiply",
  description: "Multiply two numbers",
  parameters: z.object({ x: z.number(), y: z.number() }),
  handler: ({ x, y }) => Promise.resolve(`${x * y}`),
});

export const weatherSkill = {
  name: "weather",
  description: "Get weather information",
  instructions: "Always ask for location before checking weather",
  tools: [
    tool({
      name: "get_forecast",
      description: "Get weather forecast for a location",
      parameters: z.object({ location: z.string() }),
      handler: ({ location }) => Promise.resolve(`Sunny in ${location}`),
    }),
    tool({
      name: "get_temperature",
      description: "Get current temperature",
      parameters: z.object({ location: z.string() }),
      handler: ({ location }) => Promise.resolve(`25°C in ${location}`),
    }),
  ],
};

export const withRetries = (
  maxAttempts: number,
  fn: () => Promise<void>,
) =>
async () => {
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      await fn();
      return;
    } catch (e) {
      if (attempt === maxAttempts) throw e;
      console.log(`Attempt ${attempt}/${maxAttempts} failed, retrying...`);
    }
  }
};

export const llmTest = (
  name: string,
  fn: () => Promise<void>,
  retries = 3,
) => Deno.test(name, withRetries(retries, fn));
