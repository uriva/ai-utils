import { assertEquals } from "@std/assert";
import type { Injector } from "@uri/inject";
import {
  type AgentSpec,
  type HistoryEvent,
  injectCacher,
  participantUtteranceTurn,
} from "../mod.ts";
import {
  agentDeps,
  b64,
  injectSecrets,
  noopRewriteHistory,
  runWithProvider,
} from "../test_helpers.ts";

const inlineUploadCacheId = "gemini-inline-upload-v1";

const countingCacher = (
  counters: Map<string, number>,
  stores: Map<string, Map<string, unknown>>,
) =>
(cacheId: string): Injector => {
  if (!stores.has(cacheId)) stores.set(cacheId, new Map());
  const store = stores.get(cacheId)!;
  return ((f: (...args: unknown[]) => Promise<unknown>) =>
  async (...args: unknown[]) => {
    const key = JSON.stringify(args);
    if (store.has(key)) return store.get(key);
    counters.set(cacheId, (counters.get(cacheId) ?? 0) + 1);
    const result = await f(...args);
    store.set(key, result);
    return result;
  }) as Injector;
};

const inlineImageEvent = participantUtteranceTurn({
  name: "user",
  text: "Here is a picture for context.",
  attachments: [
    { kind: "inline", mimeType: "image/jpeg", dataBase64: b64 },
  ],
});

const askName = (text: string) =>
  participantUtteranceTurn({ name: "user", text });

const baseSpec = (): Omit<AgentSpec, "provider"> => ({
  maxIterations: 2,
  onMaxIterationsReached: () => {},
  tools: [],
  prompt: "You can see images attached by the user. Reply briefly.",
  lightModel: true,
  rewriteHistory: noopRewriteHistory,
  timezoneIANA: "UTC",
});

Deno.test({
  name: "inline attachment is uploaded only once across turns [gemini]",
  fn: injectSecrets(async () => {
    const counters = new Map<string, number>();
    const stores = new Map<string, Map<string, unknown>>();
    await injectCacher(countingCacher(counters, stores))(async () => {
      const history: HistoryEvent[] = [
        inlineImageEvent,
        askName("What animal is in the picture?"),
      ];
      await agentDeps(history)(runWithProvider(undefined))(baseSpec());
      history.push(askName("And what color is it?"));
      await agentDeps(history)(runWithProvider(undefined))(baseSpec());
    })();
    assertEquals(
      counters.get(inlineUploadCacheId),
      1,
      `inline upload inner should run exactly once across both turns; counters=${
        JSON.stringify([...counters.entries()])
      }`,
    );
  }),
  sanitizeResources: false,
});
