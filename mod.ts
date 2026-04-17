import {
  accessCallModel,
  accessCallModelWrapper,
  type AgentSpec,
  type CallModel,
  injectOutputEvent,
  injectStreamChunk,
  injectStreamThinkingChunk,
  runAbstractAgent,
} from "./src/agent.ts";
import { anthropicAgentCaller } from "./src/anthropicAgent.ts";
import { runAudioTransportAgent } from "./src/audioTransportAgent.ts";
import { geminiAgentCaller } from "./src/geminiAgent.ts";
import { kimiAgentCaller } from "./src/kimiAgent.ts";
export {
  appendInternalSentTimestamp,
  formatInternalSentTimestamp,
  hasInternalSentTimestampSuffix,
  stripInternalSentTimestampSuffix,
} from "./src/internalMessageMetadata.ts";
export { z } from "zod/v4";
export * from "./src/agent.ts";
export {
  eventsToPlainText,
  eventToPlainText,
  groupToolCallPairs,
  type HistorySegment,
  partitionSegments,
  segmentHistoryEvents,
  summarizeEvents,
} from "./src/compaction.ts";
export { injectCacher } from "./src/cacher.ts";
export {
  ensureGeminiAttachmentIsActive,
  ensureGeminiAttachmentIsLink,
  geminiGenJson,
  geminiGenJsonFromConvo,
  geminiGenText,
  injectGeminiToken,
} from "./src/gemini.ts";
export {
  injectGeminiErrorLogger,
  injectTokenUsage,
  type TokenUsage,
} from "./src/geminiAgent.ts";
export { injectKimiToken } from "./src/kimiAgent.ts";
export { injectAnthropicToken } from "./src/anthropicAgent.ts";
export {
  injectOpenAiToken,
  openAiGenJson,
  openAiGenJsonFromConvo,
  openAiMatching,
} from "./src/openai.ts";
export { catchAiRefusesToAdhereToTyping, type ModelOpts } from "./src/utils.ts";
export {
  type AudioSessionConfig,
  type AudioSessionEvent,
  createAudioSession,
  type LiveAudioChunk,
} from "./src/geminiLiveSession.ts";
export {
  createDuplexPair,
  type DuplexEndpoint,
  type DuplexMessage,
} from "./src/duplex.ts";

// deno-lint-ignore no-explicit-any
const widen = (caller: (events: any) => Promise<any>): CallModel => caller;

const providerCaller = (spec: AgentSpec): CallModel => {
  if (spec.provider === "moonshot") return widen(kimiAgentCaller(spec));
  if (spec.provider === "anthropic") return widen(anthropicAgentCaller(spec));
  // Default to Gemini for audio transport or when provider is "google" or undefined
  return widen(geminiAgentCaller(spec));
};

// Picks the CallModel to use for this agent run.
// - injectCallModel(fake) wins outright (tests use this to bypass providers).
// - otherwise the provider-based caller is chosen from spec.provider.
// Then injectCallModelWrapper wraps whatever was chosen (tests use this to
// add rmmbr caching around a real provider caller).
const resolveCallModel = (spec: AgentSpec): CallModel => {
  const base: CallModel = (events) => {
    try {
      return accessCallModel(events);
    } catch {
      return providerCaller(spec)(events);
    }
  };
  return accessCallModelWrapper({ provider: spec.provider, inner: base });
};

const runAgentInner = (spec: AgentSpec): Promise<void> =>
  spec.transport?.kind === "audio"
    ? runAudioTransportAgent(spec)
    : runAbstractAgent(spec, resolveCallModel(spec));

export const runAgent = (spec: AgentSpec): Promise<void> => {
  let runner = () => runAgentInner(spec);
  if (spec.onOutputEvent) {
    runner = injectOutputEvent(spec.onOutputEvent)(runner);
  }
  if (spec.onStreamChunk) {
    runner = injectStreamChunk(spec.onStreamChunk)(runner);
  }
  if (spec.onStreamThinkingChunk) {
    runner = injectStreamThinkingChunk(spec.onStreamThinkingChunk)(runner);
  }
  return runner();
};
