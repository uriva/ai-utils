import {
  type AgentSpec,
  injectOutputEvent,
  injectStreamChunk,
  runAbstractAgent,
} from "./src/agent.ts";
import { runAudioTransportAgent } from "./src/audioTransportAgent.ts";
import { geminiAgentCaller } from "./src/geminiAgent.ts";
export {
  extractOpaqueIdentifiers,
  findNovelOpaqueIdentifiers,
} from "./src/opaqueIdentifiers.ts";
export {
  appendInternalSentTimestamp,
  formatInternalSentTimestamp,
  hasInternalSentTimestampSuffix,
  stripInternalSentTimestampSuffix,
} from "./src/internalMessageMetadata.ts";
export { z } from "zod/v4";
export * from "./src/agent.ts";
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
const runAgentInner = (spec: AgentSpec): Promise<void> =>
  spec.transport?.kind === "audio"
    ? runAudioTransportAgent(spec)
    // @ts-expect-error the caller has gemini metadata
    : runAbstractAgent(spec, geminiAgentCaller(spec));

export const runAgent = (spec: AgentSpec): Promise<void> => {
  let runner = () => runAgentInner(spec);
  if (spec.onOutputEvent) {
    runner = injectOutputEvent(spec.onOutputEvent)(runner);
  }
  if (spec.onStreamChunk) {
    runner = injectStreamChunk(spec.onStreamChunk)(runner);
  }
  return runner();
};
