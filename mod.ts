import {
  type AgentSpec,
  injectOutputEvent,
  runAbstractAgent,
} from "./src/agent.ts";
import { runAudioTransportAgent } from "./src/audioTransportAgent.ts";
import { geminiAgentCaller } from "./src/geminiAgent.ts";
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
export { injectGeminiErrorLogger } from "./src/geminiAgent.ts";
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

export { runAudioTransportAgent } from "./src/audioTransportAgent.ts";
export const runAgent = (spec: AgentSpec): Promise<void> =>
  spec.onOutputEvent
    ? injectOutputEvent(spec.onOutputEvent)(() => runAgentInner(spec))()
    : runAgentInner(spec);
