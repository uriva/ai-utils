import { type AgentSpec, runAbstractAgent } from "./src/agent.ts";
import { geminiAgentCaller } from "./src/geminiAgent.ts";
export { z } from "zod/v4";
export * from "./src/agent.ts";
export { injectCacher } from "./src/cacher.ts";
export {
  ensureGeminiAttachmentIsLink,
  geminiGenJson,
  geminiGenJsonFromConvo,
  injectGeminiToken
} from "./src/gemini.ts";
export { injectGeminiErrorLogger } from "./src/geminiAgent.ts";
export {
  injectOpenAiToken,
  openAiGenJson,
  openAiGenJsonFromConvo,
  openAiMatching
} from "./src/openai.ts";
export { catchAiRefusesToAdhereToTyping, type ModelOpts } from "./src/utils.ts";
export const runAgent = (spec: AgentSpec): Promise<void> =>
  // @ts-expect-error the caller has gemini metadata
  runAbstractAgent(spec, geminiAgentCaller(spec));
