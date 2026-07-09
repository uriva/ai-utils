import {
  accessCallModel,
  accessCallModelWrapper,
  type AgentSpec,
  type CallModel,
  createReadScratchFileTool,
  getSpecForTurn,
  type HistoryEvent,
  injectOutputEvent,
  injectScratchPad,
  injectStreamChunk,
  injectStreamThinkingChunk,
  runAbstractAgent,
  sanitizeHistorySkillsForModel,
} from "./src/agent.ts";
import { anthropicAgentCaller } from "./src/anthropicAgent.ts";
import { runAudioTransportAgent } from "./src/audioTransportAgent.ts";
import { createConsultTool } from "./src/consultTool.ts";
import { geminiAgentCaller, prepareGeminiHistory } from "./src/geminiAgent.ts";
import { validateZodSchema } from "./src/gemini.ts";
import { inspectMediaUrlTool } from "./src/inspectMediaTool.ts";
import { kimiAgentCaller } from "./src/kimiAgent.ts";
export { consultToolName } from "./src/consultTool.ts";
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
  summarizeSegmentToHistoryEvent,
} from "./src/compaction.ts";
export { injectCacher } from "./src/cacher.ts";
export {
  type CompactionOptions,
  getSpillThreshold,
  runToolResultCompaction,
} from "./src/continuousCompaction.ts";
export {
  countTextTokens,
  geminiGenText,
  injectGeminiModelVersions,
  injectGeminiToken,
  validateSchema,
  validateZodSchema,
  zodToGeminiParameters,
} from "./src/gemini.ts";
export { genJson, genJsonFromConvo } from "./src/genJson.ts";
export {
  assertNoScriptDrift,
  driftingScripts,
  type ScriptDriftError,
  scriptsPresent,
} from "./src/scriptDriftGuard.ts";
export {
  injectGeminiErrorLogger,
  injectGeminiModelCallTimeoutMs,
  injectTokenUsage,
  type TokenUsage,
} from "./src/geminiAgent.ts";
export { injectKimiToken } from "./src/kimiAgent.ts";
export { injectAnthropicToken } from "./src/anthropicAgent.ts";
export { injectOpenAiToken, openAiMatching } from "./src/openai.ts";
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

// Provider-specific pre-filter that runs OUTSIDE the cached `callModel`
// boundary. Any history normalization that has an observable side effect
// (e.g. `rewriteHistory`) must live here — otherwise cached test runs replay
// the cache and skip the side effect entirely. Inside the provider caller
// the same filters still run for correctness; the duplication is an
// idempotent no-op on an already-prepared history.
// deno-lint-ignore no-explicit-any
const widenPrepare = (fn: (events: any) => Promise<any>) =>
  fn as (events: HistoryEvent[]) => Promise<HistoryEvent[]>;

const prepareHistory =
  (spec: AgentSpec) => (events: HistoryEvent[]): Promise<HistoryEvent[]> => {
    if (spec.provider === "moonshot" || spec.provider === "anthropic") {
      return Promise.resolve(events);
    }
    return widenPrepare(prepareGeminiHistory(spec.rewriteHistory))(events);
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
  const wrapped = accessCallModelWrapper({
    provider: spec.provider,
    systemPrompt: spec.prompt,
    inner: base,
  });
  const prepare = prepareHistory(spec);
  const runner: CallModel = async (events) =>
    wrapped(await prepare(sanitizeHistorySkillsForModel(events)));

  if (spec.isConsult) {
    return injectStreamChunk((_chunk: string) => {})(
      injectStreamThinkingChunk((_chunk: string) => {})(
        runner,
      ),
    );
  }
  return runner;
};

const builtinTools = [inspectMediaUrlTool];

// The strong model runs as a single CallModel turn and we only keep its text
// reply. With tools it leads with a tool_call and emits no text; even tool-less
// it stays silent when the weaker model asks it to "act". So strip tools/skills
// and prepend a consult-role preamble framing it as an advisor that must answer
// in text — otherwise `consult` returns "[stronger model returned no text]".
const consultRolePreamble = (agentPrompt: string) =>
  `You are the stronger model in your AI family. The weaker model handling the conversation below has paused to consult you for advice.

CRITICAL: You are an ADVISOR, NOT the agent described in the instructions below. The instructions below apply ONLY to the weaker model, NOT to you. You are NOT bound by any of its rules, "do not respond" instructions, or restrictions, and you must NEVER stay silent. Your sole job is to help the weaker model by providing clear, plain-text advice and reasoning. You MUST always write a substantive response in plain text guiding the weaker model on what to do next. You MUST start your response with the word "Advice:" and then provide your detailed guidance.

For your context, here is the system prompt and instructions of the weaker model you are advising (remember: these rules apply to THEM, NOT to you):
=== WEAKER MODEL INSTRUCTIONS ===
${agentPrompt}
=== END WEAKER MODEL INSTRUCTIONS ===`;

const consultBuiltin = (spec: AgentSpec) =>
  spec.lightModel
    ? [createConsultTool(
      resolveCallModel({
        ...spec,
        lightModel: false,
        tools: [],
        skills: [],
        prompt: consultRolePreamble(spec.prompt),
        isConsult: true,
      }),
    )]
    : [];

const addBuiltinTools = (spec: AgentSpec): AgentSpec => {
  const existingToolNames = new Set(spec.tools.map(({ name }) => name));
  const scratchTool = spec.toolOutputScratchPad
    ? [createReadScratchFileTool(spec.toolOutputScratchPad)]
    : [];
  return {
    ...spec,
    tools: [
      ...spec.tools,
      ...builtinTools.filter(({ name }) => !existingToolNames.has(name)),
      ...scratchTool.filter(({ name }) => !existingToolNames.has(name)),
      ...consultBuiltin(spec).filter(({ name }) =>
        !existingToolNames.has(name)
      ),
    ],
  };
};

const runAgentInner = (spec: AgentSpec): Promise<void> => {
  const specWithBuiltins = addBuiltinTools(spec);

  const dynamicCallModel = async (history: HistoryEvent[]) => {
    const specForTurn = getSpecForTurn(specWithBuiltins, history);
    return await resolveCallModel(specForTurn)(history);
  };

  return spec.transport?.kind === "audio"
    ? runAudioTransportAgent(specWithBuiltins)
    : runAbstractAgent(specWithBuiltins, dynamicCallModel);
};

export const runAgent = (spec: AgentSpec): Promise<void> => {
  // Validate all tools and skills before starting the agent run to catch unsupported schema constructs early
  if (spec.tools) {
    for (const tool of spec.tools) {
      try {
        validateZodSchema(tool.parameters, `tool:${tool.name}`);
      } catch (e) {
        throw new Error(
          `Tool validation failed for '${tool.name}': ${
            e instanceof Error ? e.message : String(e)
          }`,
        );
      }
    }
  }
  if (spec.skills) {
    for (const skill of spec.skills) {
      if (skill.tools) {
        for (const tool of skill.tools) {
          try {
            validateZodSchema(
              tool.parameters,
              `skill:${skill.name}/tool:${tool.name}`,
            );
          } catch (e) {
            throw new Error(
              `Skill tool validation failed for '${skill.name}/${tool.name}': ${
                e instanceof Error ? e.message : String(e)
              }`,
            );
          }
        }
      }
    }
  }

  let runner = () => runAgentInner(spec);
  if (spec.toolOutputScratchPad) {
    runner = injectScratchPad(() => spec.toolOutputScratchPad)(runner);
  }
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
