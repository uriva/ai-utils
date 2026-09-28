import {
  accessCallModel,
  accessCallModelWrapper,
  type AgentSpec,
  type CallModel,
  cleanActiveMemoryTool,
  createReadScratchFileTool,
  getSpecForTurn,
  type HistoryEvent,
  injectOutputEvent,
  injectScratchPad,
  injectStreamChunk,
  injectStreamThinkingChunk,
  runAbstractAgent,
  sanitizeHistorySkillsForModel,
  searchPastHistoryTool,
} from "./src/agent.ts";
import { anthropicAgentCaller } from "./src/anthropicAgent.ts";
import { runAudioTransportAgent } from "./src/audioTransportAgent.ts";
import { geminiAgentCaller, prepareGeminiHistory } from "./src/geminiAgent.ts";
import { validateZodSchema } from "./src/gemini.ts";
import { inspectMediaUrlTool } from "./src/inspectMediaTool.ts";
import { formatAgentStateForRespan, routeTask } from "./src/respan.ts";
import { kimiAgentCaller } from "./src/kimiAgent.ts";
import { ThinkingLevel } from "@google/genai";

type AgentSpecForTurn = AgentSpec & {
  thinkingLevel?: ThinkingLevel;
};

// deno-lint-ignore no-explicit-any
const widen = (caller: (events: any) => Promise<any>): CallModel => caller;

const providerCaller = (spec: AgentSpecForTurn): CallModel => {
  if (spec.provider === "moonshot") return widen(kimiAgentCaller(spec));
  if (spec.provider === "anthropic") return widen(anthropicAgentCaller(spec));
  // Default to Gemini for audio transport or when provider is "google" or undefined
  return widen(geminiAgentCaller(spec));
};

// Provider-specific pre-filter that runs OUTSIDE the cached `callModel`
// boundary. Inside the provider caller the same filters still run for correctness;
// the duplication is an idempotent no-op on an already-prepared history.
// deno-lint-ignore no-explicit-any
const widenPrepare = (fn: (events: any) => Promise<any>) =>
  fn as (events: HistoryEvent[]) => Promise<HistoryEvent[]>;

const prepareHistory =
  (spec: AgentSpec) => (events: HistoryEvent[]): Promise<HistoryEvent[]> => {
    if (spec.provider === "moonshot" || spec.provider === "anthropic") {
      return Promise.resolve(events);
    }
    return widenPrepare(prepareGeminiHistory)(events);
  };

// Picks the CallModel to use for this agent run.
// - injectCallModel(fake) wins outright (tests use this to bypass providers).
// - otherwise the provider-based caller is chosen from spec.provider.
// Then injectCallModelWrapper wraps whatever was chosen (tests use this to
// add rmmbr caching around a real provider caller).
const resolveCallModel = (spec: AgentSpecForTurn): CallModel => {
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
  return async (events) =>
    wrapped(await prepare(sanitizeHistorySkillsForModel(events)));
};

const builtinTools = [inspectMediaUrlTool];

const addBuiltinTools = (spec: AgentSpec): AgentSpec => {
  const existingToolNames = new Set(spec.tools.map(({ name }) => name));
  const scratchTool = spec.toolOutputScratchPad
    ? [createReadScratchFileTool(spec.toolOutputScratchPad)]
    : [];
  const cleanTool = spec.enableCleanActiveMemory !== false
    ? [cleanActiveMemoryTool()]
    : [];
  const searchTool = spec.enableHistorySearch !== false
    ? [searchPastHistoryTool()]
    : [];
  return {
    ...spec,
    tools: [
      ...spec.tools,
      ...builtinTools.filter(({ name }) => !existingToolNames.has(name)),
      ...scratchTool.filter(({ name }) => !existingToolNames.has(name)),
      ...cleanTool.filter(({ name }) => !existingToolNames.has(name)),
      ...searchTool.filter(({ name }) => !existingToolNames.has(name)),
    ],
  };
};

const runAgentInner = (spec: AgentSpec): Promise<void> => {
  const specWithBuiltins = addBuiltinTools(spec);

  const dynamicCallModel = async (history: HistoryEvent[]) => {
    const specForTurn = getSpecForTurn(specWithBuiltins, history);
    const routedTier = await routeTask(
      formatAgentStateForRespan(specForTurn.prompt, history, specForTurn.tools),
    );
    const thinkingLevel = routedTier === "lite"
      ? ThinkingLevel.LOW
      : ThinkingLevel.HIGH;
    return await resolveCallModel({
      ...specForTurn,
      thinkingLevel,
    })(history);
  };

  return spec.transport?.kind === "audio"
    ? runAudioTransportAgent(specWithBuiltins)
    : runAbstractAgent(specWithBuiltins, dynamicCallModel);
};

export {
  accessCallModel,
  accessCallModelWrapper,
  accessHistory,
  type AgentInputs,
  type AgentSpec,
  type CallModel,
  type CallModelWrapper,
  callToResult,
  compileGrepPattern,
  createReadScratchFileTool,
  createSkillTools,
  disambiguateDuplicateToolCalls,
  doNothingEvent,
  doNothingTool,
  doNothingToolName,
  estimateAgentInputTokens,
  estimateTokens,
  estimateTokensLocal,
  externalEventPrefix,
  externalEventTurn,
  forcedStopUtterance,
  formatSkillsPrompt,
  formatSystemNotification,
  generateId,
  getSpecForTurn,
  getStreamChunk,
  getStreamThinkingChunk,
  handleFunctionCalls,
  type HistoryEvent,
  historyHasPendingDeferredUserWaitingNudge,
  injectAccessHistory,
  injectCallModel,
  injectCallModelWrapper,
  injectMetadataStore,
  injectOutputEvent,
  injectScratchPad,
  injectShouldAbort,
  injectStreamChunk,
  injectStreamThinkingChunk,
  injectTimerMs,
  injectToolNotFound,
  isRecord,
  learnSkillToolName,
  maxToolOutputChars,
  maxUtteranceChars,
  type MediaAttachment,
  noResponseTag,
  normalizeHistoryForModel,
  overrideIdGenerator,
  overrideTime,
  ownEditMessageTurn,
  ownThoughtTurn,
  ownThoughtTurnWithMetadata,
  ownUtteranceTurn,
  ownUtteranceTurnWithMetadata,
  participantEditMessageTurn,
  participantUtteranceTurn,
  projectHistoryToModelContext,
  type Provider,
  qualifiedToolName,
  readScratchFileToolName,
  referenceToolName,
  type RegularTool,
  resolveToolDescription,
  runAbstractAgent,
  runCommandToolName,
  sanitizeHistorySkillsForModel,
  sanitizeModelOutput,
  sanitizeWindowBoundary,
  scheduleHistoryCompaction,
  searchPastHistoryTool,
  type Skill,
  skillAutoLoadMarker,
  skillLearnedSuccessMessage,
  skillLoadedResultText,
  skillUnlearnedSuccessMessage,
  stopThoughtPrefix,
  systemNotificationPrefix,
  thinkingTokenExhaustionWarningText,
  type Tool,
  tool,
  type ToolOutputScratchPad,
  toolResultTurn,
  type ToolReturn,
  toolUseTurn,
  toolUseTurnWithMetadata,
  truncateToolOutput,
  unlearnSkillToolName,
} from "./src/agent.ts";
export { injectAnthropicToken } from "./src/anthropicAgent.ts";
export { injectCacher } from "./src/cacher.ts";
export {
  applyCleanActiveMemoryDirectives,
  compactionRetentionTokens,
  eventsToPlainText,
  eventToPlainText,
  groupToolCallPairs,
  type HistorySegment,
  partitionSegments,
  segmentHistoryEvents,
  summarizeEvents,
  summarizeSegmentToHistoryEvent,
} from "./src/compaction.ts";
export {
  compactToolResultsInMemory,
  defaultDeterministicTLDR,
  getSpillThreshold,
} from "./src/continuousCompaction.ts";
export { createDuplexPair, type DuplexMessage } from "./src/duplex.ts";
export {
  geminiFallbackVersion,
  geminiFlashVersion,
  geminiGenText,
  geminiLiteVersion,
  geminiModelVersion,
  geminiProVersion,
  geminiThinkingConfig,
  injectGeminiModelVersions,
  injectGeminiToken,
  validateSchema,
  validateZodSchema,
  zodToGeminiParameters,
} from "./src/gemini.ts";
export { ThinkingLevel } from "@google/genai";
export {
  clearGeminiContextCacheMap,
  geminiContextCacheBufferSeconds,
  geminiContextCacheClientTtlSeconds,
  geminiContextCacheId,
  geminiContextCacheTtlSeconds,
  getOrCreateGeminiContextCache,
  injectGeminiContextCachingEnabled,
  invalidateGeminiContextCache,
} from "./src/geminiContextCache.ts";
export {
  injectGeminiErrorLogger,
  injectGeminiModelCallTimeoutMs,
  injectPromptBlockedLogger,
  injectTokenUsage,
  safetyWarningText,
  type TokenUsage,
} from "./src/geminiAgent.ts";
export {
  type AudioSession,
  type AudioSessionEvent,
  type ClientContentParams,
  type ClientContentPart,
  type ClientContentTurn,
  createAudioSession,
  defaultLiveModel,
  geminiLiveVersion,
  type LiveFunctionDeclaration,
  toolsToDeclarations,
} from "./src/geminiLiveSession.ts";
export {
  accessDecisionProvider,
  callDecisionModel,
  type ChoiceDecisionAnswer,
  type ChoiceDecisionQuestion,
  decide,
  type DecisionAnswer,
  type DecisionModelCaller,
  type DecisionProvider,
  type DecisionQuestion,
  genDecision,
  injectDecisionModel,
  injectDecisionProvider,
  isDecisionField,
  isDecisionModelAvailable,
  isStringField,
  type NoulDecisionAnswer,
  type NoulDecisionQuestion,
  resolveDecisionProvider,
  type ScoreDecisionAnswer,
  type ScoreDecisionQuestion,
} from "./src/decisionModel.ts";
export {
  geminiGenJson,
  genJson,
  genJsonFromConvo,
  genJsonOverride,
  invalidGenJsonMessage,
} from "./src/genJson.ts";
export { injectKimiToken, kimiGenJsonFromConvo } from "./src/kimiJson.ts";
export {
  accessRespanToken,
  callRespanDecisionModel,
  decideCleanupWithRespan,
  decideSkillsWithRespan,
  defaultRespanModel,
  formatAgentStateForRespan,
  injectRespanToken,
  proRespanModel,
  respanApiUrl,
  routeTask,
  routeTaskWithRespan,
} from "./src/respan.ts";
export {
  accessJevToken,
  callJevDecisionModel,
  decideCleanupWithJev,
  decideSkillsWithJev,
  extractCandidateToolEpisodes,
  formatAgentStateForJev,
  injectJevToken,
  jevApiUrl,
  routeTaskWithJev,
} from "./src/jev.ts";
export { injectOpenAiToken } from "./src/openai.ts";
export {
  auditUtteranceForHallucination,
  hallucinationCorrectionText,
  HallucinationDecisionSchema,
} from "./src/hallucinationGate.ts";
export { searchPastHistoryToolName } from "./src/historySearch.ts";
export {
  catchAiRefusesToAdhereToTyping,
  cleanActiveMemoryToolName,
  type ModelOpts,
  type ModelTier,
} from "./src/utils.ts";
export { z } from "zod/v4";

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
