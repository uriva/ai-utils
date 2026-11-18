import { empty, nonempty } from "gamla";
import {
  type HistoryEvent,
  isSafetyBlockUtterance,
  type Skill,
  type Tool,
} from "./agent.ts";
import {
  callDecisionModel,
  cleanHistoryEventForDecision,
  isDecisionModelAvailable,
  isDecisionModelInjected,
  type NoulDecisionQuestion,
} from "./decisionModel.ts";

export const maxUnbackedActionRetries = 2;
export const unbackedActionDecisionKey = "is_unbacked_action";
export const unbackedActionThreshold = 0.7;

export const unbackedActionCorrectionText =
  "[SYSTEM NOTICE]: Your previous response told the user you are starting or working on their request (or stated in your internal thought that you were calling a tool), but you emitted ZERO tool calls — which would immediately end your turn with no tool executed and no background task running. Emit the required tool call in the SAME response alongside any brief acknowledgement.";

export const unbackedActionQuestion: NoulDecisionQuestion = {
  type: "noul",
  instructions:
    "The assistant emitted ZERO tool calls in this step (`tool_calls_emitted_in_this_step` is empty), so if accepted, the agent's turn immediately ends with NO new tool executed and NO new background task started. Decide whether the assistant's internal thought or proposed reply commits an unbacked-action error (intending or promising to invoke a tool or start background work right now without emitting a tool call).",
  criteria: {
    true:
      "The assistant's internal thought states it is calling or about to call a tool in this step (e.g. 'I am now calling ...', 'Let me run ...') OR the proposed reply tells the user it is currently starting, preparing, researching, or delegating the task right now and will follow up when ready, WITHOUT asking the user for required input first and WITHOUT an already-running background task from prior history explaining the wait.",
    false:
      "The assistant is directly answering the user's question, greeting or chatting, declining or explaining capabilities/limitations, reporting results of already-completed actions, waiting on a background task that was already started in an earlier turn, OR asking the user for information, confirmation, or input before acting.",
  },
};

const availableToolAndSkillNames = (
  // deno-lint-ignore no-explicit-any
  tools: Tool<any>[],
  skills: Skill[],
): string[] => [
  ...new Set(
    [
      ...tools.map((t) => t.name),
      ...skills.flatMap((s) => [s.name, ...(s.tools ?? []).map((t) => t.name)]),
    ].filter((name) => name.length >= 3),
  ),
];

const extractThoughtText = (events: HistoryEvent[]): string =>
  events
    .filter((e): e is Extract<HistoryEvent, { type: "own_thought" }> =>
      e.type === "own_thought" && typeof e.text === "string"
    )
    .map((e) => e.text)
    .join("\n");

const extractUtteranceText = (events: HistoryEvent[]): string =>
  events
    .filter((e): e is Extract<HistoryEvent, { type: "own_utterance" }> =>
      e.type === "own_utterance" && typeof e.text === "string"
    )
    .map((e) => e.text)
    .join("\n");

export const referencedToolsInThoughts = (
  modelResponse: HistoryEvent[],
  // deno-lint-ignore no-explicit-any
  tools: Tool<any>[],
  skills: Skill[],
): string[] => {
  const lowerThought = extractThoughtText(modelResponse).toLowerCase();
  if (!lowerThought) return [];
  return availableToolAndSkillNames(tools, skills).filter((name) =>
    lowerThought.includes(name.toLowerCase())
  );
};

const hasAnyModelMetadata = (events: HistoryEvent[]): boolean =>
  events.some((e) => "modelMetadata" in e && Boolean(e.modelMetadata));

export const shouldAuditUnbackedAction = ({
  modelResponse,
  emitEvents,
  tools,
  skills,
  retryCount,
  hadExecutedWorkflowTool,
  executedToolNames,
  mockModelInjected,
}: {
  modelResponse: HistoryEvent[];
  emitEvents: HistoryEvent[];
  // deno-lint-ignore no-explicit-any
  tools: Tool<any>[];
  skills: Skill[];
  retryCount: number;
  hadExecutedWorkflowTool?: boolean;
  executedToolNames?: ReadonlySet<string>;
  mockModelInjected: boolean;
}): boolean =>
  retryCount < maxUnbackedActionRetries &&
  (executedToolNames ? true : !hadExecutedWorkflowTool) &&
  (!empty(tools) || !empty(skills)) &&
  !emitEvents.some((e) => e.type === "tool_call") &&
  emitEvents.some((e) =>
    e.type === "own_utterance" &&
    typeof e.text === "string" &&
    Boolean(e.text.trim())
  ) &&
  !modelResponse.some(isSafetyBlockUtterance) &&
  (hasAnyModelMetadata(modelResponse) || isDecisionModelInjected()) &&
  (!mockModelInjected || isDecisionModelInjected()) &&
  isDecisionModelAvailable() &&
  nonempty(
    referencedToolsInThoughts(modelResponse, tools, skills).filter((name) =>
      !executedToolNames?.has(name.toLowerCase())
    ),
  );

export const auditUnbackedAction = async (
  history: HistoryEvent[],
  modelResponse: HistoryEvent[],
  emitEvents: HistoryEvent[],
  // deno-lint-ignore no-explicit-any
  tools: Tool<any>[],
  skills: Skill[],
): Promise<boolean> => {
  if (!isDecisionModelAvailable()) return false;
  try {
    const state = {
      recent_history: history
        .slice(-6)
        .map(cleanHistoryEventForDecision)
        .filter(Boolean),
      assistant_internal_thought: extractThoughtText(modelResponse),
      assistant_proposed_reply: extractUtteranceText(emitEvents),
      tool_calls_emitted_in_this_step: [],
      referenced_tools_or_skills: referencedToolsInThoughts(
        modelResponse,
        tools,
        skills,
      ),
    };
    const answers = await callDecisionModel(state, {
      [unbackedActionDecisionKey]: unbackedActionQuestion,
    });
    const answer = answers[unbackedActionDecisionKey];
    return Boolean(
      answer && answer.type === "noul" &&
        answer.noul >= unbackedActionThreshold,
    );
  } catch (err) {
    console.warn(
      "[unbacked-action-gate] decision check failed, failing open:",
      err,
    );
    return false;
  }
};
