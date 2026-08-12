import { assert } from "@std/assert";
import { z } from "zod/v4";
import {
  checkHallucination,
  type HistoryEvent,
  ownThoughtTurn,
  ownUtteranceTurn,
  participantUtteranceTurn,
  tool,
  toolResultTurn,
  toolUseTurn,
} from "../mod.ts";
import {
  agentDeps,
  injectSecrets,
  noopRewriteHistory,
  runForAllProviders,
} from "../test_helpers.ts";

export const scheduleTaskToolName = "schedule_task";

const thirdPartyName = "דנה";
const thirdPartyWhatsApp = "972540000001";

const scheduleTaskTool = tool({
  name: scheduleTaskToolName,
  description:
    "Create a task for the bot to handle at a scheduled time. The task runs on the specified network (WhatsApp, Telegram, Email, etc.) and will be queued and executed based on the schedule. For WhatsApp messages, set preferredNetwork to 'whatsapp' and pass the phone number (digits only) as targetConversationId. Leave runAt empty for ASAP.",
  parameters: z.object({
    description: z.string().describe(
      "Description of what needs to be done, including context of why and who this is for.",
    ),
    preferredNetwork: z.string().describe(
      "Which messaging network to use (whatsapp, telegram, email...).",
    ),
    targetConversationId: z.string().describe(
      "The conversation to run the task in. For whatsapp, the phone number digits.",
    ),
    runAt: z.string().describe(
      "When to run the task, ISO 8601. Leave empty for ASAP.",
    ),
  }),
  handler: () => Promise.resolve(JSON.stringify({ success: true })),
});

const queryTasksTool = tool({
  name: "query_tasks",
  description:
    "Query existing scheduled tasks for this bot. Call without parameters to list all tasks. Returns task IDs and their schedules — needed before cancelling.",
  parameters: z.object({ taskId: z.string().optional() }),
  handler: () => Promise.resolve("[]"),
});

const listActiveConversationsTool = tool({
  name: "list_active_conversations",
  description:
    "List the bot's active conversations across networks, with contact names and conversation identifiers.",
  parameters: z.object({}),
  handler: () =>
    Promise.resolve(
      JSON.stringify([
        { name: "owner", network: "whatsapp", conversationId: "972500000000" },
        {
          name: thirdPartyName,
          network: "whatsapp",
          conversationId: thirdPartyWhatsApp,
        },
      ]),
    ),
});

const assistantPrompt =
  `אתה עוזר אישי. אתה מדבר כעת עם הבעלים שלך בצ'אט וואטסאפ.
מטלות שמתוזמנות אליך מגיעות כהערה פנימית שמתחילה ב-"PROACTIVE TASK" — בצע אותן.
ענה בקצרה, בשפה של המשתמש.`;

const questionToOwner =
  `היי, ${thirdPartyName} שאלה אם תרצה לתאם איתה שיחת עדכון באיזשהו שלב, ואם כן, מתי מתאים לך?`;

const ownerAnswer = "כן. סופש. אני חושב שאני אחזור.";

const relayScenarioHistory = (): HistoryEvent[] => [
  ownThoughtTurn(
    `PROACTIVE TASK: You have a task to complete: Message the owner: "${questionToOwner}"`,
  ),
  ownUtteranceTurn(questionToOwner),
  participantUtteranceTurn({ name: "owner", text: ownerAnswer }),
];

const checkerSpec = {
  prompt: assistantPrompt,
  tools: [scheduleTaskTool, queryTasksTool, listActiveConversationsTool],
  skills: [],
  timezoneIANA: "Asia/Jerusalem",
};

const judgeDraft = (history: HistoryEvent[], draft: string) =>
  checkHallucination(
    [...history, ownUtteranceTurn(draft)],
    checkerSpec,
    "Asia/Jerusalem",
  );

Deno.test({
  name:
    "checker flags a concluding reply that commits to updating a third party without performing or scheduling the update",
  fn: injectSecrets(async () => {
    const verdict = await judgeDraft(
      relayScenarioHistory(),
      "הבנתי, אעדכן אותה שמתאים לתזמן לסוף השבוע.",
    );
    assert(
      verdict.isHallucinating,
      `checker let through a commitment to update ${thirdPartyName} with no tool call performing or scheduling it: ${verdict.explanation}`,
    );
  }),
});

Deno.test({
  name:
    "checker flags a concluding reply that claims an answer was recorded when nothing was recorded",
  fn: injectSecrets(async () => {
    const verdict = await judgeDraft(
      relayScenarioHistory(),
      "התשובה שלך נרשמה במערכת לצד הפנייה. אם תרצה שאשלח לדנה הודעה ישירה, רק תעדכן אותי מה פרטי הקשר שלה.",
    );
    assert(
      verdict.isHallucinating,
      `checker let through a false claim that the answer was recorded in the system: ${verdict.explanation}`,
    );
  }),
});

Deno.test({
  name:
    "checker does not flag a commitment when the relay was actually scheduled in the turn",
  fn: injectSecrets(async () => {
    const relayCall = toolUseTurn({
      name: scheduleTaskToolName,
      args: {
        description:
          `Message ${thirdPartyName}: the owner said the weekend works for the check-in`,
        preferredNetwork: "whatsapp",
        targetConversationId: thirdPartyWhatsApp,
        runAt: "",
      },
    });
    const verdict = await judgeDraft(
      [
        ...relayScenarioHistory(),
        relayCall,
        toolResultTurn({
          result: JSON.stringify({ success: true }),
          toolCallId: relayCall.id,
        }),
      ],
      "מעולה, עדכנתי את דנה שמתאים לך בסוף השבוע.",
    );
    assert(
      !verdict.isHallucinating,
      `false positive: the relay was scheduled yet the reply was flagged: ${verdict.explanation}`,
    );
  }),
});

Deno.test({
  name: "checker does not flag a plain acknowledgment without any commitment",
  fn: injectSecrets(async () => {
    const verdict = await judgeDraft(relayScenarioHistory(), "רשמתי, תודה!");
    assert(
      !verdict.isHallucinating,
      `false positive on a plain acknowledgment: ${verdict.explanation}`,
    );
  }),
});

const singleAgentAttempt = async (
  runAgentWithProvider: Parameters<Parameters<typeof runForAllProviders>[1]>[0],
) => {
  const history = relayScenarioHistory();
  const initialLength = history.length;
  await agentDeps(history)(runAgentWithProvider)({
    lightModel: true,
    maxOutputTokens: 8000,
    maxIterations: 10,
    disableStreaming: true,
    timezoneIANA: "Asia/Jerusalem",
    prompt: assistantPrompt,
    tools: [scheduleTaskTool, queryTasksTool, listActiveConversationsTool],
    rewriteHistory: noopRewriteHistory,
  });
  const newEvents = history.slice(initialLength);
  const relayScheduled = newEvents.some((e) =>
    e.type === "tool_call" && e.name === scheduleTaskToolName
  );
  const utterances = newEvents
    .filter((e) => e.type === "own_utterance")
    .map((e) => ("text" in e ? e.text : ""))
    .join("\n");
  if (!relayScheduled && utterances) {
    const verdict = await judgeDraft(history, utterances);
    assert(
      !verdict.isHallucinating,
      `agent committed to relay the answer to ${thirdPartyName} but concluded its turn without scheduling the relay:\n${utterances}\nExplanation: ${verdict.explanation}`,
    );
  }
};

runForAllProviders(
  "agent does not commit to updating a third party without actually doing it",
  async (runAgentWithProvider) => {
    for (let attempt = 1; attempt <= 3; attempt++) {
      await singleAgentAttempt(runAgentWithProvider);
    }
  },
);
