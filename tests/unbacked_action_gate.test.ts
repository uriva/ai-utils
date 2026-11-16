import { assert, assertEquals } from "@std/assert";
import { runAgent, tool, z } from "../mod.ts";
import {
  type HistoryEvent,
  injectCallModel,
  ownThoughtTurnWithMetadata,
  ownUtteranceTurnWithMetadata,
  participantUtteranceTurn,
  toolResultTurn,
  toolUseTurn,
} from "../src/agent.ts";
import { injectDecisionModel } from "../src/decisionModel.ts";
import {
  auditUnbackedAction,
  unbackedActionCorrectionText,
  unbackedActionDecisionKey,
} from "../src/unbackedActionGate.ts";
import { agentDeps, injectSecrets } from "../test_helpers.ts";

const delegateToolName = "delegate_background_task";

const makeDelegateTool = (executedPrompts: string[]) =>
  tool({
    name: delegateToolName,
    description:
      "Delegate a multi-step task to an isolated background worker thread.",
    parameters: z.object({ prompt: z.string() }),
    handler: ({ prompt }) => {
      executedPrompts.push(prompt);
      return Promise.resolve();
    },
  });

Deno.test(
  "runAgent retries with correctional thought when model promises background action with zero tool calls",
  async () => {
    const history: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text:
          "Please generate 5 printable grammar worksheets personalized for Alice and Emma.",
      }),
    ];
    const executedPrompts: string[] = [];
    const callCounter = { count: 0 };
    const seenCorrectionThoughts: string[] = [];

    const scriptedModel = (events: HistoryEvent[]) => {
      callCounter.count += 1;
      const matchingThoughts = events
        .filter((e) =>
          e.type === "own_thought" && e.text === unbackedActionCorrectionText
        )
        .map((e) => (e.type === "own_thought" ? e.text : ""));
      seenCorrectionThoughts.push(...matchingThoughts);

      return Promise.resolve(
        callCounter.count === 1
          ? [
            ownThoughtTurnWithMetadata(
              `I will acknowledge the user briefly and call ${delegateToolName} right now.`,
              { type: "gemini", responseId: "r-1" },
            ),
            ownUtteranceTurnWithMetadata(
              "Sure! I am preparing the 5 personalized worksheets now and will send them to you as soon as they are ready.",
              { type: "gemini", responseId: "r-1" },
            ),
          ]
          : [
            ownThoughtTurnWithMetadata(
              `Emitting acknowledgement together with ${delegateToolName}.`,
              { type: "gemini", responseId: "r-2" },
            ),
            ownUtteranceTurnWithMetadata(
              "Sure! Preparing the 5 worksheets now — I will send them right over when ready.",
              { type: "gemini", responseId: "r-2" },
            ),
            toolUseTurn({
              name: delegateToolName,
              args: {
                prompt:
                  "Create 5 printable grammar worksheets for Alice and Emma.",
              },
            }),
          ],
      );
    };

    const decisionCounter = { count: 0 };
    const mockDecisionModel = (
      _state: unknown,
      questions: Record<string, unknown>,
    ) => {
      if (questions && unbackedActionDecisionKey in questions) {
        decisionCounter.count += 1;
        return Promise.resolve({
          [unbackedActionDecisionKey]: {
            type: "noul" as const,
            noul: 0.95,
          },
        });
      }
      return Promise.resolve({});
    };

    await injectDecisionModel(mockDecisionModel)(
      injectCallModel(scriptedModel)(async () => {
        await agentDeps(history)(runAgent)({
          maxIterations: 5,
          timezoneIANA: "UTC",
          prompt: "You are a helpful assistant.",
          tools: [makeDelegateTool(executedPrompts)],
        });
      }),
    )();

    assertEquals(callCounter.count, 2);
    assertEquals(decisionCounter.count, 1);
    assertEquals(seenCorrectionThoughts.length, 1);
    assertEquals(executedPrompts, [
      "Create 5 printable grammar worksheets for Alice and Emma.",
    ]);
    const utterances = history.filter((e) => e.type === "own_utterance");
    assertEquals(
      utterances.length,
      1,
      "Premature unbacked acknowledgement must not be emitted before retry",
    );
  },
);

Deno.test(
  "runAgent skips unbacked-action decision call on normal replies and post-tool status replies",
  async () => {
    const executedPrompts: string[] = [];
    const decisionCounter = { count: 0 };
    const mockDecisionModel = (
      _state: unknown,
      questions: Record<string, unknown>,
    ) => {
      if (questions && unbackedActionDecisionKey in questions) {
        decisionCounter.count += 1;
      }
      return Promise.resolve({});
    };

    const directAnswerHistory: HistoryEvent[] = [
      participantUtteranceTurn({ name: "user", text: "What is 2 + 2?" }),
    ];
    await injectDecisionModel(mockDecisionModel)(
      injectCallModel(() =>
        Promise.resolve([
          ownThoughtTurnWithMetadata(
            "Simple arithmetic question; answering directly.",
            { type: "gemini", responseId: "r-direct" },
          ),
          ownUtteranceTurnWithMetadata(
            "2 + 2 is 4.",
            { type: "gemini", responseId: "r-direct" },
          ),
        ])
      )(async () => {
        await agentDeps(directAnswerHistory)(runAgent)({
          maxIterations: 5,
          timezoneIANA: "UTC",
          prompt: "You are a helpful assistant.",
          tools: [makeDelegateTool(executedPrompts)],
        });
      }),
    )();
    assertEquals(
      decisionCounter.count,
      0,
      "Normal direct answers that do not reference tools in thought must incur zero decision calls",
    );

    const postToolHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Start provisioning my background line.",
      }),
    ];
    const stepCounter = { count: 0 };
    const provisionTool = tool({
      name: "start_provisioning",
      description: "Starts cloud provisioning asynchronously.",
      parameters: z.object({}),
      handler: () => Promise.resolve("Cloud activation started (~1 minute)."),
    });
    await injectDecisionModel(mockDecisionModel)(
      injectCallModel(() => {
        stepCounter.count += 1;
        return Promise.resolve(
          stepCounter.count === 1
            ? [toolUseTurn({ name: "start_provisioning", args: {} })]
            : [
              ownThoughtTurnWithMetadata(
                "start_provisioning succeeded; informing the user.",
                { type: "gemini", responseId: "r-post" },
              ),
              ownUtteranceTurnWithMetadata(
                "Your line is being activated in the background (~1 minute); I will message you as soon as it is live!",
                { type: "gemini", responseId: "r-post" },
              ),
            ],
        );
      })(async () => {
        await agentDeps(postToolHistory)(runAgent)({
          maxIterations: 5,
          timezoneIANA: "UTC",
          prompt: "You are a helpful assistant.",
          tools: [provisionTool],
        });
      }),
    )();
    assertEquals(
      decisionCounter.count,
      0,
      "Post-tool status replies after an executed workflow tool must incur zero decision calls",
    );
  },
);

Deno.test(
  "auditUnbackedAction - live decision model evaluation",
  injectSecrets(async () => {
    const tools = [makeDelegateTool([])];
    const userHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text:
          "Can you compile a 5-page quarterly expense report for Alex and Sam?",
      }),
    ];

    const phantomDelegation = [
      ownThoughtTurnWithMetadata(
        `I will send a brief acknowledgement to the user before kicking off the delegation. I am now calling ${delegateToolName} with a detailed prompt.`,
        { type: "gemini", responseId: "live-1" },
      ),
      ownUtteranceTurnWithMetadata(
        "Gladly! I am compiling the 5-page quarterly expense report for Alex and Sam right now. It will take a moment — I will send it to you as soon as it is ready.",
        { type: "gemini", responseId: "live-1" },
      ),
    ];
    const flagged = await auditUnbackedAction(
      userHistory,
      phantomDelegation,
      phantomDelegation.filter((e) => e.type === "own_utterance"),
      tools,
      [],
    );
    assert(
      flagged,
      "Expected phantom delegation (promising ongoing background preparation with zero tool calls) to be flagged",
    );

    const capabilityExplanation = [
      ownThoughtTurnWithMetadata(
        `The user is asking me to message a third party, which ${delegateToolName} cannot do on a shared demo channel. I will explain the limitation and ask if they want a dedicated line.`,
        { type: "gemini", responseId: "live-2" },
      ),
      ownUtteranceTurnWithMetadata(
        "On this shared number I cannot message third parties directly. Would you like to set up a dedicated number so I can reach out to them?",
        { type: "gemini", responseId: "live-2" },
      ),
    ];
    const notFlagged = await auditUnbackedAction(
      [
        participantUtteranceTurn({
          name: "user",
          text: "Message John for me and ask when he is free.",
        }),
      ],
      capabilityExplanation,
      capabilityExplanation.filter((e) => e.type === "own_utterance"),
      tools,
      [],
    );
    assertEquals(
      notFlagged,
      false,
      "Capability explanation / question asking user for confirmation must not be flagged",
    );

    const alreadyRunningHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Prepare the report.",
      }),
      toolUseTurn({
        id: "tc-prev",
        name: delegateToolName,
        args: { prompt: "Prepare the report." },
      }),
      toolResultTurn({
        toolCallId: "tc-prev",
        result: "Background worker started.",
      }),
      participantUtteranceTurn({
        name: "user",
        text: "Is it ready yet?",
      }),
    ];
    const statusReply = [
      ownThoughtTurnWithMetadata(
        `The ${delegateToolName} task is still running in the background; I will let the user know it is still in progress.`,
        { type: "gemini", responseId: "live-3" },
      ),
      ownUtteranceTurnWithMetadata(
        "It is still running in the background — I will send the report over as soon as it finishes!",
        { type: "gemini", responseId: "live-3" },
      ),
    ];
    const statusFlagged = await auditUnbackedAction(
      alreadyRunningHistory,
      statusReply,
      statusReply.filter((e) => e.type === "own_utterance"),
      tools,
      [],
    );
    assertEquals(
      statusFlagged,
      false,
      "Status update on an already-running background task from prior history must not be flagged",
    );
  }),
);
