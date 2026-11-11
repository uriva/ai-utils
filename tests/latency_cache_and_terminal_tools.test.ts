import { assert, assertEquals } from "@std/assert";
import type { GenerateContentParameters } from "@google/genai";
import { z } from "zod/v4";
import {
  doNothingEvent,
  formatSystemNotification,
  getSpecForTurn,
  type HistoryEvent,
  injectGeminiToken,
  learnSkillToolName,
  ownUtteranceTurn,
  participantUtteranceTurn,
  runAgent,
  tool,
  toolUseTurn,
} from "../mod.ts";
import { injectCallModel } from "../src/agent.ts";
import { injectGeminiSdkExchange } from "../src/geminiAgent.ts";
import { agentDeps } from "../test_helpers.ts";

Deno.test(
  "runAgent keeps systemInstruction identical across conversations when conversationContext differs",
  async () => {
    const sampleTool = tool({
      name: "lookup_status",
      description: "Look up an order status.",
      parameters: z.object({ orderId: z.string() }),
      handler: () => Promise.resolve("shipped"),
    });
    const capturedReqs: GenerateContentParameters[] = [];
    const fakeExchange = (
      _signal: AbortSignal,
      { req }: { req: GenerateContentParameters; disableStreaming?: boolean },
    ) => {
      capturedReqs.push(req);
      return Promise.resolve({
        parts: [{ text: "Your order shipped." }],
        finishReason: undefined,
        usageMetadata: undefined,
        promptBlockReason: undefined,
      });
    };

    const runWithContext = (conversationContext: string, userName: string) =>
      injectGeminiToken("test-token")(
        injectGeminiSdkExchange(fakeExchange)(() =>
          agentDeps([
            participantUtteranceTurn({
              name: userName,
              text: "Where is my package?",
            }),
          ])(runAgent)({
            maxIterations: 2,
            disableStreaming: true,
            tools: [sampleTool],
            prompt: "You are a support assistant.",
            conversationContext,
            timezoneIANA: "UTC",
          })
        ),
      )();

    await runWithContext("1:1 chat with Alice (+15550000001)", "Alice");
    await runWithContext("1:1 chat with Bob (+15550000002)", "Bob");

    assertEquals(capturedReqs.length, 2);
    const [reqAlice, reqBob] = capturedReqs;
    assertEquals(
      reqAlice.config?.systemInstruction,
      reqBob.config?.systemInstruction,
    );
    assert(
      !JSON.stringify(reqAlice.config?.systemInstruction).includes("Alice"),
    );
    assertEquals(
      JSON.stringify(reqAlice.config?.tools),
      JSON.stringify(reqBob.config?.tools),
    );
    assert(Array.isArray(reqAlice.contents) && Array.isArray(reqBob.contents));
    assertEquals(
      JSON.stringify(reqAlice.contents.slice(0, 2)),
      JSON.stringify(reqBob.contents.slice(0, 2)),
    );
    assert(
      JSON.stringify(reqAlice.contents).includes(
        formatSystemNotification("1:1 chat with Alice (+15550000001)"),
      ),
    );
    assert(
      JSON.stringify(reqBob.contents).includes(
        formatSystemNotification("1:1 chat with Bob (+15550000002)"),
      ),
    );
  },
);

Deno.test(
  "getSpecForTurn preserves the static skills catalog prefix when a skill becomes active",
  async () => {
    const baseSpec = {
      tools: [],
      skills: [
        {
          name: "alpha",
          description: "Alpha operations",
          instructions: "ALPHA_DETAILED_INSTRUCTIONS",
          tools: [],
        },
        {
          name: "beta",
          description: "Beta operations",
          instructions: "BETA_DETAILED_INSTRUCTIONS",
          tools: [],
        },
      ],
      prompt: "You are a helpful assistant.",
    };

    const specBefore = getSpecForTurn(baseSpec, []);
    const historyAfterLearn: HistoryEvent[] = [
      participantUtteranceTurn({ name: "user", text: "use alpha" }),
      {
        id: "learn-1",
        type: "tool_call",
        isOwn: true,
        name: learnSkillToolName,
        parameters: { skillName: "alpha" },
        timestamp: 1000,
      },
      {
        id: "learn-res-1",
        type: "tool_result",
        isOwn: true,
        toolCallId: "learn-1",
        result: "Skill learned.",
        timestamp: 1001,
      },
    ];

    const specAfter = getSpecForTurn(baseSpec, historyAfterLearn);
    assert(
      specAfter.prompt.startsWith(specBefore.prompt),
      "Activating a skill must append active instructions without mutating the preceding prompt prefix",
    );
    assert(specAfter.prompt.includes("ALPHA_DETAILED_INSTRUCTIONS"));

    const iterReqs: GenerateContentParameters[] = [];
    const skillRunHistory: HistoryEvent[] = [
      participantUtteranceTurn({ name: "user", text: "use alpha" }),
    ];
    await injectGeminiToken("test-token")(
      injectGeminiSdkExchange((_signal, { req }) => {
        iterReqs.push(req);
        return Promise.resolve({
          parts: iterReqs.length === 1
            ? [{
              functionCall: {
                id: "learn-alpha-1",
                name: learnSkillToolName,
                args: { skillName: "alpha" },
              },
              thoughtSignature: "c2lnLWFscGhhLTE=",
            }]
            : [{ text: "Done with alpha." }],
          finishReason: undefined,
          usageMetadata: undefined,
          promptBlockReason: undefined,
        });
      })(() =>
        agentDeps(skillRunHistory)(runAgent)({
          maxIterations: 4,
          disableStreaming: true,
          tools: [],
          skills: baseSpec.skills,
          prompt: baseSpec.prompt,
          timezoneIANA: "UTC",
        })
      ),
    )();
    assertEquals(iterReqs.length, 2);
    assertEquals(
      iterReqs[0].config?.systemInstruction,
      iterReqs[1].config?.systemInstruction,
      "Skill activation must keep systemInstruction byte-identical across iter=1 and iter=2",
    );
    assert(
      Array.isArray(iterReqs[0].contents) &&
        Array.isArray(iterReqs[1].contents),
    );
    assertEquals(
      JSON.stringify(iterReqs[0].contents.slice(0, 2)),
      JSON.stringify(iterReqs[1].contents.slice(0, 2)),
    );
    assert(
      JSON.stringify(iterReqs[1].contents).includes(
        "ALPHA_DETAILED_INSTRUCTIONS",
      ),
    );
  },
);

Deno.test(
  "terminal answersUser tool finishes in 1 model call on success and retries on error or alongside non-terminal tools",
  async () => {
    const sentQuotes: string[] = [];
    const quoteReplyTool = tool({
      name: "quote_reply",
      description: "Reply by quoting a message.",
      parameters: z.object({ target: z.string(), text: z.string() }),
      handler: ({ target, text }) => {
        if (target === "missing") {
          return Promise.resolve("Target message not found.");
        }
        sentQuotes.push(text);
        return Promise.resolve({
          result: "Quoted reply sent.",
          terminal: true,
          answersUser: true,
        });
      },
    });
    const fetchDataTool = tool({
      name: "fetch_data",
      description: "Fetch external data.",
      parameters: z.object({ key: z.string() }),
      handler: ({ key }) => Promise.resolve(`value:${key}`),
    });

    const successCalls: number[] = [];
    const successHistory: HistoryEvent[] = [
      participantUtteranceTurn({ name: "user", text: "When is check-in?" }),
    ];
    await injectCallModel(() => {
      successCalls.push(1);
      return Promise.resolve([
        toolUseTurn({
          id: "q-1",
          name: "quote_reply",
          args: { target: "When is check-in?", text: "Check-in is at 3 PM." },
        }),
      ]);
    })(() =>
      agentDeps(successHistory)(runAgent)({
        maxIterations: 5,
        tools: [quoteReplyTool],
        prompt: "Answer clearly.",
        timezoneIANA: "UTC",
      })
    )();
    assertEquals(successCalls.length, 1);
    assertEquals(sentQuotes, ["Check-in is at 3 PM."]);
    assert(successHistory.some((e) => e.type === "tool_result"));

    const recoveryCalls: number[] = [];
    const recoveryHistory: HistoryEvent[] = [
      participantUtteranceTurn({ name: "user", text: "When is check-out?" }),
    ];
    await injectCallModel(() => {
      recoveryCalls.push(1);
      return Promise.resolve(
        recoveryCalls.length === 1
          ? [
            toolUseTurn({
              id: "q-err",
              name: "quote_reply",
              args: { target: "missing", text: "11 AM" },
            }),
          ]
          : [ownUtteranceTurn("Check-out is at 11 AM.")],
      );
    })(() =>
      agentDeps(recoveryHistory)(runAgent)({
        maxIterations: 5,
        tools: [quoteReplyTool],
        prompt: "Answer clearly.",
        timezoneIANA: "UTC",
      })
    )();
    assertEquals(recoveryCalls.length, 2);
    assert(
      recoveryHistory.some((e) =>
        e.type === "own_utterance" && e.text === "Check-out is at 11 AM."
      ),
    );

    const mixedCalls: number[] = [];
    const mixedHistory: HistoryEvent[] = [
      participantUtteranceTurn({ name: "user", text: "Check status" }),
    ];
    await injectCallModel(() => {
      mixedCalls.push(1);
      return Promise.resolve(
        mixedCalls.length === 1
          ? [
            toolUseTurn({
              id: "q-ack",
              name: "quote_reply",
              args: { target: "Check status", text: "All set, shipped!" },
            }),
            toolUseTurn({
              id: "f-1",
              name: "fetch_data",
              args: { key: "order" },
            }),
          ]
          : [doNothingEvent("already answered via quote")],
      );
    })(() =>
      agentDeps(mixedHistory)(runAgent)({
        maxIterations: 5,
        tools: [quoteReplyTool, fetchDataTool],
        prompt: "Answer clearly.",
        timezoneIANA: "UTC",
      })
    )();
    assertEquals(mixedCalls.length, 2);
  },
);

Deno.test(
  "terminal acknowledgesUser reaction tool ends in 1 call when paired with utterance, and allows clean do_nothing on iter 2 when used alone",
  async () => {
    const reactTool = tool({
      name: "react_emoji",
      description: "React with an emoji.",
      parameters: z.object({ emoji: z.string() }),
      handler: ({ emoji }) =>
        Promise.resolve({
          result: `Reacted ${emoji}`,
          terminal: true,
          acknowledgesUser: true,
        }),
    });

    const pairedCalls: number[] = [];
    const pairedHistory: HistoryEvent[] = [
      participantUtteranceTurn({ name: "user", text: "Great news!" }),
    ];
    await injectCallModel(() => {
      pairedCalls.push(1);
      return Promise.resolve([
        ownUtteranceTurn("Glad to hear it!"),
        toolUseTurn({ id: "r-1", name: "react_emoji", args: { emoji: "🎉" } }),
      ]);
    })(() =>
      agentDeps(pairedHistory)(runAgent)({
        maxIterations: 5,
        tools: [reactTool],
        prompt: "Be friendly.",
        timezoneIANA: "UTC",
      })
    )();
    assertEquals(pairedCalls.length, 1);

    const soloAckCalls: number[] = [];
    const soloAckHistory: HistoryEvent[] = [
      participantUtteranceTurn({ name: "user", text: "Thanks!" }),
    ];
    await injectCallModel(() => {
      soloAckCalls.push(1);
      return Promise.resolve(
        soloAckCalls.length === 1
          ? [
            toolUseTurn({
              id: "r-2",
              name: "react_emoji",
              args: { emoji: "👍" },
            }),
          ]
          : [doNothingEvent("reaction sufficient")],
      );
    })(() =>
      agentDeps(soloAckHistory)(runAgent)({
        maxIterations: 5,
        tools: [reactTool],
        prompt: "Be concise.",
        timezoneIANA: "UTC",
      })
    )();
    assertEquals(soloAckCalls.length, 2);
  },
);
