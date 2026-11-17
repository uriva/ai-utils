import { assert, assertEquals } from "@std/assert";
import { ThinkingLevel } from "@google/genai";
import { pipe } from "gamla";
import { z } from "zod/v4";
import { runAgent } from "../mod.ts";
import {
  externalEventTurn,
  type HistoryEvent,
  injectAccessHistory,
  injectCallModel,
  injectOutputEvent,
  learnSkillToolName,
  normalizeHistoryForModel,
  ownUtteranceTurn,
  participantUtteranceTurn,
  type Skill,
  toolResultTurn,
  toolUseTurn,
} from "../src/agent.ts";
import { injectDecisionModel } from "../src/decisionModel.ts";
import { buildReq } from "../src/geminiAgent.ts";
import {
  alreadyVisibleMediaMessage,
  injectInspectMediaTimeoutMs,
  inspectMediaUrlToolName,
} from "../src/inspectMediaTool.ts";

const inMemoryDeps = (inMemoryHistory: HistoryEvent[]) =>
  pipe(
    injectAccessHistory(() => Promise.resolve(inMemoryHistory)),
    injectOutputEvent((event) => {
      inMemoryHistory.push(event);
      return Promise.resolve();
    }),
  );

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

type ToolCallEvent = Extract<HistoryEvent, { type: "tool_call" }>;
type ToolResultEvent = Extract<HistoryEvent, { type: "tool_result" }>;

const isLearnCall = (e: HistoryEvent): e is ToolCallEvent =>
  e.type === "tool_call" && e.name === learnSkillToolName;

const isToolResult = (e: HistoryEvent): e is ToolResultEvent =>
  e.type === "tool_result";

const isRecord = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v);

Deno.test(
  "normalizeHistoryForModel treats interrupted synchronous tool_calls as interrupted rather than pending background tasks when deferredToolNames is specified",
  () => {
    const syncCall = toolUseTurn({
      name: "run_command",
      args: { command: "workspace/write_file", params: { path: "receipt.md" } },
    });
    const deferredCall = toolUseTurn({
      name: "run_command",
      args: {
        command: "automation/run_in_new_thread",
        params: { prompt: "x" },
      },
    });
    const deferredSet = new Set(["run_in_new_thread", "timeout-wakeup"]);

    const syncHistory: HistoryEvent[] = [
      participantUtteranceTurn({ name: "user", text: "prepare receipt" }),
      syncCall,
      participantUtteranceTurn({ name: "user", text: "done" }),
    ];
    const callNormalize: (
      h: HistoryEvent[],
      d?: ReadonlySet<string>,
    ) => HistoryEvent[] = normalizeHistoryForModel;
    const normalizedSync = callNormalize(syncHistory, deferredSet);
    assert(
      !normalizedSync.some((e) =>
        e.type === "own_utterance" && e.text.includes("background task")
      ),
      "interrupted synchronous tool call must NOT be replaced with a pending background task acknowledgement",
    );
    assert(
      !normalizedSync.some((e) =>
        e.type === "own_thought" && e.text.includes("background task")
      ),
      "interrupted synchronous tool call must NOT inject pending background task notification",
    );
    const syncResult = normalizedSync.find(
      (e): e is ToolResultEvent =>
        isToolResult(e) && e.toolCallId === syncCall.id,
    );
    assert(
      syncResult && syncResult.result.toLowerCase().includes("interrupted"),
      `expected synthetic interrupted tool_result for sync call, got: ${
        JSON.stringify(syncResult)
      }`,
    );

    const deferredHistory: HistoryEvent[] = [
      participantUtteranceTurn({ name: "user", text: "research options" }),
      deferredCall,
      participantUtteranceTurn({ name: "user", text: "are you there?" }),
    ];
    const normalizedDeferred = callNormalize(deferredHistory, deferredSet);
    assert(
      normalizedDeferred.some((e) =>
        e.type === "own_utterance" && e.text.includes("background task")
      ),
      "genuinely deferred tool call must still emit background task acknowledgement",
    );
  },
);

Deno.test(
  "decideSkillsWithDecisionModel evaluates trailing external_event rather than stale answered participant_utterance",
  async () => {
    const travelSkill: Skill = {
      name: "travel_guide",
      description: "City sightseeing and food guide",
      instructions: "Help plan sightseeing.",
      tools: [],
    };
    const templatesSkill: Skill = {
      name: "message_templates",
      description: "Send approved messaging templates for system updates",
      instructions: "Check templates before sending.",
      tools: [],
    };
    const history: HistoryEvent[] = [
      {
        ...participantUtteranceTurn({
          name: "user",
          text: "Activate travel_guide for my sightseeing trip",
        }),
        timestamp: 1_700_000_000_000,
      },
      {
        ...ownUtteranceTurn("Travel guide is ready! How can I help?"),
        timestamp: 1_700_000_005_000,
      },
      {
        ...externalEventTurn(
          "SYSTEM UPDATE: Account referral link is now active. Check approved templates.",
        ),
        timestamp: 1_701_000_000_000,
      },
    ];
    const observedRequests: string[] = [];

    await pipe(
      inMemoryDeps(history),
      injectDecisionModel((state, questions) => {
        const reqText = isRecord(state) &&
            typeof state.current_user_request === "string"
          ? state.current_user_request
          : "";
        if (reqText) observedRequests.push(reqText);
        return Promise.resolve(
          Object.fromEntries(
            Object.keys(questions).map((k) =>
              k === "thinking_level"
                ? [k, { type: "choice" as const, choice: "low" }]
                : [
                  k,
                  {
                    type: "noul" as const,
                    noul: reqText.includes("travel_guide") &&
                        k === "travel_guide"
                      ? 0.95
                      : reqText.includes("referral") &&
                          k === "message_templates"
                      ? 0.95
                      : 0.05,
                  },
                ]
            ),
          ),
        );
      }),
      injectCallModel(() => Promise.resolve([ownUtteranceTurn("Handled.")])),
    )(() =>
      runAgent({
        prompt: "You are a helpful assistant.",
        tools: [],
        skills: [travelSkill, templatesSkill],
        maxIterations: 3,
        timezoneIANA: "UTC",
      })
    )();

    assert(
      observedRequests.some((r) => r.includes("SYSTEM UPDATE")),
      `expected decision model current_user_request to reflect trailing external_event, got: ${
        JSON.stringify(observedRequests)
      }`,
    );
    const learnedSkills = history
      .filter(isLearnCall)
      .map((e) =>
        isRecord(e.parameters) && typeof e.parameters.skillName === "string"
          ? e.parameters.skillName
          : ""
      );
    assertEquals(
      learnedSkills,
      ["message_templates"],
      `expected only message_templates to be auto-learned on referral external_event, got: ${
        JSON.stringify(learnedSkills)
      }`,
    );
  },
);

Deno.test(
  "runAgent deduplicates concurrent auto-learn and model learn_skill calls and skips redundant re-learn of active skills",
  async () => {
    const mediaSkill: Skill = {
      name: "video_director",
      description: "Direct and produce short videos",
      instructions: "Follow the brief -> approval -> render pipeline.",
      tools: [{
        name: "check_quota",
        description: "Check video API quota",
        parameters: z.object({}),
        handler: () => Promise.resolve("quota=ok"),
      }],
    };
    const history: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Create a short video reel from this photo",
      }),
    ];
    const counter = { iter: 0 };

    await pipe(
      inMemoryDeps(history),
      injectDecisionModel(async (_state, questions) => {
        await delay(10);
        return Object.fromEntries(
          Object.keys(questions).map((k) =>
            k === "thinking_level"
              ? [k, { type: "choice" as const, choice: "high" }]
              : [
                k,
                {
                  type: "noul" as const,
                  noul: k === "video_director" ? 0.95 : 0.0,
                },
              ]
          ),
        );
      }),
      injectCallModel(async () => {
        counter.iter += 1;
        if (counter.iter === 1) {
          await delay(30);
          return [
            toolUseTurn({
              name: learnSkillToolName,
              args: { skillName: "video_director" },
            }),
          ];
        }
        if (counter.iter === 2) {
          return [
            toolUseTurn({
              name: learnSkillToolName,
              args: { skillName: "video_director" },
            }),
          ];
        }
        return [ownUtteranceTurn("Ready to direct your video.")];
      }),
    )(() =>
      runAgent({
        prompt: "You are a helpful assistant.",
        tools: [],
        skills: [mediaSkill],
        maxIterations: 5,
        timezoneIANA: "UTC",
      })
    )();

    const learnCalls = history.filter(isLearnCall);
    assertEquals(
      learnCalls.length,
      1,
      `expected concurrent auto-learn + model learn_skill + redundant re-learn to coalesce to 1 active learn_skill event, got ${learnCalls.length}: ${
        JSON.stringify(learnCalls)
      }`,
    );
  },
);

Deno.test(
  "tool_result GCS attachment with renderInline renders directly as fileData without prompting for inspect_media_url, and inspect_media_url skips re-fetching already-rendered URLs",
  async () => {
    const gcsFrameUri =
      "https://storage.googleapis.com/example-bucket/frame_01.jpg";
    const extraFields: Record<string, unknown> = { renderInline: true };
    const events = [
      participantUtteranceTurn({ name: "user", text: "inspect frame" }),
      toolUseTurn({
        id: "call-read",
        name: "run_command",
        args: {
          command: "code_execution/read",
          params: { path: "/tmp/f.jpg" },
        },
      }),
      toolResultTurn({
        result: "Image read successfully: /tmp/f.jpg (image/jpeg)",
        toolCallId: "call-read",
        attachments: [{
          kind: "file" as const,
          mimeType: "image/jpeg",
          fileUri: gcsFrameUri,
          ...extraFields,
        }],
      }),
    ];
    const isGeminiInputEvent = (
      e: HistoryEvent,
    ): e is Parameters<ReturnType<typeof buildReq>>[0][number] =>
      !("modelMetadata" in e) || e.modelMetadata === undefined;
    const req = buildReq(
      ThinkingLevel.HIGH,
      "p",
      [],
      "UTC",
      undefined,
    )(events.filter(isGeminiInputEvent));
    const historyEvents: HistoryEvent[] = [...events];
    const contentsList = Array.isArray(req.contents) ? req.contents : [];
    const secondTurn = contentsList[2];
    const readParts: Record<string, unknown>[] = isRecord(secondTurn) &&
        Array.isArray(secondTurn.parts)
      ? secondTurn.parts
      : [];
    const mediaParts = readParts.filter((p) => "fileData" in p);
    const respText = JSON.stringify(
      readParts.find((p) => "functionResponse" in p),
    );
    assertEquals(
      mediaParts.length,
      1,
      "renderInline GCS attachment from code_execution/read must render as fileData",
    );
    assert(
      !respText.includes("call inspect_media_url"),
      "must not tell model to call inspect_media_url when attachment is already rendered inline",
    );

    const counter = { iter: 0 };
    await pipe(
      inMemoryDeps(historyEvents),
      injectCallModel(() => {
        counter.iter += 1;
        return Promise.resolve(
          counter.iter === 1
            ? [
              toolUseTurn({
                name: inspectMediaUrlToolName,
                args: { url: gcsFrameUri, mimeType: "image/jpeg" },
              }),
            ]
            : [ownUtteranceTurn("Frame inspected.")],
        );
      }),
    )(() =>
      runAgent({
        prompt: "You are a helpful assistant.",
        tools: [],
        maxIterations: 4,
        timezoneIANA: "UTC",
      })
    )();

    const redundantResult = historyEvents.findLast(isToolResult);
    assertEquals(
      redundantResult?.result,
      alreadyVisibleMediaMessage(gcsFrameUri),
    );
  },
);

Deno.test(
  "7 parallel inspect_media_url calls with 1 hung server complete cleanly via timeout instead of stalling the turn",
  async () => {
    const validPng = new Uint8Array([
      0x89,
      0x50,
      0x4e,
      0x47,
      0x0d,
      0x0a,
      0x1a,
      0x0a,
    ]);
    const ac = new AbortController();
    const server = Deno.serve(
      { port: 0, signal: ac.signal, onListen: () => {} },
      (req) => {
        const path = new URL(req.url).pathname;
        if (path === "/sheet_7.png") {
          return new Promise<Response>((_resolve, reject) => {
            req.signal.addEventListener(
              "abort",
              () => reject(req.signal.reason),
            );
          });
        }
        return new Response(validPng, {
          status: 200,
          headers: { "content-type": "image/png" },
        });
      },
    );
    const port = server.addr.port;
    const history: HistoryEvent[] = [
      participantUtteranceTurn({ name: "user", text: "inspect all 7 sheets" }),
    ];
    const counter = { iter: 0 };

    await pipe(
      inMemoryDeps(history),
      injectInspectMediaTimeoutMs(() => 60),
      injectCallModel(() => {
        counter.iter += 1;
        return Promise.resolve(
          counter.iter === 1
            ? [1, 2, 3, 4, 5, 6, 7].map((i) =>
              toolUseTurn({
                name: inspectMediaUrlToolName,
                args: {
                  url: `http://127.0.0.1:${port}/sheet_${i}.png`,
                  mimeType: "image/png",
                },
              })
            )
            : [ownUtteranceTurn("Reviewed available sheets.")],
        );
      }),
    )(() =>
      runAgent({
        prompt: "You are a helpful assistant.",
        tools: [],
        maxIterations: 4,
        timezoneIANA: "UTC",
      })
    )();

    ac.abort();
    await server.finished;

    const results = history.filter(isToolResult);
    assertEquals(results.length, 7);
    assert(
      results.some((r) => r.result.includes("timed out after 60ms")),
      `expected hung 7th call to resolve with timeout message, got: ${
        JSON.stringify(results.map((r) => r.result))
      }`,
    );
  },
);
