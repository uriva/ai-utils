import { assert } from "@std/assert";
import { pipe } from "gamla";
import { z } from "zod/v4";
import {
  injectGeminiModelCallTimeoutMs,
  injectGeminiModelVersions,
  tool,
} from "../mod.ts";
import type { HistoryEvent } from "../src/agent.ts";
import {
  agentDeps,
  injectSecrets,
  noopRewriteHistory,
  runWithProvider,
} from "../test_helpers.ts";

const canRunLiveGemini = Deno.env.get("TEST_PROVIDER") === "google" &&
  !!Deno.env.get("GEMINI_API_KEY");

const reproModel = Deno.env.get("GEMINI_REPRO_MODEL") ??
  "gemini-3-flash-preview";
const reproTimeoutMs = Number(
  Deno.env.get("GEMINI_REPRO_TIMEOUT_MS") ?? 60_000,
);
const reproVariant = Deno.env.get("GEMINI_REPRO_VARIANT") ?? "default";

const designBrief = [
  "Fictional project: a premium seaside hotel landing page.",
  "Goal: create a cinematic, conversion-focused web page for mobile and desktop.",
  "Style: elegant editorial layout, deep blue palette, gold accents, soft gradients, spacious typography.",
  "Required sections: hero, trust badges, rooms, experiences, gallery, social proof, FAQ, and a booking CTA.",
  "Implementation: self-contained HTML and CSS, accessible markup, responsive layout, no external build step.",
  "Constraint: continue implementation work by using the file-writing tool rather than sending a status-only reply.",
].join("\n");

const repeatedDesignNotes = Array.from(
  { length: 36 },
  (_, index) =>
    `Design note ${
      index + 1
    }: ${designBrief}\nMicrocopy variant: emphasize calm luxury, clear booking intent, and a polished visual hierarchy.`,
).join("\n\n");

const syntheticHistory: HistoryEvent[] = [
  {
    type: "participant_utterance",
    id: "user-1",
    timestamp: 1,
    isOwn: false,
    name: "user",
    text:
      `Please build a polished landing page for a fictional seaside hotel. Use the provided tools to create the page files.\n\nSynthetic model-under-test marker: ${reproModel}. Synthetic variant marker: ${reproVariant}.\n\n${designBrief}`,
  },
  {
    type: "own_thought",
    id: "thought-1",
    timestamp: 2,
    isOwn: true,
    text: repeatedDesignNotes,
  },
  {
    type: "participant_utterance",
    id: "user-2",
    timestamp: 5,
    isOwn: false,
    name: "user",
    text: "Are you working on the page?",
  },
];

const workspaceTool = (index: number) =>
  tool({
    name: `workspace_action_${index}`,
    description:
      `Generic synthetic workspace action ${index}. Use this for planning, inspecting, validating, and updating fictional project files in a local test workspace. This tool is intentionally generic and contains no platform-specific behavior.`,
    parameters: z.object({
      target: z.string().describe("Synthetic target file or workspace area."),
      instructions: z.string().describe(
        "Detailed synthetic instructions for the workspace action.",
      ),
      rationale: z.string().describe("Brief reason for this action."),
    }),
    handler: () => Promise.resolve(`Synthetic workspace action ${index} done.`),
  });

const writeLandingPage = tool({
  name: "write_landing_page",
  description:
    "Create or update the landing page files for the fictional hotel project. This is the required tool when the user asks whether the page work is progressing.",
  parameters: z.object({
    html: z.string().describe("Complete HTML for the landing page."),
    css: z.string().describe("Complete CSS for the landing page."),
  }),
  handler: () => Promise.resolve("Landing page files written."),
});

Deno.test({
  name: "Gemini buffered agent call emits tool call before timeout [google]",
  ignore: !canRunLiveGemini,
  sanitizeResources: false,
  fn: pipe(
    injectSecrets,
    injectGeminiModelVersions(() => ({
      pro: reproModel,
      flash: reproModel,
      fallback: reproModel,
    })),
    injectGeminiModelCallTimeoutMs(() => reproTimeoutMs),
  )(async () => {
    const history = [...syntheticHistory];
    await agentDeps(history)(runWithProvider(undefined))({
      maxIterations: 1,
      lightModel: true,
      disableStreaming: true,
      maxOutputTokens: 16000,
      tools: [
        ...Array.from({ length: 17 }, (_, index) => workspaceTool(index + 1)),
        writeLandingPage,
      ],
      prompt:
        `You are a web design agent working in a synthetic test project. The project and hotel are fictional. When the user asks whether you are working on the page, continue the work by calling write_landing_page. Do not only send a status update.\n\n${repeatedDesignNotes}`,
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });

    assert(
      history.some((event) =>
        event.type === "tool_call" && event.name === "write_landing_page"
      ),
      `Expected write_landing_page tool call, got events: ${
        history.map((event) => event.type).join(", ")
      }`,
    );
  }),
});
