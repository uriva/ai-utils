import { assertEquals } from "@std/assert";
import {
  type DecisionProvider,
  formatAgentStateForJev,
  geminiFlashVersion,
  geminiModelVersion,
  geminiProVersion,
  injectDecisionProvider,
  injectJevToken,
  injectRespanToken,
  participantUtteranceTurn,
  routeTask,
  ThinkingLevel,
  toolResultTurn,
} from "../mod.ts";
import { buildReq } from "../src/geminiAgent.ts";
import type { HistoryEvent } from "../src/agent.ts";

const decisionProviders: DecisionProvider[] = [
  ...(Deno.env.get("RESPAN_API_KEY") ? ["respan" as const] : []),
  ...(Deno.env.get("JEV_API_KEY") ? ["jev" as const] : []),
];

Deno.test("geminiModelVersions eliminates 3.7-flash and aligns pro with 3.8-flash", () => {
  assertEquals(geminiProVersion, "gemini-3.8-flash");
  assertEquals(geminiFlashVersion, "gemini-3.8-flash");
});

Deno.test("geminiModelVersion resolves lite, pro, and flash", () => {
  assertEquals(geminiModelVersion("lite"), "gemini-3.5-flash-lite");
  assertEquals(geminiModelVersion("pro"), "gemini-3.8-flash");
  assertEquals(geminiModelVersion("flash"), "gemini-3.8-flash");
});

Deno.test("routeTask falls back to flash when no token is present", async () => {
  await injectRespanToken("")(async () => {
    await injectJevToken("")(async () => {
      const tier = await routeTask("Hello, what is your name?");
      assertEquals(tier, "flash");
    });
  });
});

for (const provider of decisionProviders) {
  Deno.test(`routeTask routes simple greeting to lite [${provider}]`, async () => {
    await injectDecisionProvider(provider)(async () => {
      const tier = await routeTask("Hi, what time does the venue open?");
      assertEquals(tier, "lite");
    });
  });

  Deno.test(`routeTask routes complex coding/architecture to flash [${provider}]`, async () => {
    await injectDecisionProvider(provider)(async () => {
      const tier = await routeTask(
        "Implement a distributed Byzantine fault tolerant consensus algorithm in Rust with formal TLA+ specification and unit tests.",
      );
      assertEquals(tier, "flash");
    });
  });

  Deno.test(`routeTask routes tool action request to flash [${provider}]`, async () => {
    await injectDecisionProvider(provider)(async () => {
      const state = formatAgentStateForJev(
        "You are a media editing assistant that processes videos.",
        [
          participantUtteranceTurn({
            name: "User",
            text: "continue this video for 90 more seconds",
          }),
        ],
        [{ name: "download_clip" }, { name: "extract_frame" }],
      );
      const tier = await routeTask(state);
      assertEquals(tier, "flash");
    });
  });

  Deno.test(`routeTask routes search and recommendation queries to lite [${provider}]`, async () => {
    await injectDecisionProvider(provider)(async () => {
      const state = formatAgentStateForJev(
        "You are an events concierge guiding users to concerts and parties.",
        [
          participantUtteranceTurn({
            name: "User",
            text:
              "What live concerts and parties are happening in Berlin this weekend?",
          }),
        ],
        [{ name: "query" }, { name: "upsert_subscription" }],
      );
      const tier = await routeTask(state);
      assertEquals(tier, "lite");
    });
  });

  Deno.test(`routeTask routes presenting search/query results to lite [${provider}]`, async () => {
    await injectDecisionProvider(provider)(async () => {
      const callEvent: HistoryEvent = {
        type: "tool_call",
        id: "call-query-1",
        timestamp: Date.now(),
        isOwn: true,
        name: "query",
        parameters: { location: "Berlin", categories: ["live music"] },
      };
      const resultEvent = toolResultTurn({
        result:
          "1. Jazz Night at A-Trane\n📍 Bleibtreustr. 1, Berlin\n2. Indie Fest at SO36\n📍 Oranienstr. 190, Berlin",
        toolCallId: "call-query-1",
      });
      const state = formatAgentStateForJev(
        "You are an events concierge guiding users to concerts and parties.",
        [
          participantUtteranceTurn({
            name: "User",
            text:
              "What live concerts and parties are happening in Berlin this weekend?",
          }),
          callEvent,
          resultEvent,
        ],
        [{ name: "query" }, { name: "upsert_subscription" }],
      );
      const tier = await routeTask(state);
      assertEquals(tier, "lite");
    });
  });

  Deno.test(`routeTask routes post-tool-result turn with constraints to flash [${provider}]`, async () => {
    await injectDecisionProvider(provider)(async () => {
      const callEvent: HistoryEvent = {
        type: "tool_call",
        id: "call-1",
        timestamp: Date.now(),
        isOwn: true,
        name: "download_clip",
        parameters: { start: "00:01:00", end: "00:02:00" },
      };
      const resultEvent = toolResultTurn({
        result:
          "Download complete and delivered. Do NOT download further continuations.",
        toolCallId: "call-1",
      });
      const state = formatAgentStateForJev(
        "You are a media assistant. Deliver requested clips without extra continuations.",
        [
          participantUtteranceTurn({
            name: "User",
            text: "Start at 00:01:00 and continue for 1 minute.",
          }),
          callEvent,
          resultEvent,
        ],
        [{ name: "download_clip" }],
      );
      const tier = await routeTask(state);
      assertEquals(tier, "flash");
    });
  });
}

Deno.test("formatAgentStateForJev extracts user request and tool names", () => {
  const state = formatAgentStateForJev(
    "You are a helpful events concierge.",
    [participantUtteranceTurn({
      name: "User",
      text: "What events are happening in Berlin?",
    })],
    [{ name: "query" }, { name: "event_by_id" }],
  );
  assertEquals(state.trigger_content, "What events are happening in Berlin?");
  assertEquals(state.agent_role, "You are a helpful events concierge.");
  assertEquals(state.tools_available, ["query", "event_by_id"]);
});

Deno.test("formatAgentStateForJev preserves tool_result content and tool_call in trigger_content and recent_turns", () => {
  const callEvent: HistoryEvent = {
    type: "tool_call",
    id: "call-99",
    timestamp: Date.now(),
    isOwn: true,
    name: "fetch_data",
    parameters: { query: "orders" },
  };
  const resultEvent = toolResultTurn({
    result: "Operation completed successfully with 5 items.",
    toolCallId: "call-99",
  });
  const state = formatAgentStateForJev(
    "You are a database assistant.",
    [
      participantUtteranceTurn({
        name: "User",
        text: "Fetch recent orders.",
      }),
      callEvent,
      resultEvent,
    ],
    [{ name: "fetch_data" }],
  );
  assertEquals(
    state.trigger_content,
    "Operation completed successfully with 5 items.",
  );
  assertEquals(state.trigger_type, "tool_result");
  const recent = state.recent_turns as { type: string; text?: string }[];
  assertEquals(recent.length, 3);
  assertEquals(recent[1].type, "tool_call");
  assertEquals(recent[1].text, 'fetch_data({"query":"orders"})');
  assertEquals(recent[2].type, "tool_result");
  assertEquals(
    recent[2].text,
    "Operation completed successfully with 5 items.",
  );
});

Deno.test("buildReq always uses flash model and applies ThinkingLevel constants without numbers", () => {
  const lowReq = buildReq(
    ThinkingLevel.LOW,
    "You are a helpful assistant.",
    [],
    "UTC",
    undefined,
  )([]);
  assertEquals(lowReq.model, "gemini-3.8-flash");
  assertEquals(lowReq.config?.thinkingConfig?.thinkingLevel, ThinkingLevel.LOW);
  assertEquals(lowReq.config?.thinkingConfig?.includeThoughts, true);
  assertEquals(
    "thinkingBudget" in (lowReq.config?.thinkingConfig ?? {}),
    false,
  );

  const highReq = buildReq(
    ThinkingLevel.HIGH,
    "You are a helpful assistant.",
    [],
    "UTC",
    undefined,
  )([]);
  assertEquals(highReq.model, "gemini-3.8-flash");
  assertEquals(
    highReq.config?.thinkingConfig?.thinkingLevel,
    ThinkingLevel.HIGH,
  );
  assertEquals(highReq.config?.thinkingConfig?.includeThoughts, true);
  assertEquals(
    "thinkingBudget" in (highReq.config?.thinkingConfig ?? {}),
    false,
  );
});

Deno.test("routeTask falls back from Respan to Jev when Respan fails", async () => {
  if (!Deno.env.get("JEV_API_KEY")) return;
  await injectRespanToken("invalid-respan-token")(async () => {
    await injectDecisionProvider("respan")(async () => {
      const tier = await routeTask("Hi, what time does the venue open?");
      assertEquals(tier, "lite");
    });
  });
});

Deno.test("routeTask falls back from Jev to Respan when Jev fails", async () => {
  if (!Deno.env.get("RESPAN_API_KEY")) return;
  await injectJevToken("invalid-jev-token")(async () => {
    await injectDecisionProvider("jev")(async () => {
      const tier = await routeTask("Hi, what time does the venue open?");
      assertEquals(tier, "lite");
    });
  });
});

Deno.test("routeTask falls back to flash when both providers fail", async () => {
  await injectRespanToken("invalid-token")(async () => {
    await injectJevToken("invalid-token")(async () => {
      const tier = await routeTask("Hi, what time does the venue open?");
      assertEquals(tier, "flash");
    });
  });
});
