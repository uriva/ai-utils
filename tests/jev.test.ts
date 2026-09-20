import { assertEquals } from "@std/assert";
import {
  formatAgentStateForJev,
  geminiFlashVersion,
  geminiProVersion,
  injectJevToken,
  participantUtteranceTurn,
  routeTaskWithJev,
} from "../mod.ts";

const testJevToken = Deno.env.get("JEV_API_KEY") ||
  "apikey_2323b4352d1a3ea4a0787a1689e16cabcd9_096712d3a92df92963ceec3b230db7e9551e027e997111bd989ace89f4716c75";
const withTestJevToken = injectJevToken(testJevToken);

Deno.test("geminiModelVersions eliminates 3.7-flash and aligns pro with 3.8-flash", () => {
  assertEquals(geminiProVersion, "gemini-3.8-flash");
  assertEquals(geminiFlashVersion, "gemini-3.8-flash");
});

Deno.test("routeTaskWithJev falls back to flash when no token is present", async () => {
  await injectJevToken("")(async () => {
    const tier = await routeTaskWithJev("Hello, what is your name?");
    assertEquals(tier, "flash");
  })();
});

Deno.test("routeTaskWithJev routes simple greeting to lite", async () => {
  await withTestJevToken(async () => {
    const tier = await routeTaskWithJev("Hi, what time does the venue open?");
    assertEquals(tier, "lite");
  })();
});

Deno.test("routeTaskWithJev routes complex coding/architecture to flash", async () => {
  await withTestJevToken(async () => {
    const tier = await routeTaskWithJev(
      "Implement a distributed Byzantine fault tolerant consensus algorithm in Rust with formal TLA+ specification and unit tests.",
    );
    assertEquals(tier, "flash");
  })();
});

Deno.test("formatAgentStateForJev extracts user request and tool names", () => {
  const state = formatAgentStateForJev(
    "You are a helpful events concierge.",
    [participantUtteranceTurn({
      name: "User",
      text: "What events are happening in Berlin?",
    })],
    [{ name: "query" }, { name: "event_by_id" }],
  );
  assertEquals(state.user_request, "What events are happening in Berlin?");
  assertEquals(state.agent_role, "You are a helpful events concierge.");
  assertEquals(state.tools_available, ["query", "event_by_id"]);
});
