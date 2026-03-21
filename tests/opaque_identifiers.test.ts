import { assertEquals } from "@std/assert";
import {
  extractOpaqueIdentifiers,
  findNovelOpaqueIdentifiers,
  modelOutputHasNovelOpaqueIdentifiers,
  ownUtteranceTurn,
  participantUtteranceTurn,
} from "../mod.ts";

Deno.test("extractOpaqueIdentifiers handles urls and params", () => {
  assertEquals(
    extractOpaqueIdentifiers(
      "https://example.com/orders/ord_12345?session_id=sess_67890&slug=summer-sale",
    ).sort(),
    ["ord_12345", "sess_67890"],
  );
});

Deno.test("extractOpaqueIdentifiers ignores regular words", () => {
  assertEquals(
    extractOpaqueIdentifiers(
      "Use https://example.com/users/profile?tab=settings and mention checkout later",
    ),
    [],
  );
});

Deno.test("findNovelOpaqueIdentifiers detects ids absent from sources", () => {
  assertEquals(
    findNovelOpaqueIdentifiers(
      "Open https://example.com/orders/ord_NEW123?session_id=sess_67890",
      [
        "Prompt mentions https://example.com/orders/ord_OLD999?session_id=sess_67890",
        { result: "Known user usr_abc123" },
      ],
    ),
    ["ord_NEW123"],
  );
});

Deno.test("modelOutputHasNovelOpaqueIdentifiers flags unknown ids inside urls", () => {
  assertEquals(
    modelOutputHasNovelOpaqueIdentifiers(
      "Known session sess_67890",
      [participantUtteranceTurn({ name: "user", text: "continue" })],
      [
        ownUtteranceTurn(
          "Open https://example.com/orders/ord_NEW123?session_id=sess_67890",
        ),
      ],
    ),
    true,
  );
});

Deno.test("modelOutputHasNovelOpaqueIdentifiers allows legitimate known ids", () => {
  assertEquals(
    modelOutputHasNovelOpaqueIdentifiers(
      "Known order ord_12345",
      [
        participantUtteranceTurn({ name: "user", text: "continue" }),
        {
          type: "tool_result",
          isOwn: true,
          id: crypto.randomUUID(),
          timestamp: Date.now(),
          name: "fetchOrder",
          result: "Order id is ord_12345 and session sess_67890",
        },
      ],
      [
        ownUtteranceTurn(
          "Open https://example.com/orders/ord_12345?session_id=sess_67890",
        ),
      ],
    ),
    false,
  );
});
