import { assertEquals } from "@std/assert";
import { callToResult } from "../src/agent.ts";
import { z } from "zod/v4";

Deno.test("global tool output sanitization - resolves carriage returns and collapses duplicates", async () => {
  const dummyTool = {
    name: "dummy",
    description: "test",
    parameters: z.object({}),
    handler: () => {
      return Promise.resolve(
        "Downloading...\r[===       ] 10%\r[======    ] 50%\r[==========] 100%\nDone!\nSuccess\nSuccess\nSuccess",
      );
    },
  };

  const resolver = callToResult([dummyTool]);
  const res = await resolver({
    name: "dummy",
    args: {},
    id: "call-1",
  });

  assertEquals(res?.toolCallId, "call-1");
  assertEquals(
    res?.result,
    "[==========] 100%\nDone!\nSuccess (repeated 3 times)",
  );
});
