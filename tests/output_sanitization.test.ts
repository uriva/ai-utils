import { assertEquals } from "@std/assert";
import { callToResult } from "../src/agent.ts";
import { z } from "zod/v4";

Deno.test("global tool output sanitization - resolves carriage returns, collapses duplicates, and collapses similar prefixes", async () => {
  const dummyTool = {
    name: "dummy",
    description: "test",
    parameters: z.object({}),
    handler: () => {
      return Promise.resolve(
        [
          "Downloading...\r[===       ] 10%\r[======    ] 50%\r[==========] 100%",
          "Done!",
          "Success",
          "Success",
          "Success",
          "\u001b[32mDownload https://jsr.io/@std/semver/meta.json\u001b[0m",
          "\u001b[32mDownload https://jsr.io/@std/fmt/meta.json\u001b[0m",
          "\u001b[32mDownload https://jsr.io/@std/path/meta.json\u001b[0m",
        ].join("\n"),
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
    [
      "[==========] 100%",
      "Done!",
      "Success (repeated 3 times)",
      "Download https://jsr.io/@std/... (collapsed 3 structurally similar lines)",
    ].join("\n"),
  );
});
