import { assert } from "@std/assert";
import {
  type HistoryEvent,
  ownThoughtTurn,
  participantUtteranceTurn,
} from "../src/agent.ts";
import {
  agentDeps,
  b64,
  noopRewriteHistory,
  recognizedTheDog,
  runForAllProviders,
} from "../test_helpers.ts";

runForAllProviders(
  "own_thought attachments are forwarded to model",
  async (runAgentWithProvider) => {
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Describe what you observe.",
      }),
      ownThoughtTurn(
        "[Internal context from another system: an image was attached below for you to analyze.]",
        [{ kind: "inline", mimeType: "image/jpeg", dataBase64: b64 }],
      ),
    ];
    await agentDeps(mockHistory)(runAgentWithProvider)({
      maxIterations: 3,
      onMaxIterationsReached: () => {},
      tools: [],
      prompt:
        "You can see images attached alongside internal thoughts. Describe what you see.",
      lightModel: true,
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
    assert(
      mockHistory.some(recognizedTheDog),
      `AI did not describe the image attached to the internal thought. History: ${
        JSON.stringify(mockHistory, null, 2)
      }`,
    );
  },
);
