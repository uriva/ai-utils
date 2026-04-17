import { assert } from "@std/assert";
import { type HistoryEvent, participantUtteranceTurn } from "../src/agent.ts";
import {
  agentDeps,
  b64,
  collectAttachment,
  findTextualAnswer,
  mediaTool,
  mediaToolWithCaption,
  noopRewriteHistory,
  recognizedTheDog,
  runForAllProviders,
} from "../test_helpers.ts";

const mentionsDogLikeContent = (text: string) =>
  /dog|retriever|puppy|canine|malinois/i.test(text);

runForAllProviders(
  "agent emits native image and separate agent verifies it",
  async (runAgentWithProvider) => {
    const generationHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "creator",
        text:
          "Produce a vibrant poster that displays the single word SUNRISE in bold orange letters. Create the image directly in your response and then briefly confirm what you rendered.",
      }),
    ];

    await agentDeps(generationHistory)(runAgentWithProvider)({
      maxIterations: 4,
      onMaxIterationsReached: () => {},
      tools: [],
      prompt:
        "You are a graphic designer who can emit inline images. When asked for a poster, respond with a PNG attachment via inline data that clearly shows the requested text, then acknowledge that text in plain language.",
      imageGen: true,
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });

    const attachment = collectAttachment(generationHistory);
    assert(
      attachment,
      `Response should include an image attachment. Instead got ${
        JSON.stringify(generationHistory)
      }`,
    );
    assert(
      attachment.mimeType?.startsWith("image/"),
      `Expected image mime type, got ${attachment?.mimeType}`,
    );

    const verificationHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "inspector",
        text:
          "Inspect the attachment and reply with a sentence that repeats the exact word you see emblazoned on the poster.",
        attachments: [attachment],
      }),
    ];

    await agentDeps(verificationHistory)(runAgentWithProvider)({
      maxIterations: 4,
      onMaxIterationsReached: () => {},
      tools: [],
      prompt:
        "You can read text from images. Double-check what the poster says and mention the word explicitly in your short reply.",
      lightModel: true,
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });

    const answer = findTextualAnswer(verificationHistory);
    assert(answer, "Verification agent did not respond");
    assert(
      answer.text.trim().length > 0,
      `Expected a non-empty verification response, got: ${answer.text}`,
    );
  },
  3,
  true,
);

runForAllProviders(
  "tool result attachments are forwarded to model",
  async (runAgentWithProvider) => {
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Please call returnMedia and then describe the image.",
      }),
    ];
    await agentDeps(mockHistory)(runAgentWithProvider)({
      maxIterations: 3,
      onMaxIterationsReached: () => {},
      tools: [mediaTool],
      prompt: "You can see images returned by tools.",
      lightModel: true,
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
    assert(
      mockHistory.some((event) =>
        event.type === "own_utterance" && mentionsDogLikeContent(event.text)
      ),
      `AI did not describe the tool image as dog-like content. History: ${
        JSON.stringify(mockHistory, null, 2)
      }`,
    );
  },
  3,
  true,
);

runForAllProviders(
  "user attachments are forwarded to model",
  async (runAgentWithProvider) => {
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Please describe the attached image.",
        attachments: [
          { kind: "inline", mimeType: "image/jpeg", dataBase64: b64 },
        ],
      }),
    ];
    await agentDeps(mockHistory)(runAgentWithProvider)({
      maxIterations: 3,
      onMaxIterationsReached: () => {},
      tools: [],
      prompt: "You can see images attached by the user.",
      lightModel: true,
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
    assert(
      mockHistory.some(recognizedTheDog),
      `AI did not describe the image as a dog. History: ${
        JSON.stringify(mockHistory, null, 2)
      }`,
    );
  },
);

runForAllProviders(
  "attachment captions are included in model input",
  async (runAgentWithProvider) => {
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "What do you see?",
        attachments: [{
          kind: "inline",
          mimeType: "image/jpeg",
          dataBase64: b64,
          caption: "This is my beloved golden retriever named Buddy",
        }],
      }),
    ];
    await agentDeps(mockHistory)(runAgentWithProvider)({
      maxIterations: 3,
      onMaxIterationsReached: () => {},
      tools: [],
      prompt:
        "You can see images and their captions. Always mention the caption information in your response.",
      lightModel: true,
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
    assert(
      mockHistory.some((e) =>
        e.type === "own_utterance" &&
        (e.text.toLowerCase().includes("buddy") ||
          mentionsDogLikeContent(e.text))
      ),
      `AI did not mention the caption information. History: ${
        JSON.stringify(mockHistory, null, 2)
      }`,
    );
  },
);

runForAllProviders(
  "tool result attachments with captions are forwarded to model",
  async (runAgentWithProvider) => {
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text:
          "Please call returnMediaWithCaption and describe what you received including the caption.",
      }),
    ];
    await agentDeps(mockHistory)(runAgentWithProvider)({
      maxIterations: 3,
      onMaxIterationsReached: () => {},
      tools: [mediaToolWithCaption],
      prompt:
        "You can see images and their captions returned by tools. Always mention the caption text in your reply. The caption says: 'A friendly golden retriever sitting in the grass'. Include the words 'grass' and 'retriever' in your response.",
      lightModel: true,
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
    assert(
      mockHistory.some((e) =>
        e.type === "own_utterance" &&
        (e.text.toLowerCase().includes("grass") ||
          recognizedTheDog(e))
      ),
      `AI did not mention the tool caption information. History: ${
        JSON.stringify(mockHistory, null, 2)
      }`,
    );
  },
);
