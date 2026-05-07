import { assert } from "@std/assert";
import { z } from "zod/v4";
import { type HistoryEvent, participantUtteranceTurn } from "../src/agent.ts";
import {
  agentDeps,
  b64,
  collectAttachment,
  findTextualAnswer,
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
  false,
);

runForAllProviders(
  "tool inline attachment is forwarded to model",
  async (runAgentWithProvider) => {
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text:
          "Please call returnRawDogImageForVisualChoiceTest and describe the attached animal directly.",
      }),
    ];
    await agentDeps(mockHistory)(runAgentWithProvider)({
      maxIterations: 3,
      onMaxIterationsReached: () => {},
      tools: [{
        name: "returnRawDogImageForVisualChoiceTest",
        description: "Returns raw image bytes as an inline attachment",
        parameters: z.object({}),
        handler: () =>
          Promise.resolve({
            result: "Raw image attached.",
            attachments: [{
              kind: "inline" as const,
              mimeType: "image/jpeg",
              dataBase64: b64,
            }],
          }),
      }],
      prompt:
        "You can see raw images returned by tools. Do not call inspect_media_url for inline media; describe the attached image directly.",
      lightModel: true,
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
    assert(
      mockHistory.some(recognizedTheDog),
      `AI did not describe the raw tool image as a dog. History: ${
        JSON.stringify(mockHistory, null, 2)
      }`,
    );
  },
  3,
  true,
  false,
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
  "user file attachment with external url is uploaded to gemini",
  async (runAgentWithProvider) => {
    const ac = new AbortController();
    const server = Deno.serve({ port: 0, signal: ac.signal }, async (req) => {
      const url = new URL(req.url);
      if (url.pathname === "/dog.jpg") {
        const data = await Deno.readFile("./dog.jpg");
        return new Response(data, {
          headers: { "content-type": "image/jpeg" },
        });
      }
      return new Response("Not found", { status: 404 });
    });
    const addr = server.addr as Deno.NetAddr;
    const imageUrl = `http://localhost:${addr.port}/dog.jpg`;

    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Please describe the attached image.",
        attachments: [
          { kind: "file", mimeType: "image/jpeg", fileUri: imageUrl },
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

    ac.abort();
    try {
      await server.finished;
    } catch {
      // expected on abort
    }

    assert(
      mockHistory.some(recognizedTheDog),
      `AI did not describe the image as a dog. History: ${
        JSON.stringify(mockHistory, null, 2)
      }`,
    );
  },
  3,
  true,
  false,
);

runForAllProviders(
  "tool file attachment url is inspected only by explicit tool call",
  async (runAgentWithProvider) => {
    const ac = new AbortController();
    const server = Deno.serve({ port: 0, signal: ac.signal }, async (req) => {
      const url = new URL(req.url);
      if (url.pathname === "/dog.jpg") {
        const data = await Deno.readFile("./dog.jpg");
        return new Response(data, {
          headers: { "content-type": "image/jpeg" },
        });
      }
      return new Response("Not found", { status: 404 });
    });
    const addr = server.addr as Deno.NetAddr;
    const imageUrl = `http://localhost:${addr.port}/dog.jpg`;

    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text:
          "Please call returnMediaUrl and inspect the returned URL before describing the image.",
      }),
    ];

    await agentDeps(mockHistory)(runAgentWithProvider)({
      maxIterations: 5,
      onMaxIterationsReached: () => {},
      tools: [{
        name: "returnMediaUrl",
        description: "Returns an image URL as a file attachment",
        parameters: z.object({}),
        handler: () =>
          Promise.resolve({
            result: `Download URL: ${imageUrl}`,
            attachments: [{
              kind: "file" as const,
              mimeType: "image/jpeg",
              fileUri: imageUrl,
              caption: "dog photo",
            }],
          }),
      }],
      prompt:
        "When a tool returns a media URL, call inspect_media_url to look at it before answering.",
      lightModel: true,
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });

    ac.abort();
    try {
      await server.finished;
    } catch {
      // expected on abort
    }

    assert(
      mockHistory.some((event) =>
        event.type === "tool_call" && event.name === "inspect_media_url"
      ),
      `AI did not explicitly inspect the media URL. History: ${
        JSON.stringify(mockHistory, null, 2)
      }`,
    );
    assert(
      mockHistory.some(recognizedTheDog),
      `AI did not describe the inspected image as a dog. History: ${
        JSON.stringify(mockHistory, null, 2)
      }`,
    );
  },
  3,
  true,
  false,
);
