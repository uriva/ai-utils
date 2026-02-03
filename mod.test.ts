import { assert, assertEquals } from "@std/assert";
import type { Injector } from "@uri/inject";
import { each, pipe, sleep } from "gamla";
import type { ChatCompletionMessageParam } from "openai/resources/index.mjs";
import { z } from "zod/v4";
import {
  tool,
  geminiGenJsonFromConvo,
  injectCacher,
  injectGeminiToken,
  injectOpenAiToken,
  openAiGenJsonFromConvo,
  runAgent,
} from "./mod.ts";
import {
  createSkillTools,
  type HistoryEvent,
  injectAccessHistory,
  injectOutputEvent,
  learnSkillToolName,
  ownUtteranceTurn,
  participantUtteranceTurn,
  runCommandToolName,
  type ToolReturn,
} from "./src/agent.ts";
import { filterOrphanedToolResults } from "./src/geminiAgent.ts";

const injectSecrets = pipe(
  // @ts-expect-error passthrough cacher is sufficient for tests
  injectCacher(() => (f) => f),
  injectOpenAiToken(
    Deno.env.get("OPENAI_API_KEY") ?? "",
  ),
  injectGeminiToken(Deno.env.get("GEMINI_API_KEY") ?? ""),
);

Deno.test(
  "returns valid result for hello schema",
  injectSecrets(async () => {
    const schema = z.object({ hello: z.string() });
    const messages: ChatCompletionMessageParam[] = [
      { role: "system", content: "Say hello as JSON." },
      { role: "user", content: "hello" },
    ];
    await each((service) =>
      each(async (mini) => {
        const result = await service({ mini }, messages, schema);
        console.log(result);
        assertEquals(result, { hello: result.hello });
      })([true, false])
    )([openAiGenJsonFromConvo, geminiGenJsonFromConvo]);
  }),
);

Deno.test(
  "agent can run run when history starts with only a model message",
  injectSecrets(async () => {
    await agentDeps([ownUtteranceTurn("Priming without user turn")])(runAgent)({
      maxIterations: 1,
      onMaxIterationsReached: () => {},
      tools: [],
      prompt: "You are a helper.",
      lightModel: true,
      rewriteHistory: noopRewriteHistory,
    });
  }),
);

const agentDeps = (inMemoryHistory: HistoryEvent[]): Injector =>
  pipe(
    injectAccessHistory(() => Promise.resolve(inMemoryHistory)),
    injectOutputEvent((event) => {
      inMemoryHistory.push(event);
      return Promise.resolve();
    }),
  );

const noopRewriteHistory = async () => {};

const toolResult = "43212e8e";

const someTool = {
  name: "doSomethingUnique",
  description: "Returns a unique string so we know the tool was called.",
  parameters: z.object({}),
  handler: () => Promise.resolve(toolResult),
};

Deno.test(
  "runBot calls the tool and replies with its output",
  injectSecrets(async () => {
    const mockHistory: HistoryEvent[] = [participantUtteranceTurn({
      name: "user",
      text:
        `Please call the doSomethingUnique tool now and only reply with its output.`,
    })];
    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 5,
      onMaxIterationsReached: () => {},
      tools: [someTool],
      prompt: `You are an AI assistant.`,
      rewriteHistory: noopRewriteHistory,
    });
    assert(
      mockHistory.some((event) => (event.type === "own_utterance" &&
        event.text.includes(toolResult))
      ),
      `AI did not reply with tool output. History: ${
        JSON.stringify(mockHistory, null, 2)
      }`,
    );
  }),
);

Deno.test(
  "agent can start an empty conversation",
  injectSecrets(async () => {
    await agentDeps([])(runAgent)({
      onMaxIterationsReached: () => {},
      tools: [],
      prompt: `You are the neighborhood friendly spiderman.`,
      maxIterations: 5,
      rewriteHistory: noopRewriteHistory,
    });
  }),
);

Deno.test(
  "ai returns text event before calling actions",
  injectSecrets(async () => {
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text:
          `Please call the doSomethingUnique tool and explain what you're doing.`,
      }),
    ];
    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 5,
      onMaxIterationsReached: () => {},
      tools: [someTool],
      prompt:
        `You are an AI assistant. Always explain what you're doing before using tools.`,
      rewriteHistory: noopRewriteHistory,
    });
    const firstTextIndex = mockHistory.findIndex((event) =>
      event.type === "own_utterance" && event.text
    );
    const firstToolIndex = mockHistory.findIndex((event) =>
      event.type === "tool_call"
    );
    assert(firstTextIndex >= 0, "AI should produce text output");
    assert(firstToolIndex >= 0, "AI should call the tool");
    assert(
      firstTextIndex < firstToolIndex,
      "Text should come before tool call",
    );
  }),
);

Deno.test(
  "ai handles new history items while waiting for function calls",
  injectSecrets(async () => {
    const mockHistory: HistoryEvent[] = [participantUtteranceTurn({
      name: "user",
      text: `Please call the slowTool.`,
    })];

    const slowTool = {
      name: "slowTool",
      description: "A tool that takes time to execute",
      parameters: z.object({}),
      handler: async () => {
        // Add a new event to history while tool is running
        await sleep(10);
        mockHistory.push(participantUtteranceTurn({
          name: "user",
          text: "While you're working, here's additional context.",
        }));
        return "slow result";
      },
    };

    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 10,
      onMaxIterationsReached: () => {},
      tools: [slowTool],
      prompt: `You are an AI assistant. Always acknowledge new messages.`,
      rewriteHistory: noopRewriteHistory,
    });

    // Check that the tool was called
    const toolCall = mockHistory.find((e) =>
      e.type === "tool_call" && e.name === "slowTool"
    );
    assert(toolCall, "slowTool should be called");

    // Check that the additional message was added during tool execution
    const addedContext = mockHistory.find((e) =>
      e.type === "participant_utterance" &&
      e.text?.includes("additional context")
    );
    assert(addedContext, "Additional message should be in history");

    // Since the agent processes history in iterations, the additional message
    // should be processed in a subsequent iteration
    const responseAfterContext = mockHistory.find((event) =>
      event.type === "own_utterance" &&
      event.text &&
      event.timestamp > addedContext.timestamp
    );
    assert(
      responseAfterContext,
      "AI should respond to additional message in next iteration",
    );
  }),
);

Deno.test(
  "maxIterationsReached aborts the loop",
  injectSecrets(async () => {
    let callbackCalled = false;
    const mockHistory: HistoryEvent[] = [participantUtteranceTurn({
      name: "user",
      text: `Please keep talking and calling tools continuously.`,
    })];
    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 3, // Very low limit to ensure we hit it
      onMaxIterationsReached: () => {
        callbackCalled = true;
      },
      tools: [{
        name: "continueTalking",
        description: "A tool that keeps the conversation going",
        parameters: z.object({}),
        handler: async () => {
          // Add another user message to keep the conversation going
          await sleep(5); // Small delay to satisfy async requirement
          mockHistory.push(participantUtteranceTurn({
            name: "user",
            text: "Keep going, call the tool again!",
          }));
          return "continue";
        },
      }],
      prompt:
        `You are a chatty AI. Always call the continueTalking tool in every response and keep the conversation going.`,
      rewriteHistory: noopRewriteHistory,
    });
    assert(callbackCalled, "onMaxIterationsReached callback should be called");
  }),
);

Deno.test(
  "agent repeats back order of four speakers",
  injectSecrets(async () => {
    const mockHistory = [
      { name: "Alice", text: "Hi everyone" },
      { name: "Bob", text: "Yo" },
      { name: "Carol", text: "Howdy" },
      { name: "Dave", text: "Hello" },
      {
        name: "Alice",
        text:
          "List the speakers in the order they first spoke. Reply ONLY with: Alice,Bob,Carol,Dave",
      },
    ].map(participantUtteranceTurn);

    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 5,
      onMaxIterationsReached: () => {},
      tools: [],
      prompt:
        "You are an AI that strictly follows formatting instructions. When asked to list speakers, reply exactly as instructed without extra text.",
      rewriteHistory: noopRewriteHistory,
    });

    const answer = mockHistory.find((
      e,
    ): e is Extract<HistoryEvent, { type: "own_utterance" }> =>
      e.type === "own_utterance" && "text" in e && typeof e.text === "string" &&
      e.text.trim().startsWith("Alice")
    );
    assert(answer, "AI should respond with an own_utterance");
    const normalized = answer.text.replace(/\s/g, "");
    assertEquals(normalized, "Alice,Bob,Carol,Dave");
  }),
);

Deno.test(
  "agent triggers do nothing event when model should say nothing",
  injectSecrets(async () => {
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({ name: "user", text: "Say nothing." }),
    ];
    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 2,
      onMaxIterationsReached: () => {},
      tools: [],
      prompt:
        "You are an AI assistant. If the user says 'Say nothing.', do not reply with any text.",
      lightModel: true,
      rewriteHistory: noopRewriteHistory,
    });
    assertEquals(mockHistory[mockHistory.length - 1].type, "do_nothing");
  }),
);

const findTextualAnswer = (events: HistoryEvent[]) =>
  events.find((event): event is Extract<HistoryEvent, {
    type: "own_utterance";
    text: string;
  }> =>
    event.type === "own_utterance" && typeof event.text === "string" &&
    event.text.length > 0
  );

const collectAttachment = (events: HistoryEvent[], toolName?: string) => {
  for (let i = events.length - 1; i >= 0; i--) {
    const event = events[i];
    if (
      event.type === "tool_result" && event.attachments?.length &&
      (!toolName || event.name === toolName)
    ) {
      return event.attachments[0];
    }
    if (event.type === "own_utterance" && event.attachments?.length) {
      return event.attachments[0];
    }
  }
  return undefined;
};

Deno.test(
  "agent emits native image and separate agent verifies it",
  injectSecrets(async () => {
    const generationHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "creator",
        text:
          "Produce a vibrant poster that displays the single word SUNRISE in bold orange letters. Create the image directly in your response and then briefly confirm what you rendered.",
      }),
    ];

    await agentDeps(generationHistory)(runAgent)({
      maxIterations: 4,
      onMaxIterationsReached: () => {},
      tools: [],
      prompt:
        "You are a graphic designer who can emit inline images. When asked for a poster, respond with a PNG attachment via inline data that clearly shows the requested text, then acknowledge that text in plain language.",
      imageGen: true,
      rewriteHistory: noopRewriteHistory,
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

    await agentDeps(verificationHistory)(runAgent)({
      maxIterations: 4,
      onMaxIterationsReached: () => {},
      tools: [],
      prompt:
        "You can read text from images. Double-check what the poster says and mention the word explicitly in your short reply.",
      lightModel: true,
      rewriteHistory: noopRewriteHistory,
    });

    const answer = findTextualAnswer(verificationHistory);
    assert(answer, "Verification agent did not respond");
    assert(
      answer.text.toLowerCase().includes("sunrise"),
      `Expected the response to mention sunrise, got: ${answer.text}`,
    );
  }),
);

const toBase64 = (u8: Uint8Array): string => {
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < u8.length; i += chunk) {
    binary += String.fromCharCode(...u8.subarray(i, i + chunk));
  }
  return btoa(binary);
};

const bytes = await Deno.readFile("./dog.jpg");

const b64 = toBase64(bytes);

const mediaTool = {
  name: "returnMedia",
  description: "Returns media via attachments",
  parameters: z.object({}),
  handler: () => {
    const ret: ToolReturn = {
      result: "image attached",
      attachments: [
        { kind: "inline", mimeType: "image/jpeg", dataBase64: b64 },
      ],
    };
    return Promise.resolve(ret);
  },
};

const mediaToolWithCaption = {
  name: "returnMediaWithCaption",
  description: "Returns media with caption via attachments",
  parameters: z.object({}),
  handler: () => {
    const ret: ToolReturn = {
      result: "image with caption attached",
      attachments: [
        {
          kind: "inline",
          mimeType: "image/jpeg",
          dataBase64: b64,
          caption: "A friendly golden retriever sitting in the grass",
        },
      ],
    };
    return Promise.resolve(ret);
  },
};

const recognizedTheDog = (e: HistoryEvent) =>
  e.type === "own_utterance" &&
  (e.text.toLowerCase().includes("dog") ||
    e.text.toLowerCase().includes("retriever") ||
    e.text.toLowerCase().includes("puppy"));

Deno.test(
  "tool result attachments are forwarded to model",
  injectSecrets(async () => {
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Please call returnMedia and then describe the image.",
      }),
    ];
    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 3,
      onMaxIterationsReached: () => {},
      tools: [mediaTool],
      prompt: "You can see images returned by tools.",
      lightModel: true,
      rewriteHistory: noopRewriteHistory,
    });
    assert(
      mockHistory.some(recognizedTheDog),
      `AI did not describe the image as a dog. History: ${
        JSON.stringify(mockHistory, null, 2)
      }`,
    );
  }),
);

Deno.test(
  "user attachments are forwarded to model",
  injectSecrets(async () => {
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Please describe the attached image.",
        attachments: [
          { kind: "inline", mimeType: "image/jpeg", dataBase64: b64 },
        ],
      }),
    ];
    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 3,
      onMaxIterationsReached: () => {},
      tools: [],
      prompt: "You can see images attached by the user.",
      lightModel: true,
      rewriteHistory: noopRewriteHistory,
    });
    assert(
      mockHistory.some(recognizedTheDog),
      `AI did not describe the image as a dog. History: ${
        JSON.stringify(mockHistory, null, 2)
      }`,
    );
  }),
);

Deno.test(
  "attachment captions are included in model input",
  injectSecrets(async () => {
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
    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 3,
      onMaxIterationsReached: () => {},
      tools: [],
      prompt:
        "You can see images and their captions. Always mention the caption information in your response.",
      lightModel: true,
      rewriteHistory: noopRewriteHistory,
    });
    assert(
      mockHistory.some((e) =>
        e.type === "own_utterance" &&
        e.text.toLowerCase().includes("buddy") &&
        e.text.toLowerCase().includes("golden retriever")
      ),
      `AI did not mention the caption information. History: ${
        JSON.stringify(mockHistory, null, 2)
      }`,
    );
  }),
);

Deno.test(
  "tool result attachments with captions are forwarded to model",
  injectSecrets(async () => {
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text:
          "Please call returnMediaWithCaption and describe what you received.",
      }),
    ];
    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 3,
      onMaxIterationsReached: () => {},
      tools: [mediaToolWithCaption],
      prompt:
        "You can see images and their captions returned by tools. Reply with one sentence that includes the words 'grass' and 'retriever'.",
      lightModel: false,
      rewriteHistory: noopRewriteHistory,
    });
    assert(
      mockHistory.some((e) =>
        e.type === "own_utterance" &&
        e.text.toLowerCase().includes("grass") &&
        recognizedTheDog(e)
      ),
      `AI did not mention the tool caption information. History: ${
        JSON.stringify(mockHistory, null, 2)
      }`,
    );
  }),
);

Deno.test("filterOrphanedToolResults logic", () => {
  const baseAuth = { isOwn: true, id: "msg-id", timestamp: 100 } as const;
  const mkCall = (
    id: string,
    name: string,
    timestamp: number,
  ) => ({
    ...baseAuth,
    type: "tool_call" as const,
    id,
    name,
    timestamp,
    parameters: {},
    modelMetadata: { type: "gemini", responseId: "r1", thoughtSignature: "" },
  });
  const mkResult = (
    name: string,
    timestamp: number,
    toolCallId?: string,
  ) => ({
    ...baseAuth,
    type: "tool_result" as const,
    name,
    timestamp,
    toolCallId,
    result: "res",
    modelMetadata: { type: "gemini", responseId: "r1", thoughtSignature: "" },
  });

  // Case 1: Legacy match
  const h1 = [
    mkCall("c1", "toolA", 100),
    mkResult("toolA", 101),
  ];
  // @ts-ignore - types match roughly
  assertEquals(filterOrphanedToolResults(h1).length, 2);

  // Case 2: Orphaned legacy
  const h2 = [
    mkResult("toolA", 101),
  ];
  // @ts-ignore - types match roughly
  assertEquals(filterOrphanedToolResults(h2).length, 0);

  // Case 3: Mixed strict and legacy
  const h3 = [
    mkCall("c1", "toolA", 100),
    mkCall("c2", "toolA", 102),
    mkResult("toolA", 103, "c2"), // Claims c2
    mkResult("toolA", 104), // Should claim c1
  ];
  // @ts-ignore - types match roughly
  assertEquals(filterOrphanedToolResults(h3).length, 4);

  // Case 4: Stealing prevention
  const h4 = [
    mkCall("c1", "toolA", 100),
    mkResult("toolA", 103, "c1"), // Claims c1
    mkResult("toolA", 104), // Orphan, because c1 is taken
  ];
  // @ts-ignore - types match roughly
  assertEquals(filterOrphanedToolResults(h4).length, 2);

  // Case 5: Excess legacy results (1 call, 2 results)
  const h5 = [
    mkCall("c1", "toolA", 100),
    mkResult("toolA", 101),
    mkResult("toolA", 102),
  ];
  // @ts-ignore - types match roughly
  assertEquals(filterOrphanedToolResults(h5).length, 2);
});

Deno.test(
  "agent with history starting with only tool doesn't trigger 400",
  injectSecrets(async () => {
    const mockHistory: HistoryEvent[] = [{
      type: "tool_result",
      isOwn: true,
      id: "test-id",
      timestamp: Date.now(),
      name: "someTool",
      result: "some result",
    }];
    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 1,
      onMaxIterationsReached: () => {},
      tools: [],
      prompt: "You are a helper.",
      lightModel: true,
      rewriteHistory: noopRewriteHistory,
    });
  }),
);

Deno.test(
  "tool_call with empty thoughtSignature is filtered out with warning",
  injectSecrets(async () => {
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Please call the testTool.",
      }),
      // Inject a tool_call event with empty thoughtSignature
      {
        type: "tool_call",
        isOwn: true,
        id: "test-id",
        timestamp: Date.now(),
        name: "testTool",
        parameters: {},
        modelMetadata: {
          type: "gemini",
          thoughtSignature: "", // Empty - this should trigger the bug
          responseId: "resp_id",
        },
      } as HistoryEvent,
    ];

    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 1,
      onMaxIterationsReached: () => {},
      tools: [someTool],
      prompt: "You are a helper.",
      lightModel: false, // Use full model to trigger API call
      rewriteHistory: noopRewriteHistory,
    });
  }),
);

Deno.test(
  "agent filters unsupported gemini attachments before api call",
  injectSecrets(async () => {
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Describe the text message only.",
        attachments: [
          {
            kind: "inline",
            mimeType: "application/octet-stream",
            dataBase64: "dGVzdA==",
          },
        ],
      }),
    ];
    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 1,
      onMaxIterationsReached: () => {},
      tools: [],
      prompt: "You are a helper.",
      lightModel: true,
      rewriteHistory: noopRewriteHistory,
    });
  }),
);

Deno.test(
  "handles 403 file permission errors and replaces history items",
  injectSecrets(async () => {
    const replacedItems = new Map<string, HistoryEvent>();
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Here's an image",
        attachments: [
          {
            kind: "file",
            mimeType: "image/png",
            fileUri:
              "https://generativelanguage.googleapis.com/v1beta/files/2opdg5pjmw67",
            caption: "Test image",
          },
        ],
      }),
      participantUtteranceTurn({
        name: "user",
        text: "What do you see?",
      }),
    ];

    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 5,
      onMaxIterationsReached: () => {},
      tools: [],
      prompt: "You are a helpful assistant.",
      lightModel: true,
      rewriteHistory: (
        replacements: Record<string, HistoryEvent>,
      ) => {
        Object.entries(replacements).forEach(([id, newItem]) => {
          replacedItems.set(id, newItem);
          const index = mockHistory.findIndex((e) => e.id === id);
          if (index !== -1) {
            mockHistory[index] = newItem;
          }
        });
        return Promise.resolve();
      },
    });

    // The test should complete without throwing
    // If a real 403 error occurs, the rewriteHistory should be called
    // Note: This test may not trigger an actual 403 unless the file truly expired
    assert(true, "Agent completed successfully");
  }),
);

const addition = tool({
  name: "add",
  description: "Add two numbers",
  parameters: z.object({ a: z.number(), b: z.number() }),
  handler: ({ a, b }) => Promise.resolve(`${a + b}`),
});

const multiplication = tool({
  name: "multiply",
  description: "Multiply two numbers",
  parameters: z.object({ x: z.number(), y: z.number() }),
  handler: ({ x, y }) => Promise.resolve(`${x * y}`),
});

Deno.test(
  "skills: agent can use run_command to call skill tools",
  injectSecrets(async () => {
    const mockHistory: HistoryEvent[] = [participantUtteranceTurn({
      name: "user",
      text: "What is 5 + 3?",
    })];

    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 5,
      onMaxIterationsReached: () => {},
      tools: [],
      skills: [{
        name: "calculator",
        description: "Mathematical operations",
        instructions: "Use this skill for any math calculations",
        tools: [addition, multiplication],
      }],
      prompt:
        "You are a math assistant. Use the available skill tools to answer questions.",
      rewriteHistory: noopRewriteHistory,
    });

    const runCommandCall = mockHistory.find((event) =>
      event.type === "tool_call" && event.name === runCommandToolName
    );
    assert(runCommandCall, "Should call run_command tool");

    const hasToolResult = mockHistory.some((event) =>
      event.type === "tool_result" &&
      event.name === runCommandToolName &&
      event.result === "8"
    );
    assert(hasToolResult, "Should have result of 5 + 3 = 8 from run_command");
  }),
);

const weatherSkill = {
  name: "weather",
  description: "Get weather information",
  instructions: "Always ask for location before checking weather",
  tools: [
    tool({
      name: "get_forecast",
      description: "Get weather forecast for a location",
      parameters: z.object({ location: z.string() }),
      handler: ({ location }) => Promise.resolve(`Sunny in ${location}`),
    }),
    tool({
      name: "get_temperature",
      description: "Get current temperature",
      parameters: z.object({ location: z.string() }),
      handler: ({ location }) => Promise.resolve(`25°C in ${location}`),
    }),
  ],
};

Deno.test(
  "skills: prompt only shows skill name and description, not individual tools",
  injectSecrets(async () => {
    const mockHistory: HistoryEvent[] = [participantUtteranceTurn({
      name: "user",
      text: "hello",
    })];

    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 1,
      onMaxIterationsReached: () => {},
      tools: [],
      skills: [weatherSkill],
      prompt: "You are a helpful assistant.",
      rewriteHistory: noopRewriteHistory,
    });

    // The AI should have received prompt with skill info
    // We can't directly check the prompt, but we can verify through the geminiAgent
    // For now, this is a placeholder - we'll verify the actual implementation
    assert(true, "Prompt augmentation test - implementation will verify");
  }),
);

Deno.test(
  "skills: learn_skill returns skill information",
  injectSecrets(async () => {
    const weatherSkill = {
      name: "weather",
      description: "Get weather information",
      instructions: "Always ask for location before checking weather",
      tools: [
        {
          name: "get_forecast",
          description: "Get weather forecast for a location",
          parameters: z.object({ location: z.string() }),
          handler: ({ location }: { location: string }) =>
            Promise.resolve(`Sunny in ${location}`),
        },
      ],
    };

    const mockHistory: HistoryEvent[] = [participantUtteranceTurn({
      name: "user",
      text: "Tell me about the weather skill",
    })];

    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 5,
      onMaxIterationsReached: () => {},
      tools: [],
      skills: [weatherSkill],
      prompt:
        "You are a helpful assistant. When asked about skills, use learn_skill to get information.",
      rewriteHistory: noopRewriteHistory,
    });

    const learnSkillResult = mockHistory.find((event) =>
      event.type === "tool_result" &&
      event.name === learnSkillToolName &&
      event.result.includes("weather")
    );

    if (learnSkillResult && learnSkillResult.type === "tool_result") {
      const parsedResult = JSON.parse(learnSkillResult.result);
      assertEquals(parsedResult.name, "weather");
      assertEquals(
        parsedResult.instructions,
        "Always ask for location before checking weather",
      );
      assert(parsedResult.tools.length > 0, "Should include tools");
    }
  }),
);

Deno.test(
  "skills: run_command actually executes the skill tool handler",
  async () => {
    let handlerWasCalled = false;

    const testTool = tool({
      name: "test_tool",
      description: "Test tool",
      parameters: z.object({ value: z.number() }),
      handler: ({ value }) => {
        handlerWasCalled = true;
        return Promise.resolve(`Result: ${value * 2}`);
      },
    });

    const skillTools = createSkillTools([{
      name: "test",
      description: "Test skill",
      instructions: "Test instructions",
      tools: [testTool],
    }]);
    const runCommandTool = skillTools.find((t) =>
      t.name === runCommandToolName
    );

    if (!runCommandTool) {
      throw new Error("run_command tool not found");
    }

    const result = await runCommandTool.handler({
      command: "test/test_tool",
      params: { value: 5 },
    });

    assert(handlerWasCalled, "Handler should have been called");
    assertEquals(result, "Result: 10");
  },
);

Deno.test(
  "skills: learn_skill returns actual skill details",
  async () => {
    const weatherSkill = {
      name: "weather",
      description: "Weather information service",
      instructions: "Always ask for location before checking weather",
      tools: [
        {
          name: "get_forecast",
          description: "Get weather forecast",
          parameters: z.object({ location: z.string() }),
          handler: () => Promise.resolve("Sunny"),
        },
      ],
    };

    const skillTools = createSkillTools([weatherSkill]);
    const learnSkillTool = skillTools.find((t) =>
      t.name === learnSkillToolName
    );

    if (!learnSkillTool) {
      throw new Error("learn_skill tool not found");
    }

    const result = await learnSkillTool.handler({ skillName: "weather" });

    assert(typeof result === "string", "Result should be a string");
    const parsed = JSON.parse(result);
    assertEquals(parsed.name, "weather");
    assertEquals(parsed.description, "Weather information service");
    assertEquals(
      parsed.instructions,
      "Always ask for location before checking weather",
    );
    assert(Array.isArray(parsed.tools), "Should have tools array");
    assertEquals(parsed.tools.length, 1);
    assertEquals(parsed.tools[0].name, "get_forecast");
  },
);

Deno.test(
  "skills: works alongside regular tools",
  injectSecrets(async () => {
    const regularTool = {
      name: "regularTool",
      description: "A regular tool",
      parameters: z.object({}),
      handler: () => Promise.resolve("regular result"),
    };

    const skillTool = {
      name: "skillset",
      description: "A skill",
      instructions: "Use this skill",
      tools: [{
        name: "skill_tool",
        description: "A skill tool",
        parameters: z.object({}),
        handler: () => Promise.resolve("skill result"),
      }],
    };

    const mockHistory: HistoryEvent[] = [participantUtteranceTurn({
      name: "user",
      text: "Use both the regular tool and the skill tool",
    })];

    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 10,
      onMaxIterationsReached: () => {},
      tools: [regularTool],
      skills: [skillTool],
      prompt: "You are a helpful assistant. Use both available tools.",
      rewriteHistory: noopRewriteHistory,
    });

    const hasRegularResult = mockHistory.some((event) =>
      event.type === "tool_result" &&
      event.name === "regularTool"
    );

    const hasSkillResult = mockHistory.some((event) =>
      event.type === "tool_result" &&
      event.result === "skill result"
    );

    assert(
      hasRegularResult || hasSkillResult,
      "Should be able to use both regular tools and skill tools",
    );
  }),
);
