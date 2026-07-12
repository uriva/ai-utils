import { assert } from "@std/assert";
import { checkHallucination } from "../mod.ts";
import type { HistoryEvent } from "../src/agent.ts";
import { injectSecrets } from "../test_helpers.ts";
import { z } from "zod/v4";

Deno.test({
  name:
    "hallucination checker does not flag tool_result-sourced information as hallucination",
  fn: injectSecrets(async () => {
    const prompt = "You are a helpful assistant that helps users find events.";
    const history: HistoryEvent[] = [
      {
        type: "participant_utterance",
        isOwn: false,
        name: "user",
        text: "Any screenings in Tel Aviv this week?",
        attachments: [],
        id: "a1",
        timestamp: 1742634033000,
      },
      {
        type: "tool_call",
        id: "tc1",
        timestamp: 1742634040000,
        isOwn: true,
        name: "event_discovery/query",
        parameters: { query: "screenings in Tel Aviv this week" },
      },
      {
        id: "tr1",
        timestamp: 1742634041000,
        type: "tool_result",
        isOwn: true,
        result: JSON.stringify({
          events: [
            {
              name: "Movie Night – Pretty Woman",
              price: "175₪",
              venue: "Clayla",
              link: "https://example.com/pretty-woman",
            },
          ],
        }),
        toolCallId: "tc1",
      },
      {
        type: "own_utterance",
        isOwn: true,
        text:
          "There is Movie Night – Pretty Woman for 175₪ at Clayla! Details: https://example.com/pretty-woman",
        id: "o1",
        timestamp: 1742634045000,
      },
    ];

    const result = await checkHallucination(history, {
      prompt,
      tools: [],
      skills: [],
    });

    assert(
      !result.isHallucinating,
      `False positive: checker flagged tool_result data as hallucination. Explanation: ${result.explanation}`,
    );
  }),
});

Deno.test({
  name:
    "hallucination checker does not flag own_thought-injected context as hallucination",
  fn: injectSecrets(async () => {
    const prompt = "You are a personal assistant.";
    const history: HistoryEvent[] = [
      {
        type: "own_thought",
        isOwn: true,
        text:
          "PROACTIVE TASK: Remind the user about their dentist appointment tomorrow at 10:00 with Dr. Sharon Levy. Copay is 30₪.",
        id: "ot1",
        timestamp: 1742634020000,
      },
      {
        type: "participant_utterance",
        isOwn: false,
        name: "user",
        text: "Hi",
        attachments: [],
        id: "a2",
        timestamp: 1742634033000,
      },
      {
        type: "own_utterance",
        isOwn: true,
        text:
          "Hi! Just wanted to remind you that you have a dentist appointment tomorrow at 10:00 with Dr. Sharon Levy. The copay is 30₪.",
        id: "o2",
        timestamp: 1742634035000,
      },
    ];

    const result = await checkHallucination(history, {
      prompt,
      tools: [],
      skills: [],
    });

    assert(
      !result.isHallucinating,
      `False positive: checker flagged injected context data as hallucination. Explanation: ${result.explanation}`,
    );
  }),
});

Deno.test({
  name:
    "hallucination checker does not flag external_event-sourced facts as hallucination",
  fn: injectSecrets(async () => {
    const prompt = "You are a helpful coding assistant.";
    const history: HistoryEvent[] = [
      {
        type: "participant_utterance",
        isOwn: false,
        name: "user",
        text: "did the deploy work?",
        attachments: [],
        id: "ee-user",
        timestamp: 1742634033000,
      },
      {
        type: "external_event",
        isOwn: false,
        text:
          "A background VM command finished with exit code 0. The deploy to tasks-dashboard.deno.dev succeeded and is live.",
        id: "ee1",
        timestamp: 1742634040000,
      },
      {
        type: "own_utterance",
        isOwn: true,
        text:
          "Yes! The deploy finished successfully (exit code 0) and your app is live at tasks-dashboard.deno.dev.",
        id: "o3",
        timestamp: 1742634045000,
      },
    ];

    const result = await checkHallucination(history, {
      prompt,
      tools: [],
      skills: [],
    });

    assert(
      !result.isHallucinating,
      `False positive: checker flagged external_event data as hallucination. Explanation: ${result.explanation}`,
    );
  }),
});

Deno.test({
  name:
    "hallucination checker does not flag prompt-supported community group links as hallucination",
  fn: injectSecrets(async () => {
    const prompt =
      "You are an AI builder assistant. Link to our discord: https://discord.gg/bcjxhRfARJ. Link to our English whatsapp group: https://chat.whatsapp.com/GTZNL5MgrYTL0jK2ZCimwY";
    const history: HistoryEvent[] = [
      {
        type: "participant_utterance",
        isOwn: false,
        name: "user",
        text: "Please build Nonibot for my store",
        attachments: [],
        id: "h1",
        timestamp: 1742634033000,
      },
      {
        type: "tool_call",
        id: "tc1",
        timestamp: 1742634040000,
        isOwn: true,
        name: "create_bot",
        parameters: {
          name: "Nonibot",
          prompt: "You are a customer assistant",
        },
      },
      {
        id: "tr1",
        timestamp: 1742634041000,
        type: "tool_result",
        isOwn: true,
        result: JSON.stringify({
          success: true,
          botId: "nonibot-id-123",
          chatLink: "https://aliceandbot.com/chat/nonibot",
          whatsappDemoLink: "https://wa.me/prompt2bot_demo?start=nonibot",
          telegramDemoLink: "https://t.me/prompt2bot_demo_bot?start=nonibot",
        }),
        toolCallId: "tc1",
      },
      {
        type: "own_utterance",
        isOwn: true,
        text:
          "Nonibot has been successfully created! 🎉\n\nIf you need help, feel free to join our English WhatsApp Group (https://chat.whatsapp.com/GTZNL5MgrYTL0jK2ZCimwY) or our Discord Community (https://discord.gg/bcjxhRfARJ).",
        id: "o4",
        timestamp: 1742634045000,
      },
    ];

    const createBotTool = {
      name: "create_bot",
      description: "Creates a new custom AI bot.",
      parameters: z.object({
        name: z.string(),
        prompt: z.string(),
      }),
      handler: () => Promise.resolve("success"),
    };

    const result = await checkHallucination(history, {
      prompt,
      tools: [createBotTool],
      skills: [],
    });

    assert(
      !result.isHallucinating,
      `False positive: checker flagged community group links as hallucination. Explanation: ${result.explanation}`,
    );
  }),
});
