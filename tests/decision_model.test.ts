import { assertEquals } from "@std/assert";
import { pipe } from "gamla";
import {
  callDecisionModel,
  decide,
  decideWithReasoningOnPositive,
  type DecisionProvider,
  defaultGeminiDecisionModel,
  formatAgentStateForDecisionModel,
  geminiThinkingLevelCacheId,
  genDecision,
  genJson,
  injectCacher,
  injectDecisionProvider,
  injectGeminiDecisionToken,
  injectGeminiDecisionUrl,
  injectGeminiToken,
  injectJevToken,
  injectRespanToken,
  resolveDecisionProvider,
  routeThinkingLevelWithGemini,
  z,
} from "../mod.ts";
import { injectSecrets } from "../test_helpers.ts";

const sampleInquiry =
  "I cannot log into my account. The password reset link gives a 404 error.";

const classificationPrompt =
  "Classify the customer support request and determine urgency.";

const mixedPrompt =
  "Analyze the issue, categorize it, determine urgency, and summarize in one sentence.";

const decisionProviders: DecisionProvider[] = [
  ...(Deno.env.get("GEMINI_API_KEY") ? ["gemini" as const] : []),
];

Deno.test("resolveDecisionProvider defaults to gemini when Gemini token is available", () => {
  injectGeminiToken("test-gemini-key")(() => {
    injectRespanToken("test-respan-key")(() => {
      injectJevToken("test-jev-key")(() => {
        assertEquals(resolveDecisionProvider(), "gemini");
      })();
    })();
  })();
});

decisionProviders.forEach((provider) => {
  Deno.test(
    `callDecisionModel answers choice and noul questions [${provider}]`,
    injectSecrets(async () => {
      await injectDecisionProvider(provider)(async () => {
        const answers = await callDecisionModel(sampleInquiry, {
          urgency: {
            type: "noul",
            instructions: "Is this request urgent?",
          },
          department: {
            type: "choice",
            instructions: "Which department handles this?",
            criteria: {
              auth: "Login, authentication, password reset, or account access",
              billing: "Invoices, payments, and credit cards",
              sales: "Pricing inquiries and product demos",
            },
          },
        });

        assertEquals(answers.urgency.type, "noul");
        assertEquals(answers.department.type, "choice");
        assertEquals(
          answers.department.type === "choice" && answers.department.choice,
          "auth",
        );
      })();
    }),
  );

  Deno.test(
    `decide evaluates object schema with boolean and enum [${provider}]`,
    injectSecrets(async () => {
      await injectDecisionProvider(provider)(async () => {
        const SupportSchema = z.object({
          isUrgent: z.boolean().describe(
            "Is the user blocked or experiencing an urgent issue?",
          ),
          department: z.enum(["auth", "billing", "sales"]).describe(
            "Responsible department",
          ),
        });

        const classify = decide(classificationPrompt, SupportSchema);
        const result = await classify(sampleInquiry);

        assertEquals(result.isUrgent, true);
        assertEquals(result.department, "auth");
      })();
    }),
  );

  Deno.test(
    `genDecision is an alias for decide [${provider}]`,
    injectSecrets(async () => {
      await injectDecisionProvider(provider)(async () => {
        const FlagSchema = z.object({
          isLoginIssue: z.boolean().describe(
            "Is this a login or auth problem?",
          ),
        });

        const check = genDecision(classificationPrompt, FlagSchema);
        const result = await check(sampleInquiry);

        assertEquals(result.isLoginIssue, true);
      })();
    }),
  );

  Deno.test(
    `decide evaluates standalone boolean schema [${provider}]`,
    injectSecrets(async () => {
      await injectDecisionProvider(provider)(async () => {
        const isLoginIssue = decide(
          "Is this inquiry related to login credentials or authentication?",
          z.boolean(),
        );
        const result = await isLoginIssue(sampleInquiry);

        assertEquals(result, true);
      })();
    }),
  );

  Deno.test(
    `genJson with pure decision schema routes to decision model without strings [${provider}]`,
    injectSecrets(async () => {
      await injectDecisionProvider(provider)(async () => {
        const DecisionOnlySchema = z.object({
          isAccountLocked: z.boolean().describe(
            "Is the user unable to access their account?",
          ),
          department: z.enum(["auth", "billing"]).describe(
            "Responsible department",
          ),
        });

        const classify = genJson(
          { provider: "google", tier: "flash" },
          classificationPrompt,
          DecisionOnlySchema,
        );
        const result = await classify(sampleInquiry);

        assertEquals(result.isAccountLocked, true);
        assertEquals(result.department, "auth");
      })();
    }),
  );

  Deno.test(
    `genJson with mixed schema routes directly to generative model when string fields are present [${provider}]`,
    injectSecrets(async () => {
      await injectDecisionProvider(provider)(async () => {
        const MixedSchema = z.object({
          isUrgent: z.boolean().describe(
            "Is the user experiencing an urgent block?",
          ),
          department: z.enum(["auth", "billing", "sales"]).describe(
            "Responsible department",
          ),
          summary: z.string().describe(
            "One sentence summary of the customer problem",
          ),
        });

        const analyze = genJson(
          { provider: "google", tier: "flash" },
          mixedPrompt,
          MixedSchema,
        );
        const result = await analyze(sampleInquiry);

        assertEquals(result.isUrgent, true);
        assertEquals(result.department, "auth");
        assertEquals(typeof result.summary, "string");
        assertEquals(result.summary.length > 5, true);
      })();
    }),
  );

  Deno.test(
    `decideWithReasoningOnPositive generates reasoning only on positive decision [${provider}]`,
    injectSecrets(async () => {
      await injectDecisionProvider(provider)(async () => {
        const UrgencyReportSchema = z.object({
          isUrgent: z.boolean().describe(
            "Is the user experiencing an urgent block?",
          ),
          explanation: z.string().describe(
            "Explanation of why the issue is urgent, or empty if not urgent.",
          ),
        });

        const check = decideWithReasoningOnPositive(
          { provider: "google", tier: "flash" },
          "Determine if the user problem is urgent and explain why.",
          UrgencyReportSchema,
        );

        const positiveResult = await check(sampleInquiry);
        assertEquals(positiveResult.isUrgent, true);
        assertEquals(typeof positiveResult.explanation, "string");
        assertEquals(positiveResult.explanation.length > 0, true);

        const negativeResult = await check("What are your business hours?");
        assertEquals(negativeResult.isUrgent, false);
        assertEquals(negativeResult.explanation, "");
      })();
    }),
  );

  Deno.test(
    `decide evaluates assistant message when instructions target assistant [${provider}]`,
    injectSecrets(async () => {
      await injectDecisionProvider(provider)(async () => {
        const HonestySchema = z.object({
          disclosedFailure: z.boolean().describe(
            "Did the assistant disclose that saving failed?",
          ),
          claimedSuccess: z.boolean().describe(
            "Did the assistant claim that saving succeeded?",
          ),
        });

        const checkHonesty = decide(
          "Evaluate the assistant message to the user following an operation where saving items failed.",
          HonestySchema,
        );
        const result = await checkHonesty(
          "I attempted to save the items to the automation system, but every call failed with a Malformed JSON error.",
        );

        assertEquals(result.disclosedFailure, true);
        assertEquals(result.claimedSuccess, false);
      })();
    }),
  );
});

Deno.test(
  "genJson falls back to LLM when decision token is empty",
  injectSecrets(async () => {
    await pipe(
      injectGeminiDecisionToken(""),
      injectRespanToken(""),
      injectJevToken(""),
      injectGeminiToken(Deno.env.get("GEMINI_API_KEY") ?? ""),
    )(async () => {
      const FallbackSchema = z.object({
        isUrgent: z.boolean().describe(
          "Is the user experiencing an urgent block?",
        ),
        department: z.enum(["auth", "billing", "sales"]).describe(
          "Responsible department",
        ),
      });

      const classify = genJson(
        { provider: "google", tier: "flash" },
        classificationPrompt,
        FallbackSchema,
      );
      const result = await classify(sampleInquiry);

      assertEquals(typeof result.isUrgent, "boolean");
      assertEquals(result.department, "auth");
    })();
  }),
);

Deno.test(
  "Gemini Decisions adapter translates noul criteria, sanitizes question keys, and short-circuits single-literal choices",
  async () => {
    const capturedRequests: {
      apiKey: string | null;
      body: Record<string, unknown>;
    }[] = [];

    const server = Deno.serve({ port: 0 }, async (req) => {
      const body = await req.json();
      capturedRequests.push({
        apiKey: req.headers.get("x-goog-api-key"),
        body,
      });
      const questions = body.questions ?? {};
      const wireKeys = Object.keys(questions);
      const mappedCalendarKey = wireKeys.find((k) => k !== "urgency") ?? "q_1";
      return new Response(
        JSON.stringify({
          answers: {
            urgency: {
              type: "noul",
              noul: 0.94,
              confidence: 0.94,
            },
            [mappedCalendarKey]: {
              type: "noul",
              noul: 0.88,
              confidence: 0.88,
            },
          },
        }),
        { headers: { "Content-Type": "application/json" } },
      );
    });

    const url = `http://localhost:${server.addr.port}`;

    await pipe(
      injectGeminiDecisionUrl(url),
      injectGeminiDecisionToken("fake-gemini-key"),
      injectDecisionProvider("gemini"),
    )(async () => {
      const answers = await callDecisionModel(
        { message: `unique-state-${Date.now()}` },
        {
          urgency: {
            type: "noul",
            instructions: "Is this urgent?",
            criteria: {
              true: "Immediate outage",
              false: "General question",
            },
          },
          "Google Calendar.v2": {
            type: "noul",
            instructions: "Does this need Google Calendar?",
          },
          soleLiteral: {
            type: "choice",
            instructions: "Fixed version",
            criteria: { v1: "v1" },
          },
        },
      );

      assertEquals(answers.urgency.type, "noul");
      assertEquals(
        answers.urgency.type === "noul" && answers.urgency.noul,
        0.94,
      );
      assertEquals(
        answers["Google Calendar.v2"]?.type === "noul" &&
          answers["Google Calendar.v2"].noul,
        0.88,
      );
      assertEquals(
        answers.soleLiteral?.type === "choice" && answers.soleLiteral.choice,
        "v1",
      );
      assertEquals(capturedRequests.length, 1);
      assertEquals(capturedRequests[0].apiKey, "fake-gemini-key");
      assertEquals(capturedRequests[0].body.model, defaultGeminiDecisionModel);
      const sentQuestions = capturedRequests[0].body.questions as Record<
        string,
        Record<string, unknown>
      >;
      assertEquals("soleLiteral" in sentQuestions, false);
      assertEquals("Google Calendar.v2" in sentQuestions, false);
      assertEquals(sentQuestions.urgency.criteria, {
        true_description: "Immediate outage",
        false_description: "General question",
      });
    })();

    await server.shutdown();
  },
);

Deno.test(
  "formatAgentStateForDecisionModel and routeThinkingLevelWithGemini handle emojis split at UTF-16 slice boundaries without lone surrogates",
  injectSecrets(async () => {
    const promptWithBoundaryEmoji = `${"a".repeat(499)}🔥 trailing prompt text`;
    const turnWith300BoundaryEmoji = `${"b".repeat(299)}🔥 trailing turn text`;
    const triggerWith1000BoundaryEmoji = `${
      "c".repeat(999)
    }🔥 simple greeting hello`;

    const state = formatAgentStateForDecisionModel(
      promptWithBoundaryEmoji,
      [
        {
          id: "1",
          timestamp: 1000,
          type: "participant_utterance",
          isOwn: false,
          name: "User",
          text: turnWith300BoundaryEmoji,
        },
        {
          id: "2",
          timestamp: 2000,
          type: "participant_utterance",
          isOwn: false,
          name: "User",
          text: triggerWith1000BoundaryEmoji,
        },
      ],
      [],
    );

    const serialized = JSON.stringify(state);
    assertEquals(
      /\\u[dD][89a-fA-F][0-9a-fA-F]{2}/.test(serialized),
      false,
      "Serialized decision state must not contain lone UTF-16 surrogates",
    );

    const level = await routeThinkingLevelWithGemini(state);
    assertEquals(level === "LOW" || level === "HIGH", true);
  }),
);

Deno.test(
  "routeThinkingLevelWithGemini routes caching through injected makeCache rather than bypassing injectCacher",
  async () => {
    const seenCacheIds: string[] = [];
    const customCacher = (cacheId: string) =>
      ((fn: (...args: unknown[]) => Promise<unknown>) =>
      (...args: unknown[]) => {
        seenCacheIds.push(cacheId);
        return fn(...args);
      }) as never;

    const server = Deno.serve({ port: 0 }, async (req) => {
      await req.json();
      return new Response(
        JSON.stringify({
          answers: {
            thinking_level: {
              type: "choice",
              choice: "low",
              confidence: 0.99,
              probabilities: { low: 0.99, high: 0.01 },
            },
          },
        }),
        { headers: { "Content-Type": "application/json" } },
      );
    });

    const url = `http://localhost:${server.addr.port}`;
    await pipe(
      injectCacher(customCacher),
      injectGeminiDecisionUrl(url),
      injectGeminiDecisionToken("fake-token"),
    )(async () => {
      const level = await routeThinkingLevelWithGemini(
        `unique-route-state-${Date.now()}-${Math.random()}`,
      );
      assertEquals(level, "LOW");
      assertEquals(seenCacheIds.includes(geminiThinkingLevelCacheId), true);
    })();

    await server.shutdown();
  },
);

Deno.test(
  "decide evaluates boolean schema with 'Whether ... is classified as ...' description with Gemini Decision 1",
  injectSecrets(async () => {
    const moderatorPrompt = [
      "You are a group moderator agent.",
      "Your task is to analyze incoming group messages and determine if they are spam or violate the group's custom rules.",
      "CRITICAL: If custom group rules are provided under 'CRITICAL CUSTOM GROUP RULES FOR ALLOWED/DISALLOWED CONTENT', they strictly override any default rules. For example, if a custom rule explicitly allows publishing links to events or parties, you must classify event/party link messages as NOT spam (isSpam: false), even if they appear promotional or broadcasted. Conversely, if a custom rule forbids certain topics, languages, or message types (even without links), any message violating that custom rule MUST be classified as spam (isSpam: true). Custom admin rules always take absolute precedence over default spam criteria.",
      "A message is classified as spam if it contains promotional links, unsolicited advertisements, scam links, affiliate marketing, phishing, bulk broadcast advertising, or violates the group's custom rules. This includes unsolicited invitations to join other chat groups or channels (especially those about stock trading, investments, crypto, courses, or general community promotion).",
      "Ignore regular conversational messages, helpful resource shares, or official team links unless they violate a custom group rule.",
      "Be decisive and accurate. If it is clearly spam/advertisement/unsolicited promotion or violates a custom group rule, return isSpam: true. Otherwise, return isSpam: false.",
    ].join(" ");

    const ModerationSchema = z.object({
      isSpam: z.boolean().describe(
        "Whether the message is classified as spam, unsolicited promotion, scam, or a violation of the custom group rules.",
      ),
    });

    const spamMsg = [
      "Group Name: Developers Chat",
      "Group Description: General developer discussion",
      "Sender Name: Alice",
      "Sender ID: user-123",
      "Message: Join our VIP stock trading group for daily market signals and analysis! https://example.com/join-stocks",
    ].join("\n");

    const cleanMsg = [
      "Group Name: Developers Chat",
      "Group Description: General developer discussion",
      "Sender Name: Bob",
      "Sender ID: user-456",
      "Message: Bug reported.\ndebug-report: https://example.com/debug?id=123",
    ].join("\n");

    const checkSpam = decide(moderatorPrompt, ModerationSchema);

    await pipe(
      injectDecisionProvider("gemini"),
      injectJevToken(""),
      injectRespanToken(""),
    )(async () => {
      const spamRes = await checkSpam(spamMsg);
      assertEquals(spamRes.isSpam, true);

      const cleanRes = await checkSpam(cleanMsg);
      assertEquals(cleanRes.isSpam, false);
    })();
  }),
);
