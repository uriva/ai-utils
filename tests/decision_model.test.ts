import { assertEquals } from "@std/assert";
import {
  callDecisionModel,
  decide,
  decideWithReasoningOnPositive,
  type DecisionProvider,
  formatAgentStateForDecisionModel,
  genDecision,
  genJson,
  injectCacher,
  injectDecisionProvider,
  injectGeminiToken,
  injectJevToken,
  injectRespanToken,
  injectRespanUrl,
  z,
} from "../mod.ts";
import { routeThinkingLevelWithRespan } from "../src/respan.ts";
import { injectSecrets } from "../test_helpers.ts";

const sampleInquiry =
  "I cannot log into my account. The password reset link gives a 404 error.";

const classificationPrompt =
  "Classify the customer support request and determine urgency.";

const mixedPrompt =
  "Analyze the issue, categorize it, determine urgency, and summarize in one sentence.";

const decisionProviders: DecisionProvider[] = [
  ...(Deno.env.get("RESPAN_API_KEY") ? ["respan" as const] : []),
  ...(Deno.env.get("JEV_API_KEY") ? ["jev" as const] : []),
];

for (const provider of decisionProviders) {
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
      });
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
      });
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
      });
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
      });
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
      });
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
      });
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

        // Positive case: urgent login issue
        const positiveResult = await check(sampleInquiry);
        assertEquals(positiveResult.isUrgent, true);
        assertEquals(typeof positiveResult.explanation, "string");
        assertEquals(positiveResult.explanation.length > 0, true);

        // Negative case: non-urgent casual question
        const negativeResult = await check("What are your business hours?");
        assertEquals(negativeResult.isUrgent, false);
        assertEquals(negativeResult.explanation, "");
      });
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
      });
    }),
  );
}

Deno.test(
  "genJson falls back to LLM when decision token is empty",
  injectSecrets(async () => {
    await injectRespanToken("")(async () => {
      await injectJevToken("")(async () => {
        await injectGeminiToken(Deno.env.get("GEMINI_API_KEY")!)(async () => {
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
        });
      });
    });
  }),
);

Deno.test(
  "Respan uses span-02 by default and falls back to span-01-free and span-01-pro on 429/402 quota exhaustion",
  async () => {
    const requestedModels: string[] = [];
    const server = Deno.serve({ port: 0 }, async (req) => {
      const body = await req.json();
      requestedModels.push(body.model);
      if (body.model === "span-02") {
        return new Response("Credits depleted", { status: 402 });
      }
      if (body.model === "span-01-free") {
        return new Response("Daily free quota exceeded", { status: 429 });
      }
      return new Response(
        JSON.stringify({
          model: "span-01-pro",
          results: [
            {
              id: "urgency",
              p_present: 0.95,
              p_absent: 0.05,
              p_not_observable: 0.0,
            },
          ],
        }),
        { headers: { "Content-Type": "application/json" } },
      );
    });

    const addr = server.addr;
    const url = `http://localhost:${addr.port}`;

    await injectRespanUrl(url)(async () => {
      await injectRespanToken("fake-token")(async () => {
        await injectDecisionProvider("respan")(async () => {
          const answers = await callDecisionModel(
            `unique-state-${Date.now()}`,
            {
              urgency: {
                type: "noul",
                instructions: "Is this urgent?",
              },
            },
          );
          assertEquals(answers.urgency.type, "noul");
          assertEquals(
            answers.urgency.type === "noul" && answers.urgency.noul > 0.9,
            true,
          );
          assertEquals(requestedModels, [
            "span-02",
            "span-01-free",
            "span-01-pro",
          ]);
        });
      });
    });

    await server.shutdown();
  },
);

Deno.test(
  "formatAgentStateForDecisionModel and routeThinkingLevelWithRespan handle emojis split at UTF-16 slice boundaries without lone surrogates",
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

    const level = await routeThinkingLevelWithRespan(state);
    assertEquals(level === "LOW" || level === "HIGH", true);
  }),
);

Deno.test(
  "routeThinkingLevelWithRespan routes caching through injected makeCache rather than bypassing injectCacher",
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
          model: "span-02",
          answers: {
            thinking_level: {
              type: "choice",
              choice: "low",
              confidence: 0.98,
              probabilities: { low: 0.99, high: 0.01 },
            },
          },
        }),
        { headers: { "Content-Type": "application/json" } },
      );
    });

    const url = `http://localhost:${server.addr.port}`;
    await injectCacher(customCacher)(
      injectRespanUrl(url)(
        injectRespanToken("fake-token")(async () => {
          const level = await routeThinkingLevelWithRespan(
            `unique-route-state-${Date.now()}-${Math.random()}`,
          );
          assertEquals(level, "LOW");
          assertEquals(seenCacheIds.includes("respan-thinking-level-v1"), true);
        }),
      ),
    )();

    await server.shutdown();
  },
);

Deno.test(
  "decide evaluates boolean schema with 'Whether ... is classified as ...' description and empty Jev token in pipe",
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

    await injectDecisionProvider("respan")(
      injectRespanToken(Deno.env.get("RESPAN_API_KEY")!)(
        injectJevToken("")(async () => {
          const spamRes = await checkSpam(spamMsg);
          assertEquals(spamRes.isSpam, true);

          const cleanRes = await checkSpam(cleanMsg);
          assertEquals(cleanRes.isSpam, false);
        }),
      ),
    )();
  }),
);
