import { assertEquals } from "@std/assert";
import {
  callDecisionModel,
  decide,
  genDecision,
  genJson,
  injectGeminiToken,
  injectJevToken,
  z,
} from "../mod.ts";
import { injectSecrets } from "../test_helpers.ts";

const sampleInquiry =
  "I cannot log into my account. The password reset link gives a 404 error.";

const classificationPrompt =
  "Classify the customer support request and determine urgency.";

const mixedPrompt =
  "Analyze the issue, categorize it, determine urgency, and summarize in one sentence.";

const testJevToken = Deno.env.get("JEV_API_KEY") ||
  "apikey_2323b4352d1a3ea4a0787a1689e16cabcd9_096712d3a92df92963ceec3b230db7e9551e027e997111bd989ace89f4716c75";

const withJev = injectJevToken(testJevToken);

Deno.test(
  "callDecisionModel answers choice and noul questions",
  injectSecrets(async () => {
    await withJev(async () => {
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
  "decide evaluates object schema with boolean and enum",
  injectSecrets(async () => {
    await withJev(async () => {
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
  "genDecision is an alias for decide",
  injectSecrets(async () => {
    await withJev(async () => {
      const FlagSchema = z.object({
        isLoginIssue: z.boolean().describe("Is this a login or auth problem?"),
      });

      const check = genDecision(classificationPrompt, FlagSchema);
      const result = await check(sampleInquiry);

      assertEquals(result.isLoginIssue, true);
    })();
  }),
);

Deno.test(
  "decide evaluates standalone boolean schema",
  injectSecrets(async () => {
    await withJev(async () => {
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
  "genJson with pure decision schema routes to decision model without strings",
  injectSecrets(async () => {
    await withJev(async () => {
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
  "genJson with mixed schema splits string fields and decision fields",
  injectSecrets(async () => {
    await withJev(async () => {
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
  "genJson falls back to LLM when Jev token is empty",
  injectSecrets(async () => {
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
      })();
    })();
  }),
);
