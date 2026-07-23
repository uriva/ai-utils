import { assert, assertEquals, assertRejects } from "@std/assert";
import { z } from "zod/v4";
import {
  assertNoScriptDrift,
  driftingScripts,
  meaningfulScriptDrift,
  type ScriptDriftError,
  scriptsPresent,
} from "../src/scriptDriftGuard.ts";
import { injectSecrets, llmTest } from "../test_helpers.ts";

// Gemini sometimes rewrites text into a visually-similar but different writing
// system when echoing it back (e.g. Hebrew rendered as Armenian homoglyphs).
// These are synthetic samples: a Hebrew input and an all-Armenian output.
const hebrewInput = "שלום זהו טקסט בעברית בלבד אנא שמור על השפה";
const armenianCorruptedOutput = "Աեը սՁլՀ ՀեՀ եՁՁյե ееՁ ееՁՁ ееееՁ Հ ееՁ";

Deno.test("scriptsPresent detects Hebrew and Latin, ignores punctuation", () => {
  const present = scriptsPresent("שלום world 2900!");
  assert(present.has("Hebrew"));
  assertEquals(present.has("Armenian"), false);
});

Deno.test("driftingScripts flags Armenian absent from a Hebrew input", () => {
  const drift = driftingScripts(hebrewInput, armenianCorruptedOutput);
  assert("Armenian" in drift, "expected Armenian to be flagged as drift");
  assert(
    drift.Armenian > 5,
    `expected several Armenian chars, got ${drift.Armenian}`,
  );
});

Deno.test("driftingScripts is empty when output preserves input script", () => {
  const drift = driftingScripts(hebrewInput, "שלום זהו טקסט תקין בעברית");
  assertEquals(Object.keys(drift).length, 0);
});

// Regression: a real production message was a long, correct Hebrew reply with a
// single stray glyph from another script (a rare model glitch). The guard must
// not treat a one-character blip as corruption — it wastes a verifier LLM call
// and, historically, hard-failed an otherwise-valid reply. Genuine homoglyph
// corruption rewrites substantial spans, not a single character.
const hebrewReplyWithOneStrayGlyph =
  "היי ניצן! הכל מעולה, הנה כמה אירועים מומלצים להערב בתל אביב עם מוזיקה טובה ואווירה מ\u0E51צוינת, תהנה!";

Deno.test("single stray glyph is not counted as meaningful drift", () => {
  const drift = driftingScripts(hebrewInput, hebrewReplyWithOneStrayGlyph);
  // The raw detector still sees the one Thai char...
  assertEquals(drift.Thai, 1);
  // ...but meaningfulScriptDrift must ignore a trivial one-off blip.
  assertEquals(
    Object.keys(
      meaningfulScriptDrift(hebrewInput, hebrewReplyWithOneStrayGlyph),
    ).length,
    0,
  );
});

// The corrupted output (input's own language rewritten into another script)
// must be rejected.
llmTest(
  "assertNoScriptDrift throws on Hebrew->Armenian homoglyph corruption",
  () =>
    injectSecrets(async () => {
      const err = await assertRejects(
        () => assertNoScriptDrift(hebrewInput, armenianCorruptedOutput),
        Error,
        "Script drift detected",
      );
      assert(
        "scriptDrift" in (err as ScriptDriftError),
        "error should carry scriptDrift details",
      );
      assert(
        "Armenian" in (err as ScriptDriftError).scriptDrift,
        "scriptDrift should include Armenian",
      );
    })(),
);

// A legitimate, input-requested language switch must NOT throw.
llmTest(
  "assertNoScriptDrift allows a legitimately requested language switch",
  () =>
    injectSecrets(async () => {
      await assertNoScriptDrift(
        "Translate the following greeting into Hebrew and reply with only the Hebrew: Hello and welcome!",
        "שלום וברוכים הבאים",
      );
    })(),
);

// A single stray glyph must never reach the verifier nor throw. This does not
// hit the LLM at all (it short-circuits on the trivial-drift threshold), so it
// is a plain deterministic test rather than an llmTest.
Deno.test("assertNoScriptDrift ignores a single stray glyph", () =>
  assertNoScriptDrift(hebrewInput, hebrewReplyWithOneStrayGlyph));

llmTest(
  "assertNoScriptDrift allows script switch on long input where instructions/user query are past top 2000 chars",
  () =>
    injectSecrets(async () => {
      const genericSystemPadding = "System guidelines for agent execution. "
        .repeat(100);
      const longInput =
        `${genericSystemPadding}\n\nLanguage rule: Match the user's language. If they ask in Nepali or Romanized Nepali, reply in Nepali.\nUser query: Namaste, tapai ko kasto chha?`;
      await assertNoScriptDrift(
        longInput,
        "नमस्ते, म सञ्चै छु। तपाइँलाई कसरी सहयोग गर्न सक्छु?",
      );
    })(),
);

import { genJsonFromConvo } from "../src/genJson.ts";

llmTest(
  "genJsonFromConvo completes successfully on clean Hebrew input",
  () =>
    injectSecrets(async () => {
      const schema = z.object({ response: z.string() });
      const result = await genJsonFromConvo(
        { provider: "google", mini: true },
        [
          { role: "system", content: "ענה בעברית בלבד כ-JSON." },
          {
            role: "user",
            content: "שלום, מה שלומך?",
          },
        ],
        schema,
      );
      assert(typeof result.response === "string");
    })(),
);
