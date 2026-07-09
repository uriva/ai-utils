import { assert, assertEquals, assertRejects } from "@std/assert";
import {
  assertNoScriptDrift,
  driftingScripts,
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
