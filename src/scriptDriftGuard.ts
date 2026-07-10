import { genJson } from "./genJson.ts";
import { z } from "zod/v4";

// Gemini (and other LLMs) sometimes transliterate text into a different,
// visually-similar writing system when echoing it back — most notably turning
// Hebrew into Armenian homoglyphs inside structured JSON output. The output
// "looks" right at a glance but is a different script entirely, so downstream
// consumers (and end users) get garbage. This guard detects when the model
// output introduces a writing system that was not present in the input, and
// (only then) asks a model whether that language switch is legitimate in
// context, throwing if it is not.

// Meaningful writing systems we track. Latin, Common (digits/punctuation/
// symbols/emoji) and Inherited (combining marks) are intentionally NOT tracked:
// they carry no language identity and must never count as drift. A switch
// between any of the tracked scripts that is not justified by the input is the
// corruption we are guarding against.
const trackedScripts = [
  "Hebrew",
  "Armenian",
  "Arabic",
  "Cyrillic",
  "Greek",
  "Han",
  "Hiragana",
  "Katakana",
  "Hangul",
  "Thai",
  "Devanagari",
  "Georgian",
] as const;

const scriptCounts = (text: string): Record<string, number> => {
  const counts: Record<string, number> = {};
  for (const script of trackedScripts) {
    const matches = text.match(new RegExp(`\\p{Script=${script}}`, "gu"));
    if (matches) counts[script] = matches.length;
  }
  return counts;
};

export const scriptsPresent = (text: string): Set<string> =>
  new Set(Object.keys(scriptCounts(text)));

// Scripts that appear in `output` but not in `input`, with how many chars.
export const driftingScripts = (
  input: string,
  output: string,
): Record<string, number> => {
  const inputScripts = scriptsPresent(input);
  const outCounts = scriptCounts(output);
  const drift: Record<string, number> = {};
  for (const [script, count] of Object.entries(outCounts)) {
    if (!inputScripts.has(script)) drift[script] = count;
  }
  return drift;
};

// Genuine homoglyph corruption rewrites whole words/spans; a lone stray glyph
// (a rare model blip, e.g. one Thai char inside a long Hebrew reply) is noise.
// Requiring more than this many chars of a new script before treating it as
// drift avoids wasting a verifier LLM call — and, historically, hard-failing an
// otherwise-valid reply — on a single-character artifact.
const minMeaningfulDriftChars = 2;

// Like driftingScripts, but drops scripts that appear only a trivial number of
// times, so a one-off stray glyph is not treated as corruption.
export const meaningfulScriptDrift = (
  input: string,
  output: string,
): Record<string, number> =>
  Object.fromEntries(
    Object.entries(driftingScripts(input, output)).filter(
      ([, count]) => count > minMeaningfulDriftChars,
    ),
  );

const sampleOfScript = (text: string, script: string): string => {
  const matches = text.match(new RegExp(`\\p{Script=${script}}+`, "gu"));
  return (matches ?? []).slice(0, 5).join(" ").slice(0, 200);
};

const verdictSchema = z.object({
  legitimate: z.boolean().describe(
    "true if introducing this writing system is a sensible, intentional response given the input; false if it looks like a transliteration/homoglyph corruption or an unrequested script switch.",
  ),
  reason: z.string().describe("Brief justification for the verdict."),
});

const verifierSystemPrompt =
  `You detect writing-system corruption in AI output. An AI was given some input text and produced output that introduced a writing system (script) that was NOT present in the input. Decide whether that is legitimate.

Legitimate examples: the input explicitly asked to translate into that language; the input asked for content in that language; a proper noun genuinely uses that script.

NOT legitimate (this is corruption): the output rewrote the input's own language into a different but visually-similar script (e.g. Hebrew rewritten as Armenian homoglyphs), or switched scripts for no reason the input supports. When the output should have preserved the input's language/script but instead used a different one, it is NOT legitimate.

Answer strictly about whether the script switch makes sense in context.`;

export type ScriptDriftError = Error & { scriptDrift: Record<string, number> };

const makeScriptDriftError = (
  drift: Record<string, number>,
  reason: string,
): ScriptDriftError => {
  const err = new Error(
    `Script drift detected (likely homoglyph corruption): output introduced ${
      Object.entries(drift).map(([s, n]) => `${s}(${n})`).join(", ")
    } not present in input. Verifier: ${reason}`,
  ) as ScriptDriftError;
  err.scriptDrift = drift;
  return err;
};

// Throws ScriptDriftError if the output introduces an unjustified writing
// system relative to the input. No-op when scripts are consistent, so the
// extra model call only happens on the rare suspicious case.
export const assertNoScriptDrift = async (
  input: string,
  output: string,
): Promise<void> => {
  const drift = meaningfulScriptDrift(input, output);
  if (Object.keys(drift).length === 0) return;

  const driftedList = Object.keys(drift);
  const samples = driftedList
    .map((s) => `- ${s}: ${sampleOfScript(output, s)}`)
    .join("\n");

  const { legitimate, reason } = await genJson(
    { provider: "google", mini: true },
    verifierSystemPrompt,
    verdictSchema,
  )(
    `INPUT (truncated):\n${
      input.slice(0, 2000)
    }\n\nThe output introduced these writing systems that were absent from the input:\n${samples}\n\nIs introducing ${
      driftedList.join(", ")
    } legitimate here?`,
  );

  if (!legitimate) throw makeScriptDriftError(drift, reason);
};
