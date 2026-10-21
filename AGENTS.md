# General

- **Reproduce first:** Never fix a quality or behavior issue without a failing
  reproducing test first.
- **Fix at the earliest point of failure:** When one root defect cascades into
  several downstream symptoms, fix the root mechanism rather than stacking
  guards at downstream symptoms.
- **Target the first failure point in history:** When reproducing a
  hallucination or quality regression, isolate the earliest turn where the bad
  behavior first enters history rather than replaying the downstream cascade.
- **No consumer-specific data (open source):** Never put consumer-specific or
  bot-specific data into this repo (no real system prompts, persona text,
  customer/brand names, pricing, bot/conversation IDs, or production excerpts)
  in source, tests, scripts, or comments. Use short, generic, synthetic English
  inputs that exercise the same mechanism.
- **English test suite:** All test prompts, instructions, mock tool outputs, and
  assertions MUST be in English, except in tests specifically verifying
  language- or script-dependent mechanisms (e.g., script-drift detection,
  non-ASCII token estimation, RTL sanitization).
- **Avoid regexes, keyword whitelists, and name lists (CRITICAL):** Never use
  long regexes, keyword dictionaries, or lists of names/words to classify
  intent, gender, language, or semantic attributes. Expose explicit tool
  parameters so the agent selects options during creation, or use LLM
  intelligence (`genJson` / model calls cached with `rmmbr`) when dynamic
  inference is required.
- **No string duplication:** Never duplicate system-audit thoughts, warning
  utterances, error messages, or tool names across source and tests. Export
  shared strings as module-level constants from source and import them in tests.

# Token Cost Discipline

Every harness feature or quality fix must account for its token cost (one-shot
vs. re-sent per model call, and extra model calls):

- Prefer deterministic (CPU-only) checks over LLM judges.
- Anything appended to `ephemeralHistory` is re-sent on every subsequent model
  call within the run — price injected notices as per-call cost, not one-shot.
- Always-on guards must be zero-cost on the happy path (no context growth and no
  extra model calls unless the guard fires).

# Architecture & Testing

- **`agent.ts` is provider-agnostic:** It must not import provider-specific
  files (`geminiAgent.ts`, `anthropicAgent.ts`, `kimiAgent.ts`). Provider
  dispatch lives in `mod.ts` under `providerCaller`. Factor provider-specific
  behavior behind an injection point.
- **Provider-agnostic loop invariants → inject a fake `CallModel`:** Tests for
  `runAgent` invariants (streaming contract, iteration logic, event emission)
  should use `injectCallModel(fake)` rather than hitting live providers (see
  `tests/thinking.test.ts`).
- **Integration & bug-repro tests → real agent + all providers + `rmmbr`:**
  - Run the real agent (`runAgent` / `agentDeps` / `runForAllProviders`) against
    the real API and assert on history events. Only set `geminiOnly=true` when
    testing a Gemini-specific feature (and state why).
  - Wrap live-API tests with `injectSecrets` (`test_helpers.ts`) so they use the
    `rmmbr` cache (`injectCallModelWrapper`). Never fall back to a passthrough
    or uncached runner; missing `RMMBR_TOKEN` must fail immediately at startup.
  - **Explicit cache keys:** `rmmbrCacheWithKey` requires an explicit
    `customKeyFn` that strips per-run randomness (`id`, `timestamp` — see
    `eventsCacheKey`). If you add a new random/timestamp field to
    `HistoryEvent`, strip it in the cache key or CI will miss the cache.
- **Running tests:** Check `deno.json` (`deno task test`). Run only the tests
  affected by your changes locally; CI runs the full suite and `deno task check`
  on push. Avoid `try`/`catch` and mocking in tests unless explicitly approved.

# Coding Style

- Minimal comments and no JSDoc unless necessary; rely on types and naming.
- Functional style with `gamla` (`pipe`, `map`, `filter`, `empty`, `nonempty`).
  Avoid `class`, `for`/`while` loops, `let`, and `switch`/`case` (prefer `if`
  with early return or ternaries).
- Arrow functions instead of `function`. Destructure in function signatures when
  clean. Inline single-use variables or extract a named function.
- Avoid nesting functions; keep functions at module level (use currying for
  dependencies). Keep function bodies small when adding logic.
- Avoid `as` type assertions; prefer type guards or type inference. Omit return
  type annotations when inferrable.

# Publishing & Downstream Updates

CI publishes to JSR on push to `main` when the `deno.json` version is bumped.
Never run `deno publish` locally. To use changes in a consumer repo (e.g.
`prompt2bot`): bump `version` in `deno.json` here, push to `main`, then bump the
`@uri/ai-utils` version in the consumer's `deno.json` and run `deno install`.
