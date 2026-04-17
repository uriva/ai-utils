# General

Never fix quality issue without first a reproducing test.

To run the tests with the required env vars consult the deno task "test" in the
`deno.json` file.

This is a Deno project published to JSR. CI publishes on push to main. Do not
attempt `deno publish` locally. Bump the version in `deno.json` before pushing
if the previous version was already published.

### Workflow for Updating this Package

When modifying this library to use in another project like `prompt2bot`:

1. Make your changes in this repo.
2. Bump the version in `deno.json`.
3. Commit and push your changes to GitHub. CI will automatically publish to JSR.
   **Do not run `deno publish` yourself**.
4. Go to the consumer project (`prompt2bot`), bump the version of `ai-utils` in
   its `deno.json`, commit and push to deploy the consumer project.

Pre-commit hook runs
`deno fmt --check && deno lint && deno check *.ts src/**/*.ts tests/**/*.ts`.

Don't use `case`, prefer an `if` with early return.

Avoid using `as`, prefer type guards or figuring out the type another way.

Avoid try/catch in tests unless explicitly approved by user.

Avoid nested functions, unless necessary. Consider currying if you need to
inject dependencies.

If a variable is used only once, consider inlining it. If it is too complex,
factor it into a well-named function instead.

Prefer gamla's `empty` than length checks - more readable.

Don't use commenting or jsdoc unless instructed or in extreme cases. Typings and
names of variables should do most of the documenting.

Use arrow functions instead of `function` keyword.

Use a functional style, prefer `map`, `filter`, `pipe` from Gamla instead of
loops and mutable variables. Don't use `class` or `for`/`while` loops unless
absolutely necessary.

When writing tests that verify a behaviour against the api, actually call the
api instead of mocking it. Only mock when given explicit permission by the user.

**All bug reproductions and behaviour tests MUST be agent-level.** Write a test
that runs the real agent (via `runAgent` / `agentDeps` / `runForAllProviders`)
against the real API, and asserts on the resulting history events. This is the
only way to catch real integration issues, because bugs live in the interaction
between Zod schemas, JSON-schema conversion, provider-specific function
declarations, strict validators on the provider side, and our call-site code. A
pure-logic unit test on e.g. `buildReq` output shape or a schema-transform
function is insufficient by itself — it will pass while production still breaks.

**All agent-level tests MUST run on every provider.** Use `runForAllProviders`
unless the bug is strictly provider-specific (and state why in a comment). Only
set `geminiOnly=true` when the test uses a Gemini-specific feature or input
format.

**All agent-level tests MUST use a real rmmbr cacher.** `test_helpers.ts` wires
a real rmmbr cache into `injectSecrets`. Do not replace it with a passthrough
cacher. Deterministic fixtures make test runs fast, stable, and cheap; uncached
runs burn tokens on every push and introduce flakiness. If you add a new test
that hits the API, make sure it's wrapped with `injectSecrets` so it picks up
the cache.

**If the rmmbr token is unavailable, the test suite MUST fail violently at
startup.** `test_helpers.ts` reads `RMMBR_TOKEN` from the environment and throws
immediately if it's missing. Never fall back to a passthrough cacher, an
empty-token in-memory/on-disk cache, or skip caching "just for this run". A
silent fallback hides configuration bugs, produces non-deterministic test
results, and burns API tokens on every push. Missing token → loud crash before
any test runs.

**The main agent path (`rawCallGemini`, `callAnthropic`, `callKimi`) is wired
through `makeCache`.** When you add a new provider call path, wrap it with
`makeCache` the same way so agent-level tests for it are cached. If you see a
remote call that isn't cached, that's a bug — fix it.

When running tests after changes, run only the tests affected by your changes.
Do not run the full local test suite for final verification - CI will run the
full suite after you push.

When adding logic, function bodies typically should not enlarge. New logic can
be encapsulated in a new function. Or, one can refactor such that the old
functions are even smaller than before. The added benefit is that there are less
diffs hard to review.

Typically it's better to "solve for the single case" then use functions like
`map` and `filter` to handle the more complex cases. To prevent indent, one can
define the single case and use `map` or similar functions in the call site.

Destructuring in the function signature makes for more readable code rather than
repeat myArg.x myArg.y everywhere. Most times this is possible, but not always.

If a type is inferrable from the function, prefer not to annotate it. This is
the common case.

# Architecture

## Gemini call chain (`src/geminiAgent.ts`)

The Gemini API call path has several layers of error handling:

1. `rawCallGemini` — makes the raw API call, parses the response. No error
   logging here.
2. `callGeminiWithRetry` — wraps `rawCallGemini` with
   `conditionalRetry(isServerError)(1000, 4, rawCallGemini)` — retries up to 4
   times with 1s delay on 500+ errors.
3. `callGemini` — wraps `callGeminiWithRetry`. On server error, tries the
   alternate model (pro<->flash). No error logging here.
4. `callGeminiWithFixHistory` — wraps `callGemini`. Handles token limit exceeded
   (drops oldest half), file permission errors (strips files), unsupported mime
   types. Error logging (`geminiError.access`) fires only here as an outer
   catch, after all recovery strategies are exhausted. This ensures only truly
   terminal errors are reported.

Error classification predicates: `isServerError` (status >= 500),
`isTokenLimitExceeded` (400 + "token count exceeds"), `is403PermissionError`,
`isFileNotActiveError`, `isUnsupportedMimeTypeError`.

## Agent and skills system (`src/agent.ts`)

- `callToResult` resolves function calls from the model. If a function name
  contains `/` (e.g. `skillName/toolName`) and is not found in `allTools`, it
  gets routed through the `run_command` handler by transforming args to
  `{command: name, params: args}`. This handles models that call skill tools
  directly instead of going through `run_command`.
- `createSkillTools` creates exactly two meta-tools: `run_command` and
  `learn_skill`. Individual skill tools are NOT added to Gemini's tool
  declarations.
- `handleFunctionCalls` processes all function calls from a single model
  response in parallel.

## Model versions (`src/gemini.ts`)

Model version constants: `geminiProVersion` = `"gemini-3.1-pro-preview"`,
`geminiFlashVersion` = `"gemini-3-flash-preview"`. These are preview models and
may have server-side bugs (500 errors) that are not caused by our requests.
