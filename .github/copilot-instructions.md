To run the tests with the required consult the deno task "test" in the
`deno.json` file.

Don't use `case`, prefer an `if` with early return.

Avoid using `as`, prefer type guards or figuring out the type another way.

Avoid try/catch in tests unless explicitly approved by user.

Avoid nested functions, unless necessary. Consider currying if you need to inject dependencies.

Use a functional style, prefer `map`, `filter`, `pipe` from Gamla instead of loops and mutable variables.

If a variable is used only once, consider inlining it. If it is too complex, factor it into a well-named function instead.

Prefer gamla's `empty` than length checks - more readable.
