To run the tests with the required consult the deno task "test" in the
`deno.json` file.

Don't use `case`, prefer an `if` with early return.

Avoid using `as`, prefer type guards or figuring out the type another way.

Avoid try/catch in tests unless explicitly approved by user.