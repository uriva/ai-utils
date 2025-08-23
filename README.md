# ai-utils

A collection of utilities for working with AI APIs (OpenAI, Gemini) in Deno.

## Features

- Unified interface for different providers.
- Schema-based JSON output using [zod](https://github.com/colinhacks/zod)
- Dependency injection for API keys and caching
- Utilities for matching, conversation history, and more

## Usage

Install via [jsr](https://jsr.io):

```ts
import {
  geminiGenJsonFromConvo,
  injectCacher,
  injectGeminiToken,
  injectOpenAiToken,
  openAiGenJsonFromConvo,
} from "@uri/ai-utils";
```

Inject your API keys and cacher:

```ts
import { pipe } from "gamla";

const injectSecrets = pipe(
  injectCacher(() => (f) => f),
  injectOpenAiToken("YOUR_OPENAI_API_KEY"),
  injectGeminiToken("YOUR_GEMINI_API_KEY"),
);
```

Call a model:

```ts
import { z } from "zod/v4";

const schema = z.object({ hello: z.string() });
const messages = [
  { role: "system", content: "Say hello as JSON." },
  { role: "user", content: "hello" },
];

const result = await injectSecrets(async () =>
  await openAiGenJsonFromConvo({ mini: false }, messages, schema)
);
console.log(result); // { hello: "..." }
```
