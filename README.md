# ai-utils

A harness for running AI agents in Deno. Give it a prompt, tools, and a
provider, and it manages the conversation loop, history, retries, compaction,
and structured output for you.

Supports Gemini, OpenAI, and Moonshot (Kimi). Works with text, media
attachments, and real-time audio.

## Install

```
deno add jsr:@uri/ai-utils
```

## Running an agent

The core function is `runAgent`. You describe what you want and it handles the
rest: calling the model, executing tool calls, appending results to history, and
looping until the model is done.

```ts
import { runAgent } from "@uri/ai-utils";

await runAgent({
  prompt: "You are a helpful assistant.",
  tools: [
    {
      name: "get_weather",
      description: "Get current weather for a city",
      inputSchema: { type: "object", properties: { city: { type: "string" } } },
      handler: async ({ city }) => ({ temp: 22, condition: "sunny" }),
    },
  ],
  maxIterations: 5,
  getHistory: async () => [],
  setHistory: async (events) => {},
});
```

`getHistory` and `setHistory` let you persist conversation state however you
want (KV store, database, in-memory array).

## Structured JSON output

When you just need a model to return typed data, use the `genJson` helpers
instead of running a full agent loop. Pass a Zod schema and get back a typed
object.

```ts
import { geminiGenJson, injectGeminiToken } from "@uri/ai-utils";
import { z } from "@uri/ai-utils";

const extract = geminiGenJson(
  { mini: true },
  "Extract the person's name and age from the text.",
  z.object({ name: z.string(), age: z.number() }),
);

const result = await injectGeminiToken("YOUR_KEY")(
  () => extract("My name is Alice and I'm 30."),
)();
// { name: "Alice", age: 30 }
```

Works the same way with OpenAI via `openAiGenJson`.

## History compaction

Long conversations get expensive and eventually exceed context windows. ai-utils
handles this automatically:

1. Segments history by 30-minute gaps
2. Partitions segments into "keep" and "summarize" based on a token budget
3. Summarizes old segments into structured summaries (key entities, decisions,
   actions taken, pending items, context)
4. Keeps tool call/result pairs atomic so they're never split

```ts
import {
  partitionSegments,
  segmentHistoryEvents,
  summarizeEvents,
} from "@uri/ai-utils";

const segments = segmentHistoryEvents(history, 30 * 60 * 1000);
const { kept, toSummarize } = partitionSegments(30000, segments);
const summary = await summarizeEvents(toSummarize.flatMap((s) => s.events));
```

## Media attachments

Attach images, audio, or other files to user messages:

```ts
import { participantUtteranceTurn } from "@uri/ai-utils";

const history = [
  participantUtteranceTurn({
    name: "user",
    text: "What's in this image?",
    attachments: [
      {
        kind: "file",
        mimeType: "image/jpeg",
        fileUri: "https://example.com/photo.jpg",
      },
    ],
  }),
];
```

## Real-time audio

For voice conversations, create a live audio session with Gemini:

```ts
import { createAudioSession } from "@uri/ai-utils";

const session = await createAudioSession({
  prompt: "You are a voice assistant.",
  onEvent: (event) => {
    if (event.type === "audio") sendToSpeaker(event.data);
  },
});

session.sendAudio(microphoneChunk);
```

## Dependency injection

API keys and caching are injected via context rather than passed around. This
keeps function signatures clean and composable.

```ts
import { injectGeminiToken, injectCacher } from "@uri/ai-utils";
import { pipe } from "gamla";

const withDeps = pipe(
  injectCacher(() => (f) => f),
  injectGeminiToken("YOUR_KEY"),
);

await withDeps(() => runAgent({ ... }))();
```
