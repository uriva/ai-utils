import { z } from "zod/v4";
import { zodToGeminiParameters } from "../src/gemini.ts";
console.log(
  JSON.stringify(
    zodToGeminiParameters(z.object({ city: z.string() })),
    null,
    2,
  ),
);
