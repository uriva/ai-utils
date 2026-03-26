import { GoogleGenAI } from "@google/genai";
const sdk = new GoogleGenAI({ apiKey: "dummy" });
const res = await sdk.models.generateContentStream({
  model: "gemini-3.1-pro-preview",
  contents: "hello",
});
console.log(typeof res, res.constructor.name);
