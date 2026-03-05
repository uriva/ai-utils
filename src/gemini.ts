import {
  type Content,
  type FunctionDeclaration,
  type GenerateContentParameters,
  GoogleGenAI,
  type Part,
} from "@google/genai";
import { context, type Injection, type Injector } from "@uri/inject";
import { coerce, empty, map, pipe, remove, sleep } from "gamla";
import type { ChatCompletionMessageParam } from "openai/resources/index.mjs";
import { z, type ZodType } from "zod/v4";
import type { MediaAttachment } from "./agent.ts";
import { makeCache } from "./cacher.ts";
import { structuredMsgs } from "./openai.ts";
import type { ModelOpts } from "./utils.ts";

// deno-lint-ignore no-explicit-any
const isRedundantAnyMember = (x: any) =>
  Object.keys(x).length === 1 && typeof (x.not) === "object" &&
  Object.keys(x.not).length === 0;

// deno-lint-ignore no-explicit-any
const removeAdditionalProperties = <T>(obj: Record<string, any>) => {
  if (typeof obj === "object" && obj !== null) {
    let newObj = { ...obj };
    if (obj.anyOf) {
      // deno-lint-ignore no-explicit-any
      newObj.anyOf = obj.anyOf.filter((x: any) =>
        x.type !== "null" && !isRedundantAnyMember(x)
      ).map(removeAdditionalProperties);
    }
    if (newObj.anyOf?.length === 1) newObj = newObj.anyOf[0];
    if (Array.isArray(obj.type)) {
      if (obj.type.find((x: string) => x === "null")) {
        newObj.nullable = true;
      }
      newObj.type = obj.type.find((x) => x !== "null");
    }
    if ("additionalProperties" in newObj) {
      newObj.additionalProperties = undefined;
    }
    for (const key in newObj) {
      if (key in newObj) {
        if (Array.isArray(newObj[key])) {
          newObj[key] = newObj[key].map(removeAdditionalProperties);
        } else if (
          typeof newObj[key] === "object" && newObj[key] !== null
        ) {
          newObj[key] = removeAdditionalProperties(newObj[key]);
        }
      }
    }
    return newObj;
  }
  return obj;
};

export const zodToGeminiParameters = (zodObj: ZodType): FunctionDeclaration => {
  const jsonSchema = removeAdditionalProperties(z.toJSONSchema(zodObj));
  // deno-lint-ignore no-unused-vars
  const { $schema, ...rest } = jsonSchema;
  return rest;
};

const tokenInjection: Injection<() => string> = context((): string => {
  throw new Error("no gemini token injected");
});

export const accessGeminiToken = tokenInjection.access;
export const injectGeminiToken = (token: string): Injector =>
  tokenInjection.inject(() => token);

const openAiToGeminiMessage = pipe(
  map(({ role, content }: ChatCompletionMessageParam): Content => ({
    role: role === "user" ? role : "model",
    parts: [{
      text: typeof content === "string" ? content : coerce(content?.toString()),
    }].filter((x) => x.text),
  })),
  remove(({ parts }: Content) => empty(parts ?? [])),
);

export const geminiProVersion = "gemini-3.1-pro-preview";
export const geminiFlashVersion = "gemini-3-flash-preview";
export const geminiFlashImageVersion = "gemini-2.5-flash-image";
export const geminiProImageVersion = "gemini-3-pro-image-preview";

export const geminiGenJsonFromConvo: <T extends ZodType>(
  { mini, maxOutputTokens }: ModelOpts,
  messages: ChatCompletionMessageParam[],
  zodType: T,
) => Promise<z.infer<T>> = async <T extends ZodType>(
  { mini, maxOutputTokens }: ModelOpts,
  messages: ChatCompletionMessageParam[],
  zodType: T,
): Promise<z.infer<T>> => {
  const cacher = makeCache("geminiCompletionResponseText-v2");
  const cachedCall = cacher((req: GenerateContentParameters) =>
    new GoogleGenAI({ apiKey: tokenInjection.access() }).models.generateContent(
      req,
    ).then(({ text }) => text || "{}")
  );
  return JSON.parse(
    await cachedCall({
      model: mini ? geminiFlashVersion : geminiProVersion,
      config: {
        responseMimeType: "application/json",
        responseSchema: zodToGeminiParameters(zodType),
        ...(maxOutputTokens ? { maxOutputTokens } : {}),
      },
      contents: pipe(openAiToGeminiMessage)(messages),
    }),
  );
};

export const geminiGenJson =
  <T extends ZodType>(opts: ModelOpts, systemMsg: string, zodType: T) =>
  (userMsg: string): Promise<z.TypeOf<T>> =>
    geminiGenJsonFromConvo(
      opts,
      structuredMsgs(systemMsg, userMsg),
      zodType,
    );

export const attachmentsToParts = (
  attachments: MediaAttachment[],
): Part[] =>
  attachments.flatMap((a): Part[] => {
    const mediaPart: Part = a.kind === "inline"
      ? { inlineData: { data: a.dataBase64, mimeType: a.mimeType } }
      : { fileData: { fileUri: a.fileUri, mimeType: a.mimeType } };
    const parts: Part[] = [mediaPart];
    if (a.caption?.trim()) parts.push({ text: a.caption });
    return parts;
  });

export const geminiGenText = async (
  { mini }: ModelOpts,
  prompt: string,
  attachments: MediaAttachment[],
): Promise<string> => {
  const result = await new GoogleGenAI({
    apiKey: tokenInjection.access(),
  }).models.generateContent({
    model: mini ? geminiFlashVersion : geminiProVersion,
    contents: [{
      role: "user",
      parts: [...attachmentsToParts(attachments), { text: prompt }],
    }],
  });
  return result.text ?? "";
};

type UploadResult = { geminiUri: string; mimeType: string };

const uploadBlobToGemini = async (
  blob: Blob,
  mimeType: string,
): Promise<UploadResult> => {
  const ai = new GoogleGenAI({ apiKey: tokenInjection.access() });
  const { uri, mimeType: mimeType2 } = await ai.files.upload({
    file: blob,
    config: { mimeType },
  });
  if (!uri || !mimeType2) {
    throw new Error("Gemini file upload failed: missing uri or mimeType");
  }
  return { geminiUri: uri, mimeType: mimeType2 };
};

const uploadToGeminiFromUrl = async (
  url: string,
  mimeType: string,
): Promise<UploadResult> => {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch file for Gemini upload: ${url}`);
  }
  return uploadBlobToGemini(await res.blob(), mimeType);
};

const uploadToGeminiFromFile = (
  mimeType: string,
  dataBase64: string,
): Promise<UploadResult> =>
  uploadBlobToGemini(
    new Blob(
      [Uint8Array.from(atob(dataBase64), (c) => c.charCodeAt(0))],
      { type: mimeType },
    ),
    mimeType,
  );

const geminiFileNameFromUri = (uri: string) => {
  const match = uri.match(/files\/([^/?]+)/);
  return match ? `files/${match[1]}` : null;
};

const isGeminiFileUri = (uri: string) =>
  uri.startsWith("https://generativelanguage.googleapis.com/");

const waitForFileActive = async (
  fileUri: string,
  attempts = 30,
): Promise<void> => {
  const name = geminiFileNameFromUri(fileUri);
  if (!name) return;
  if (attempts <= 0) {
    throw new Error(`Gemini file ${name} did not become ACTIVE after 30s`);
  }
  const ai = new GoogleGenAI({ apiKey: tokenInjection.access() });
  const file = await ai.files.get({ name });
  if (file.state === "ACTIVE") return;
  if (file.state === "FAILED") {
    throw new Error(`Gemini file ${name} failed processing`);
  }
  await sleep(1000);
  return waitForFileActive(fileUri, attempts - 1);
};

export const ensureGeminiAttachmentIsLink = async (
  attachment: MediaAttachment,
): Promise<MediaAttachment> => {
  if (
    attachment.kind === "file" &&
    attachment.fileUri.startsWith("https://generativelanguage.googleapis.com/")
  ) {
    return attachment;
  }
  if (attachment.kind === "file" && attachment.fileUri.trim()) {
    const { geminiUri, mimeType } = await uploadToGeminiFromUrl(
      attachment.fileUri,
      attachment.mimeType,
    );
    return {
      kind: "file",
      fileUri: geminiUri,
      mimeType,
      caption: attachment.caption,
    };
  }
  if (attachment.kind === "inline") {
    const { geminiUri, mimeType } = await uploadToGeminiFromFile(
      attachment.mimeType,
      attachment.dataBase64,
    );
    return {
      kind: "file",
      fileUri: geminiUri,
      mimeType,
      caption: attachment.caption,
    };
  }
  throw new Error(
    "File attachment missing fileUri or unsupported attachment kind",
  );
};

export const ensureGeminiAttachmentIsActive = async (
  attachment: MediaAttachment,
): Promise<MediaAttachment> => {
  const result = await ensureGeminiAttachmentIsLink(attachment);
  if (result.kind === "file" && isGeminiFileUri(result.fileUri)) {
    await waitForFileActive(result.fileUri);
  }
  return result;
};
