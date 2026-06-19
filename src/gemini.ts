import {
  type Content,
  type FunctionDeclaration,
  type GenerateContentParameters,
  GoogleGenAI,
  type Part,
  ThinkingLevel,
} from "@google/genai";
import { context, type Injection, type Injector } from "@uri/inject";
import { coerce, conditionalRetry, empty, map, pipe, remove } from "gamla";
import { isRetryableError, type ModelOpts } from "./utils.ts";
import type { ChatCompletionMessageParam } from "openai/resources/index.mjs";
import { z, type ZodType } from "zod/v4";
import type { MediaAttachment } from "./agent.ts";
import { makeCache } from "./cacher.ts";

import { pruneDefaultsFromRequired } from "./toolTyping.ts";

export { zodToTypingString } from "./toolTyping.ts";

// deno-lint-ignore no-explicit-any
const isRedundantAnyMember = (x: any) =>
  Object.keys(x).length === 1 && typeof (x.not) === "object" &&
  Object.keys(x.not).length === 0;

// deno-lint-ignore no-explicit-any
const removeAdditionalProperties = <T>(obj: Record<string, any>) => {
  if (typeof obj === "object" && obj !== null) {
    let newObj = pruneDefaultsFromRequired({ ...obj });
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

    // Explicitly delete unsupported keys instead of setting to undefined
    delete newObj.additionalProperties;
    delete newObj.default;
    delete newObj.$schema;

    if (newObj.const !== undefined) {
      newObj.enum = [newObj.const];
      delete newObj.const;
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
  return jsonSchema as unknown as FunctionDeclaration;
};

// Walk the processed JSON schema to ensure Gemini/LLM providers support all of its features.
// It does not support nested anyOf/oneOf/allOf/const.
// deno-lint-ignore no-explicit-any
export const validateSchema = (schema: any, path: string = "root"): void => {
  if (typeof schema !== "object" || schema === null) return;

  if ("anyOf" in schema || "any_of" in schema) {
    throw new Error(
      `Unsupported schema construct 'anyOf' at ${path}. unions or anyOf are not supported.`,
    );
  }
  if ("oneOf" in schema || "one_of" in schema) {
    throw new Error(
      `Unsupported schema construct 'oneOf' at ${path}. oneOf is not supported.`,
    );
  }
  if ("allOf" in schema || "all_of" in schema) {
    throw new Error(
      `Unsupported schema construct 'allOf' at ${path}. allOf is not supported.`,
    );
  }
  if ("const" in schema) {
    throw new Error(
      `Unsupported schema construct 'const' at ${path}. const is not supported (use enum instead).`,
    );
  }

  // Check type if present
  if ("type" in schema) {
    const supportedTypes = [
      "string",
      "number",
      "integer",
      "boolean",
      "object",
      "array",
      "null",
    ];
    if (Array.isArray(schema.type)) {
      for (const t of schema.type) {
        if (!supportedTypes.includes(t)) {
          throw new Error(
            `Unsupported type '${t}' in union type at ${path}. Supported types: ${
              supportedTypes.join(", ")
            }`,
          );
        }
      }
    } else if (typeof schema.type === "string") {
      if (!supportedTypes.includes(schema.type)) {
        throw new Error(
          `Unsupported type '${schema.type}' at ${path}. Supported types: ${
            supportedTypes.join(", ")
          }`,
        );
      }
    }
  }

  // Recursively validate properties
  if (schema.properties && typeof schema.properties === "object") {
    for (const [key, prop] of Object.entries(schema.properties)) {
      validateSchema(prop, `${path}.properties.${key}`);
    }
  }

  // Recursively validate items
  if (schema.items) {
    if (Array.isArray(schema.items)) {
      // deno-lint-ignore no-explicit-any
      schema.items.forEach((item: any, index: number) => {
        validateSchema(item, `${path}.items[${index}]`);
      });
    } else {
      validateSchema(schema.items, `${path}.items`);
    }
  }

  // Recursively validate additionalProperties
  if (
    schema.additionalProperties &&
    typeof schema.additionalProperties === "object"
  ) {
    validateSchema(schema.additionalProperties, `${path}.additionalProperties`);
  }
};

export const validateZodSchema = (
  zodObj: ZodType,
  path: string = "root",
): void => {
  const jsonSchema = removeAdditionalProperties(z.toJSONSchema(zodObj));
  validateSchema(jsonSchema, path);
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

type GeminiModelVersions = {
  pro: string;
  flash: string;
  fallback: string;
};

const defaultGeminiModelVersions: GeminiModelVersions = {
  pro: "gemini-3.5-flash",
  flash: "gemini-3.5-flash",
  fallback: "gemini-3.1-flash-lite",
};

const geminiModelVersions: Injection<() => GeminiModelVersions> = context(() =>
  defaultGeminiModelVersions
);

export const injectGeminiModelVersions = geminiModelVersions.inject;

export const geminiProVersion = defaultGeminiModelVersions.pro;
export const geminiFlashVersion = defaultGeminiModelVersions.flash;
export const geminiFallbackVersion = defaultGeminiModelVersions.fallback;

export const geminiModelVersion = (mini: boolean | undefined) => {
  const versions = geminiModelVersions.access();
  return mini ? versions.flash : versions.pro;
};

export const alternateGeminiModelVersion = (model: string) => {
  const versions = geminiModelVersions.access();
  return model === versions.pro || model === versions.flash
    ? versions.fallback
    : model;
};

export const geminiThinkingConfig = (mini: boolean | undefined) => ({
  includeThoughts: false,
  ...(mini
    ? { thinkingLevel: ThinkingLevel.LOW }
    : { thinkingLevel: ThinkingLevel.HIGH }),
});

export const geminiGenJsonFromConvo: <T extends ZodType>(
  { mini, maxOutputTokens }: ModelOpts,
  messages: ChatCompletionMessageParam[],
  zodType: T,
  attachments?: MediaAttachment[],
) => Promise<z.infer<T>> = async <T extends ZodType>(
  { mini, maxOutputTokens }: ModelOpts,
  messages: ChatCompletionMessageParam[],
  zodType: T,
  attachments?: MediaAttachment[],
): Promise<z.infer<T>> => {
  const cacher = makeCache("geminiCompletionResponseText-v2");
  const cachedCall = cacher(
    conditionalRetry(isRetryableError)(
      1000,
      3,
      (req: GenerateContentParameters) =>
        new GoogleGenAI({ apiKey: tokenInjection.access() }).models
          .generateContent(req)
          .then(({ text }) => text || "{}"),
    ),
  );
  const contents = pipe(openAiToGeminiMessage)(messages);
  if (attachments && attachments.length > 0) {
    const lastUserMessage = [...contents].reverse().find((c) =>
      c.role === "user"
    );
    if (lastUserMessage) {
      if (!lastUserMessage.parts) lastUserMessage.parts = [];
      const resolvedAttachments = await Promise.all(
        attachments.map(ensureGeminiAttachmentIsLink),
      );
      lastUserMessage.parts.push(...attachmentsToParts(resolvedAttachments));
    }
  }
  return JSON.parse(
    await cachedCall({
      model: geminiModelVersion(mini),
      config: {
        responseMimeType: "application/json",
        responseSchema: zodToGeminiParameters(zodType),
        thinkingConfig: geminiThinkingConfig(mini),
        ...(maxOutputTokens ? { maxOutputTokens } : {}),
      },
      contents,
    }),
  );
};

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
  const result = await conditionalRetry(isRetryableError)(
    1000,
    3,
    () =>
      new GoogleGenAI({
        apiKey: tokenInjection.access(),
      }).models.generateContent({
        model: geminiModelVersion(mini),
        config: { thinkingConfig: geminiThinkingConfig(mini) },
        contents: [{
          role: "user",
          parts: [...attachmentsToParts(attachments), { text: prompt }],
        }],
      }),
  )();
  return result.text ?? "";
};

type UploadResult = { geminiUri: string; mimeType: string };

const rawUploadBlobToGemini = async (
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

const isTransientFetchError = (error: unknown) =>
  error instanceof TypeError &&
  /reading a body|network|connection/i.test(error.message);

const isRetryableUploadError = (error: unknown) =>
  isRetryableError(error) || isTransientFetchError(error);

const uploadBlobToGemini = conditionalRetry(isRetryableUploadError)(
  1000,
  4,
  rawUploadBlobToGemini,
);

const fetchAndUploadToGemini = async (
  url: string,
  mimeType: string,
): Promise<UploadResult> => {
  const res = await fetch(url);
  if (!res.ok) {
    await res.body?.cancel();
    throw new Error(`Failed to fetch file for Gemini upload: ${url}`);
  }
  return uploadBlobToGemini(await res.blob(), mimeType);
};

const fetchAndUploadToGeminiCached = (
  url: string,
  mimeType: string,
): Promise<UploadResult> =>
  makeCache("gemini-file-upload-v3")(
    conditionalRetry(isRetryableUploadError)(1000, 3, fetchAndUploadToGemini),
  )(url, mimeType);

const uploadToGeminiFromUrl = fetchAndUploadToGeminiCached;

const uploadToGeminiFromFileInner = (
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

const uploadToGeminiFromFile = (
  mimeType: string,
  dataBase64: string,
): Promise<UploadResult> =>
  makeCache("gemini-inline-upload-v3")(uploadToGeminiFromFileInner)(
    mimeType,
    dataBase64,
  );

export const isGeminiFileUri = (uri: string): boolean =>
  uri.startsWith("https://generativelanguage.googleapis.com/");

const textExtensions = [
  ".ts",
  ".tsx",
  ".js",
  ".jsx",
  ".json",
  ".css",
  ".html",
  ".md",
  ".py",
  ".rs",
  ".go",
  ".sh",
  ".yaml",
  ".yml",
  ".toml",
  ".xml",
  ".txt",
  ".csv",
];

export const normalizeMimeType = (
  mimeType: string,
  fileUri?: string,
): string => {
  const normalized = mimeType.toLowerCase().trim();
  const uriLower = fileUri?.toLowerCase().trim();
  return normalized === "video/vnd.dlna.mpeg-tts" ||
      (uriLower && textExtensions.some((ext) => uriLower.endsWith(ext)))
    ? "text/plain"
    : mimeType;
};

export const ensureGeminiAttachmentIsLink = async (
  attachment: MediaAttachment,
): Promise<MediaAttachment> => {
  if (attachment.kind === "file" && isGeminiFileUri(attachment.fileUri)) {
    return attachment;
  }
  if (attachment.kind === "file" && attachment.fileUri.trim()) {
    const { geminiUri, mimeType } = await uploadToGeminiFromUrl(
      attachment.fileUri,
      normalizeMimeType(attachment.mimeType, attachment.fileUri),
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
      normalizeMimeType(attachment.mimeType, attachment.caption),
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

export const countTextTokens = async (
  text: string | undefined,
): Promise<number> => {
  if (!text) return 0;
  const sdk = new GoogleGenAI({ apiKey: tokenInjection.access() });
  const { totalTokens } = await sdk.models.countTokens({
    model: "gemini-3.5-flash",
    contents: text,
  });
  return totalTokens ?? 0;
};
