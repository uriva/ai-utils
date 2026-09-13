import {
  type FunctionDeclaration,
  GoogleGenAI,
  type ToolConfig,
} from "@google/genai";
import { context, type Injection } from "@uri/inject";
import { makeCache } from "./cacher.ts";
import { accessGeminiToken } from "./gemini.ts";

export type CachedContextEntry = {
  cacheName: string;
  expiresAt: number;
};

type GoogleToolDeclarations = {
  functionDeclarations?: FunctionDeclaration[];
};

export const geminiContextCacheTtlSeconds = 3600;
export const geminiContextCacheBufferSeconds = 120;
export const geminiContextCacheClientTtlSeconds = geminiContextCacheTtlSeconds -
  geminiContextCacheBufferSeconds;
export const geminiContextCacheId = "gemini-context-cache-v1";

const minCacheChars = 4000;

const contextCacheMap = new Map<string, CachedContextEntry>();
const inFlightCreations = new Map<string, Promise<string | null>>();

const hashString = async (text: string): Promise<string> => {
  const buf = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(text),
  );
  return Array.from(new Uint8Array(buf))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
};

const makeCacheKey = async (
  model: string,
  systemInstruction: string,
  toolsJson: string,
): Promise<string> =>
  `${model}:${await hashString(systemInstruction + "::" + toolsJson)}`;

export const clearGeminiContextCacheMap = (): void => {
  contextCacheMap.clear();
  inFlightCreations.clear();
};

export const invalidateGeminiContextCache = async (
  model: string,
  systemInstruction: string,
  tools?: GoogleToolDeclarations[],
): Promise<void> => {
  const toolsJson = tools ? JSON.stringify(tools) : "";
  const key = await makeCacheKey(model, systemInstruction, toolsJson);
  contextCacheMap.delete(key);
  inFlightCreations.delete(key);
};

const createRemoteCache = async (
  model: string,
  systemInstruction: string,
  toolsJson: string,
  toolConfigJson: string,
): Promise<string> => {
  const sdk = new GoogleGenAI({ apiKey: accessGeminiToken() });
  const tools = toolsJson ? JSON.parse(toolsJson) : undefined;
  const toolConfig = toolConfigJson ? JSON.parse(toolConfigJson) : undefined;
  const cache = await sdk.caches.create({
    model,
    config: {
      systemInstruction,
      ...(tools && tools.length > 0 ? { tools } : {}),
      ...(toolConfig ? { toolConfig } : {}),
      ttl: `${geminiContextCacheTtlSeconds}s`,
    },
  });
  if (!cache.name) {
    throw new Error("Gemini context cache creation returned empty name");
  }
  return cache.name;
};

export const getOrCreateGeminiContextCache = async (
  _sdk: GoogleGenAI,
  model: string,
  systemInstruction: string,
  tools?: GoogleToolDeclarations[],
  toolConfig?: ToolConfig,
  ttlSeconds: number = geminiContextCacheTtlSeconds,
): Promise<string | null> => {
  const toolsJson = tools ? JSON.stringify(tools) : "";
  const totalChars = systemInstruction.length + toolsJson.length;
  if (totalChars < minCacheChars) {
    return null;
  }

  const toolConfigJson = toolConfig ? JSON.stringify(toolConfig) : "";
  const key = await makeCacheKey(
    model,
    systemInstruction,
    toolsJson + "::" + toolConfigJson,
  );
  const existing = contextCacheMap.get(key);
  if (
    existing &&
    Date.now() < existing.expiresAt - geminiContextCacheBufferSeconds * 1000
  ) {
    return existing.cacheName;
  }

  if (inFlightCreations.has(key)) {
    const existingPromise = inFlightCreations.get(key);
    return existingPromise ? await existingPromise : null;
  }

  const creationPromise = (async () => {
    try {
      const cacheName = await makeCache(
        geminiContextCacheId,
        geminiContextCacheClientTtlSeconds,
      )(createRemoteCache)(
        model,
        systemInstruction,
        toolsJson,
        toolConfigJson,
      );
      if (!cacheName) return null;
      contextCacheMap.set(key, {
        cacheName,
        expiresAt: Date.now() + ttlSeconds * 1000,
      });
      return cacheName;
    } catch (_err) {
      contextCacheMap.delete(key);
      return null;
    } finally {
      inFlightCreations.delete(key);
    }
  })();

  inFlightCreations.set(key, creationPromise);
  return await creationPromise;
};

const geminiContextCachingEnabled: Injection<() => boolean> = context(
  (): boolean => true,
);

export const injectGeminiContextCachingEnabled =
  geminiContextCachingEnabled.inject;

export const isGeminiContextCachingEnabled = geminiContextCachingEnabled.access;
