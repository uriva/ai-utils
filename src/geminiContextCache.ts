import type {
  FunctionDeclaration,
  GoogleGenAI,
  ToolConfig,
} from "@google/genai";
import { context, type Injection } from "@uri/inject";

export type CachedContextEntry = {
  cacheName: string;
  expiresAt: number;
};

type GoogleToolDeclarations = {
  functionDeclarations?: FunctionDeclaration[];
};

const minCacheChars = 4000;
const defaultTtlSeconds = 3600;
const expirationBufferMs = 60 * 1000;

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

export const getOrCreateGeminiContextCache = async (
  sdk: GoogleGenAI,
  model: string,
  systemInstruction: string,
  tools?: GoogleToolDeclarations[],
  toolConfig?: ToolConfig,
  ttlSeconds: number = defaultTtlSeconds,
): Promise<string | null> => {
  const toolsJson = tools ? JSON.stringify(tools) : "";
  const totalChars = systemInstruction.length + toolsJson.length;
  if (totalChars < minCacheChars) {
    return null;
  }

  const key = await makeCacheKey(model, systemInstruction, toolsJson);
  const existing = contextCacheMap.get(key);
  if (existing && Date.now() < existing.expiresAt - expirationBufferMs) {
    return existing.cacheName;
  }

  if (inFlightCreations.has(key)) {
    const existingPromise = inFlightCreations.get(key);
    return existingPromise ? await existingPromise : null;
  }

  const creationPromise = (async () => {
    try {
      const cache = await sdk.caches.create({
        model,
        config: {
          systemInstruction,
          ...(tools && tools.length > 0 ? { tools } : {}),
          ...(toolConfig ? { toolConfig } : {}),
          ttl: `${ttlSeconds}s`,
        },
      });
      const cacheName = cache.name;
      if (!cacheName) return null;
      const expiresAt = cache.expireTime
        ? Date.parse(cache.expireTime)
        : Date.now() + ttlSeconds * 1000;
      contextCacheMap.set(key, {
        cacheName,
        expiresAt,
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
