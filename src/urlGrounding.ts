import { filter, map, nonempty, pipe } from "gamla";
import type { HistoryEvent } from "./agent.ts";

// Models sometimes fabricate URLs/hosts inside tool-call parameters or user-facing
// utterances (guessed API endpoints, invented deep links, fabricated phone numbers)
// when the information was never retrieved or provided. This module detects
// non-trivial ungrounded URLs and phone numbers deterministically on the CPU (0 LLM cost)
// so the agent loop can reject the output and instruct the model to self-correct.

const urlPattern = /https?:\/\/[^\s<>"'`)\]]+/gi;
const urlHostPattern = /https?:\/\/([a-z0-9](?:[a-z0-9.-]*[a-z0-9])?)/gi;
const hostAssignmentPattern = /\bhost\s*:\s*"([^"]+)"/g;
const wholeValueDomainPattern =
  /^(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,}$/i;
const domainTokenPattern =
  /\b(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,}\b/gi;

const phonePattern =
  /(?:tel:)?(?:\+?[0-9]{1,4}[\s\-\.\/]?\(?[0-9]{1,4}\)?[\s\-\.\/]?[0-9]{2,4}[\s\-\.\/]?[0-9]{2,9}|\b0[2-9][\s\-]?[0-9]{3}[\s\-]?[0-9]{4}\b|\b\(?[0-9]{3}\)?[\s\-\.][0-9]{3}[\s\-\.][0-9]{4}\b)/g;

const isDatePattern =
  /^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}$|^\d{1,2}[-/.]\d{1,2}[-/.]\d{4}$/;
const isIpPattern = /^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$/;
const isTimePattern = /^\d{1,2}:\d{2}(?::\d{2})?$/;

const ccTlds = new Set([
  "co.uk",
  "org.uk",
  "gov.uk",
  "ac.uk",
  "com.au",
  "net.au",
  "org.au",
  "edu.au",
  "co.il",
  "org.il",
  "net.il",
  "co.jp",
  "ne.jp",
  "co.nz",
  "com.br",
]);

const cleanTrailingPunctuation = (urlStr: string): string =>
  urlStr.replace(/[.,;:!?)]+$/, "");

export const isPlaceholderHost = (rawHost: string): boolean => {
  const host = rawHost.toLowerCase().replace(/^www\./, "");
  if (
    host.endsWith(".example") ||
    host.endsWith(".test") ||
    host.endsWith(".invalid") ||
    host.endsWith(".localhost") ||
    host === "localhost" ||
    host === "127.0.0.1"
  ) {
    return true;
  }
  const parts = host.split(".");
  const apex = parts.slice(-2).join(".");
  if (
    apex === "example.com" ||
    apex === "example.org" ||
    apex === "example.net" ||
    apex === "example.edu"
  ) {
    return true;
  }
  return /(?:^|[.-])(?:example|sample|yourdomain|your-domain|yourcompany|your-company|my-domain|mysite|placeholder)(?:[.-]|$)/i
    .test(host);
};

export const isComplexUrl = (rawUrl: string): boolean => {
  try {
    const url = new URL(cleanTrailingPunctuation(rawUrl));
    if (isPlaceholderHost(url.hostname)) return false;
    const path = url.pathname.replace(/\/+$/, "");
    if (path !== "") return true;
    if (url.search && url.search !== "") return true;
    if (
      url.port && url.port !== "" && url.port !== "80" && url.port !== "443"
    ) {
      return true;
    }
    const host = url.hostname.toLowerCase();
    const parts = host.split(".");
    if (parts.length <= 2) return false;
    if (parts[0] === "www" && parts.length === 3) return false;
    const isTwoPartTld = parts.length >= 3 &&
      ccTlds.has(`${parts[parts.length - 2]}.${parts[parts.length - 1]}`);
    if (isTwoPartTld && parts[0] === "www" && parts.length === 4) return false;
    if (isTwoPartTld && parts.length === 3) return false;
    return true;
  } catch {
    return false;
  }
};

const normalizeHost = (host: string): string =>
  host.toLowerCase().replace(/^www\./, "");

const hostsFromUrls = (text: string): string[] =>
  [...text.matchAll(urlHostPattern)].map((m) => normalizeHost(m[1]));

const hostsFromAssignments = (text: string): string[] =>
  [...text.matchAll(hostAssignmentPattern)].map((m) => m[1]).filter((v) =>
    v.includes(".")
  ).map(normalizeHost);

// Strict extraction for tool-call parameters: a URL buried inside generated
// content (HTML, documents, messages the model authors) is content, not a
// request target — the model legitimately references well-known hosts there.
// Only whole-value URLs, safescript-style `host: "..."` literals, and values
// that are entirely a domain count as call targets.
const wholeValueUrlPattern =
  /^https?:\/\/([a-z0-9](?:[a-z0-9.-]*[a-z0-9])?)\S*$/i;

const hostsInParamString = (text: string): string[] => [
  ...(wholeValueUrlPattern.exec(text.trim())?.slice(1).map(normalizeHost) ??
    []),
  ...hostsFromAssignments(text),
  ...(wholeValueDomainPattern.test(text.trim())
    ? [normalizeHost(text.trim())]
    : []),
];

// Loose extraction for ground-truth text: documentation and messages mention
// hosts in prose without a scheme, and those mentions legitimately ground
// later calls.
const hostsInGroundTruthText = (text: string): string[] => [
  ...hostsFromUrls(text),
  ...hostsFromAssignments(text),
  ...[...text.matchAll(domainTokenPattern)].map((m) => normalizeHost(m[0])),
];

const stringValues = (value: unknown): string[] =>
  typeof value === "string"
    ? [value]
    : Array.isArray(value)
    ? value.flatMap(stringValues)
    : value !== null && typeof value === "object"
    ? Object.values(value).flatMap(stringValues)
    : [];

const innerCommandName = (parameters: unknown): string[] => {
  if (typeof parameters !== "object" || parameters === null) return [];
  if (!("command" in parameters)) return [];
  const { command } = parameters;
  return typeof command === "string" ? [command] : [];
};

const isExemptToolCall =
  (exemptToolNames: string[]) => (e: HistoryEvent): boolean =>
    e.type === "tool_call" &&
    [e.name, ...innerCommandName(e.parameters)].some((name) =>
      exemptToolNames.includes(name)
    );

const toolCallParamHosts = (e: HistoryEvent): string[] =>
  e.type === "tool_call"
    ? stringValues(e.parameters).flatMap(hostsInParamString)
    : [];

export const findUngroundedToolCallHosts = (
  groundTruthTexts: string[],
  exemptToolNames: string[],
  events: HistoryEvent[],
): string[] => {
  const grounded = new Set(groundTruthTexts.flatMap(hostsInGroundTruthText));
  return pipe(
    filter((e: HistoryEvent) => !isExemptToolCall(exemptToolNames)(e)),
    map(toolCallParamHosts),
    (nested: string[][]) => [...new Set(nested.flat())],
    filter((host: string) => !grounded.has(host)),
  )(events);
};

export const ungroundedHostBlockedNotice = (hosts: string[]): string =>
  `A tool call you attempted was blocked and not executed: it targeted ${
    hosts.map((h) => `"${h}"`).join(", ")
  }, and none of these hosts appear anywhere in your instructions, tool documentation, user messages, or tool results. Never invent URLs, hosts, or API endpoints — use only endpoints documented in your instructions or ones that already appeared in the conversation. If no documented endpoint can fulfill the request, say so honestly or ask the user for the correct URL instead of guessing.`;

const extractUrls = (text: string): string[] =>
  [...text.matchAll(urlPattern)].map((m) => cleanTrailingPunctuation(m[0]));

const normalizeUrlForMatch = (urlStr: string): string => {
  try {
    const url = new URL(cleanTrailingPunctuation(urlStr));
    return `${url.hostname}${url.pathname.replace(/\/+$/, "")}${url.search}`
      .toLowerCase();
  } catch {
    return urlStr.toLowerCase();
  }
};

const cleanPhoneCandidate = (candidate: string): string =>
  candidate.replace(/^tel:/i, "").replace(/[.,;:!?)]+$/, "").trim();

const extractPhoneDigits = (phoneStr: string): string =>
  phoneStr.replace(/\D/g, "");

const stripCodeBlocks = (text: string): string =>
  text
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/`[^`\n]+`/g, " ");

const isPlaceholderPhone = (phoneStr: string): boolean => {
  const digits = phoneStr.replace(/\D/g, "");
  if (
    digits.includes("55501") ||
    digits.startsWith("555") ||
    digits.startsWith("1555")
  ) {
    return true;
  }
  if (/^(\d)\1+$/.test(digits)) return true;
  if ("0123456789012345".includes(digits)) return true;
  return false;
};

const isJustifiedInThoughts = (
  modelThoughts: string[],
  urlOrPhone: string,
): boolean => {
  if (!modelThoughts || modelThoughts.length === 0) return false;
  const combinedThoughts = modelThoughts.join("\n").toLowerCase();
  const hasExampleIntent =
    /\b(example|illustrat|sample|educational|placeholder|hypothetical|demonstrat|fictional)\b/i
      .test(combinedThoughts);
  if (!hasExampleIntent) return false;

  const cleaned = cleanTrailingPunctuation(urlOrPhone).toLowerCase();
  try {
    const url = new URL(cleaned);
    const host = url.hostname.toLowerCase();
    if (combinedThoughts.includes(host)) return true;
    if (combinedThoughts.includes(cleaned)) return true;
    const parts = host.split(".");
    if (
      parts.length >= 2 && combinedThoughts.includes(parts.slice(-2).join("."))
    ) {
      return true;
    }
    if (
      parts.filter((p) =>
        p.length >= 3 && p !== "www" && p !== "com" && p !== "org" &&
        p !== "net" && p !== "api"
      ).some((p) => combinedThoughts.includes(p))
    ) {
      return true;
    }
  } catch {
    if (combinedThoughts.includes(cleaned)) return true;
  }
  return false;
};

export const isLikelyPhoneNumber = (candidate: string): boolean => {
  const cleaned = cleanPhoneCandidate(candidate);
  if (isPlaceholderPhone(cleaned)) return false;
  if (isDatePattern.test(cleaned)) return false;
  if (isIpPattern.test(cleaned)) return false;
  if (isTimePattern.test(cleaned)) return false;
  const digits = extractPhoneDigits(cleaned);
  if (digits.length < 7 || digits.length > 15) return false;
  const hasFormatting = /[+\s\-\(\)\/]/.test(cleaned) ||
    candidate.startsWith("tel:");
  if (!hasFormatting && digits.length < 10) return false;
  return true;
};

const extractPhoneCandidates = (text: string): string[] =>
  [...text.matchAll(phonePattern)]
    .map((m) => m[0])
    .filter(isLikelyPhoneNumber);

export type UngroundedUtteranceArtifacts = {
  ungroundedUrls: string[];
  ungroundedPhones: string[];
};

export const findUngroundedUtteranceArtifacts = (
  groundTruthTexts: string[],
  utteranceTexts: string[],
  modelThoughts: string[] = [],
): UngroundedUtteranceArtifacts => {
  const combinedUtterances = utteranceTexts.map(stripCodeBlocks).join("\n");
  const combinedGroundTruth = groundTruthTexts.join("\n");
  const combinedGroundTruthNormalized = combinedGroundTruth.toLowerCase();
  const groundTruthDigits = extractPhoneDigits(combinedGroundTruth);

  const rawUrls = extractUrls(combinedUtterances);
  const complexUrls = [...new Set(rawUrls.filter(isComplexUrl))];
  const ungroundedUrls = complexUrls.filter((url) => {
    if (isJustifiedInThoughts(modelThoughts, url)) return false;
    const cleaned = cleanTrailingPunctuation(url);
    if (combinedGroundTruth.includes(cleaned)) return false;
    const normalized = normalizeUrlForMatch(cleaned);
    return !combinedGroundTruthNormalized.includes(normalized);
  });

  const rawPhones = extractPhoneCandidates(combinedUtterances);
  const ungroundedPhones = [...new Set(rawPhones)].filter((phone) => {
    if (isJustifiedInThoughts(modelThoughts, phone)) return false;
    const cleaned = cleanPhoneCandidate(phone);
    if (combinedGroundTruth.includes(cleaned)) return false;
    const digits = extractPhoneDigits(cleaned);
    if (digits && groundTruthDigits.includes(digits)) return false;
    return true;
  });

  return {
    ungroundedUrls,
    ungroundedPhones,
  };
};

export const ungroundedUtteranceBlockedNotice = (
  artifacts: UngroundedUtteranceArtifacts,
): string => {
  const parts = [
    ...(nonempty(artifacts.ungroundedUrls)
      ? [
        `unverified URL(s): ${
          artifacts.ungroundedUrls.map((u) => `"${u}"`).join(", ")
        }`,
      ]
      : []),
    ...(nonempty(artifacts.ungroundedPhones)
      ? [
        `unverified phone number(s): ${
          artifacts.ungroundedPhones.map((p) => `"${p}"`).join(", ")
        }`,
      ]
      : []),
  ];
  return `Your previous draft reply was not sent to the user: it contained ${
    parts.join(" and ")
  } that do not appear anywhere in the conversation history, instructions, or tool results. Do not fabricate specific URLs or phone numbers. Only share URLs or phone numbers that were returned by a tool or provided in your instructions. Please rewrite your reply without them, or use tools to look them up.`;
};
