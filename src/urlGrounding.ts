import { pipe } from "gamla";
import { filter, map } from "gamla";
import type { HistoryEvent } from "./agent.ts";

// Models sometimes fabricate URLs/hosts inside tool-call parameters (guessed
// API endpoints) when the capability they want is missing or a prior tool
// failed. The utterance grounding gate only audits concluding replies, so
// these invented endpoints enter history through tool calls unchecked. This
// module detects hosts in tool-call parameters that are traceable to no
// ground truth (instructions, tool documentation, user messages, tool
// results) so the agent loop can reject the call and make the model
// self-correct before anything executes.

const urlHostPattern = /https?:\/\/([a-z0-9](?:[a-z0-9.-]*[a-z0-9])?)/gi;
const hostAssignmentPattern = /\bhost\s*:\s*"([^"]+)"/g;
const wholeValueDomainPattern =
  /^(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,}$/i;
const domainTokenPattern =
  /\b(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,}\b/gi;

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
