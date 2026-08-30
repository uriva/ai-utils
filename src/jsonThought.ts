// The bracketed internal-thought marker the platform renders for
// metadata-less model-role thoughts. Shared by the agent-loop reclassifier
// (anchored, whole-text) and the Gemini part converter (global, embedded).
export const internalThoughtPrefix = "[Internal thought, visible only to you:";

export const formatInternalThought = (text: string): string =>
  `${internalThoughtPrefix} ${text}]`;

export const internalThoughtMarker =
  "\\[Internal thought, visible only to you: ([\\s\\S]*?)\\]";

export const internalThoughtRegex = new RegExp(`^${internalThoughtMarker}$`);

export const jsonThoughtPattern = /\{\s*"thought"\s*:\s*"([\s\S]*?)"\s*\}\s*/gi;

export const stripJsonThought = (text: string): string =>
  text.replace(jsonThoughtPattern, "").trim();

export const extractJsonThought = (text: string): string =>
  [...text.matchAll(jsonThoughtPattern)]
    .map((m) => m[1])
    .join("\n")
    .trim();

export const hasJsonThought = (text: string): boolean => {
  const result = jsonThoughtPattern.test(text);
  jsonThoughtPattern.lastIndex = 0; // Reset lastIndex due to g flag
  return result;
};
