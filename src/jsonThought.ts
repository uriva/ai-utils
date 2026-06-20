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
