import { unique } from "gamla";

const urlPattern = /https?:\/\/[^\s"'`<>]+/gi;

const splitPattern = /[^A-Za-z0-9_-]+/;

const uuidPattern =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

const lettersAndDigitsPattern = /^(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9_-]+$/;

const longDigitsPattern = /^\d{5,}$/;

const encodedLikePattern = /^(?=.*\d)[A-Za-z0-9_-]{8,}$/;

const canParseUrl = (value: string) => {
  try {
    new URL(value);
    return true;
  } catch {
    return false;
  }
};

const normalizeToken = (token: string) =>
  token.replace(/^[^A-Za-z0-9]+|[^A-Za-z0-9]+$/g, "");

const isOpaqueIdentifier = (token: string) => {
  // Ignore short date-like strings (e.g. "mar-22", "22-jan", "2024-05-12")
  const isDateLike = /^[a-zA-Z]{3}-\d{1,2}$/i.test(token) ||
    /^\d{1,2}-[a-zA-Z]{3}$/i.test(token) ||
    /^\d{4}-\d{2}-\d{2}$/.test(token);
  if (isDateLike) return false;

  if (uuidPattern.test(token)) return true;
  if (longDigitsPattern.test(token)) return true;
  if (lettersAndDigitsPattern.test(token) && token.length >= 6) return true;
  return encodedLikePattern.test(token) && /[_-]/.test(token);
};

const splitIntoTokens = (value: string) =>
  value.split(splitPattern).map(normalizeToken).filter(Boolean);

const urlValues = (value: string) => {
  if (!canParseUrl(value)) return splitIntoTokens(value);
  const url = new URL(value);
  return [
    ...url.pathname.split("/"),
    ...url.searchParams.values().flatMap(splitIntoTokens),
    ...splitIntoTokens(url.hash),
  ].map(normalizeToken).filter(Boolean);
};

const extractStrings = (value: unknown): string[] => {
  if (typeof value === "string") return [value];
  if (Array.isArray(value)) return value.flatMap(extractStrings);
  if (value && typeof value === "object") {
    return Object.values(value).flatMap(extractStrings);
  }
  return [];
};

const extractUrlTokens = (value: string) =>
  (value.match(urlPattern) ?? []).flatMap(urlValues);

const removeUrls = (value: string) => value.replace(urlPattern, " ");

export const extractOpaqueIdentifiers = (value: unknown): string[] =>
  unique(
    extractStrings(value)
      .flatMap((singleValue) => [
        ...extractUrlTokens(singleValue),
        ...splitIntoTokens(removeUrls(singleValue)),
      ])
      .filter(isOpaqueIdentifier),
  );

export const findNovelOpaqueIdentifiers = (
  candidate: unknown,
  knownSources: unknown[],
): string[] => {
  const known = new Set(knownSources.flatMap(extractOpaqueIdentifiers));
  return extractOpaqueIdentifiers(candidate).filter((token) =>
    !known.has(token)
  );
};
