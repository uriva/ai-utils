const englishTimestampFormatter = (timezoneIANA: string) =>
  new Intl.DateTimeFormat("en-US", {
    timeZone: timezoneIANA,
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  });

export const formatInternalSentTimestamp = (
  ts: number,
  timezoneIANA: string,
): string => englishTimestampFormatter(timezoneIANA).format(new Date(ts));

const internalSentPrefix = " — sent ";

const internalSentTimestampShape =
  /(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4},\s+\d{1,2}:\d{2}\s+(AM|PM)/;

export const appendInternalSentTimestamp = (
  text: string,
  ts: number,
  timezoneIANA: string,
): string =>
  `${text}${internalSentPrefix}${
    formatInternalSentTimestamp(ts, timezoneIANA)
  }`;

const flexibleInternalTimestampPattern = new RegExp(
  `\\s?[—–-]\\s?sent\\s+${internalSentTimestampShape.source}`,
);

export const hasInternalSentTimestampSuffix = (
  text: string,
): boolean => flexibleInternalTimestampPattern.test(text);

export const stripInternalSentTimestampSuffix = (
  text: string,
): string => text.replace(flexibleInternalTimestampPattern, "");
