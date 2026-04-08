import { type EitherOutput, type Func, throwerCatcher } from "gamla";

export type ModelOpts = { mini: boolean; maxOutputTokens?: number };

const typeAdherenceError = throwerCatcher();

export const aiRefusesToAdhereTyping = typeAdherenceError.thrower;
export const catchAiRefusesToAdhereToTyping: <G extends Func>(
  fallback: G,
) => <F extends Func>(f: F) => (...xs: Parameters<F>) => EitherOutput<F, G> =
  typeAdherenceError.catcher;

export const normalizeError = (error: unknown): Error => {
  if (error instanceof Error) return error;
  if (typeof error === "string") return new Error(error);
  if (typeof error === "object" && error !== null) {
    const err = new Error(
      (error as { message?: string }).message || JSON.stringify(error),
    );
    Object.assign(err, error);
    return err;
  }
  return new Error(String(error));
};

const errorStatus = (error: unknown) =>
  error instanceof Error && "status" in error
    ? (error as { status: number }).status
    : undefined;

export const isServerError = (error: unknown) =>
  (errorStatus(error) ?? 0) >= 500;

export const isRateLimitError = (error: unknown) => errorStatus(error) === 429;

export const isRetryableError = (error: unknown) =>
  isServerError(error) || isRateLimitError(error);
