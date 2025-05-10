import { context } from "context-inject";
import { cache } from "rmmbr";
import { FnToSameFn, TokenInjection } from "./utils.ts";

const injection: TokenInjection = context((): string => {
  throw new Error("rmmbr token not injected");
});

export const injectRmmbrToken = (x: string): FnToSameFn =>
  injection.inject(() => x);

export const makeCache = (cacheId: string) =>
  cache({
    cacheId,
    ttl: 60 * 60 * 24 * 14,
    url: "https://rmmbr.net",
    token: injection.access(),
  });
