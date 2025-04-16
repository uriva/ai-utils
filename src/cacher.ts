import { context } from "https://deno.land/x/context_inject@0.0.3/src/index.ts";
import { cache } from "npm:rmmbr@0.0.21";

const injection = context((): string => {
    throw new Error("rmmbr token not injected");
});

export const injectRmmbrToken = injection.inject;

export const makeCache = (cacheId: string) =>
    cache({
        cacheId,
        ttl: 60 * 60 * 24 * 14,
        url: "https://rmmbr.net",
        token: injection.access(),
    });
