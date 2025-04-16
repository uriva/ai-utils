import { context } from "npm:context-inject@0.0.3";
import { cache } from "npm:rmmbr@0.0.21";

// deno-lint-ignore no-explicit-any
type Func = (...xs: any[]) => any;

const injection: {
    inject: (fn: () => string) => <F extends Func>(f: F) => F;
    access: () => string;
} = context((): string => {
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
