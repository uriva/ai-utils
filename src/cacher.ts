import { context, type Injection, type Injector } from "@uri/inject";

const injection: Injection<(cacheId: string) => Injector> = context(
  (_cacheId) => (((_f) => {
    throw new Error("cacher not injected");
  }) as Injector),
);

export const injectCacher = injection.inject;

export const makeCache = injection.access;
