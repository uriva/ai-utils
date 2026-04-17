import { context, type Injection, type Injector } from "@uri/inject";

const injection: Injection<(cacheId: string) => Injector> = context(
  (_cacheId) => (((f) => f) as Injector),
);

export const injectCacher = injection.inject;

export const makeCache = injection.access;
