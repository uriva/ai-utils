import { context, type Injection, type Injector } from "@uri/inject";

const injection: Injection<
  (cacheId: string, ttlSeconds?: number) => Injector
> = context(
  (_cacheId, _ttlSeconds) => (((f) => f) as Injector),
);

export const injectCacher = injection.inject;

export const makeCache = injection.access;
