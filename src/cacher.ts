import { context, type Injection, type Injector } from "@uri/inject";

const defaultCacher = (_cacheId: string, _ttlSeconds?: number): Injector =>
  ((f) => f) as Injector;

const injection: Injection<
  (cacheId: string, ttlSeconds?: number) => Injector
> = context(defaultCacher);

export const injectCacher = injection.inject;

export const makeCache = injection.access;

export const isCustomCacherInjected = (): boolean =>
  injection.getStore() !== defaultCacher;
