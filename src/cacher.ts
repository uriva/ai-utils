import { context } from "context-inject";
import type { Func } from "gamla";

const injection = context(
  (_cacheId: string) => <F extends Func>(_f: F): Func => {
    throw new Error("cacher not injected");
  },
);

export const injectCacher = injection.inject;

export const makeCache = injection.access;
