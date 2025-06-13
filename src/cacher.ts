import { context } from "context-inject";
import type { Func } from "gamla";

const injection: {
  inject: (
    fn: (_cacheId: string) => <F extends Func>(_f: F) => Func,
  ) => <F_1 extends Func>(f: F_1) => F_1;
  access: (_cacheId: string) => <F extends Func>(_f: F) => Func;
} = context(
  (_cacheId: string) => <F extends Func>(_f: F): Func => {
    throw new Error("cacher not injected");
  },
);

export const injectCacher = injection.inject;

export const makeCache = injection.access;
