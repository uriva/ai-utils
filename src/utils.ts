import { type EitherOutput, type Func, throwerCatcher } from "gamla";

export type ModelOpts = { mini: boolean };

const typeAdherenceError = throwerCatcher();

export const aiRefusesToAdhereTyping = typeAdherenceError.thrower;
export const catchAiRefusesToAdhereToTyping: <G extends Func>(
  fallback: G,
) => <F extends Func>(f: F) => (...xs: Parameters<F>) => EitherOutput<F, G> =
  typeAdherenceError.catcher;

export type SomethingInjection<T extends Func> = {
  inject: (fn: T) => FnToSameFn;
  access: T;
};

export type TokenInjection = SomethingInjection<() => string>;

export type FnToSameFn = <F extends Func>(f: F) => F;
