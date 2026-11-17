import { context, type Injection, type Injector } from "@uri/inject";
import { map, pipe, prop } from "gamla";
import { OpenAI } from "openai";
import type {
  ChatCompletionCreateParamsNonStreaming,
  ChatCompletionMessageParam,
} from "openai/resources/index.mjs";
import z, { type ZodType } from "zod/v4";
import { makeCache } from "./cacher.ts";
import { aiRefusesToAdhereTyping, type ModelOpts } from "./utils.ts";
import { genJson } from "./genJson.ts";

const tokenInjection: Injection<() => string> = context((): string => {
  throw new Error("no openai token injected");
});

export const injectOpenAiToken = (token: string): Injector =>
  tokenInjection.inject(() => token);

type JsonSchemaNode = Record<string, unknown>;

const isPlainObject = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v);

const resolveSchemaRef = (
  node: JsonSchemaNode,
  root: JsonSchemaNode,
): JsonSchemaNode => {
  if (typeof node.$ref !== "string") return node;
  const ref = node.$ref;
  const defsMatch = /^#\/(\$defs|definitions)\/(.+)$/.exec(ref);
  if (!defsMatch) return node;
  const [, containerKey, defName] = defsMatch;
  const container = root[containerKey];
  if (isPlainObject(container) && isPlainObject(container[defName])) {
    return container[defName];
  }
  return node;
};

const schemaAcceptsNull = (
  rawNode: unknown,
  root: JsonSchemaNode,
): boolean => {
  if (!isPlainObject(rawNode)) return false;
  const node = resolveSchemaRef(rawNode, root);
  if (node.type === "null") return true;
  if (Array.isArray(node.type) && node.type.includes("null")) return true;
  if (
    Array.isArray(node.anyOf) &&
    node.anyOf.some((branch) => schemaAcceptsNull(branch, root))
  ) {
    return true;
  }
  if (
    Array.isArray(node.oneOf) &&
    node.oneOf.some((branch) => schemaAcceptsNull(branch, root))
  ) {
    return true;
  }
  return false;
};

const makeNullableSchema = (
  node: JsonSchemaNode,
  root: JsonSchemaNode,
): JsonSchemaNode => {
  if (schemaAcceptsNull(node, root)) return node;
  return {
    ...(typeof node.description === "string"
      ? { description: node.description }
      : {}),
    anyOf: [node, { type: "null" }],
  };
};

const transformStrictNode = (
  rawNode: unknown,
  root: JsonSchemaNode,
): unknown => {
  if (!isPlainObject(rawNode)) return rawNode;
  const next: JsonSchemaNode = { ...rawNode };

  for (const defsKey of ["$defs", "definitions"] as const) {
    if (isPlainObject(next[defsKey])) {
      next[defsKey] = Object.fromEntries(
        Object.entries(next[defsKey] as Record<string, unknown>).map((
          [k, v],
        ) => [k, transformStrictNode(v, root)]),
      );
    }
  }

  for (const branchKey of ["anyOf", "oneOf", "allOf"] as const) {
    if (Array.isArray(next[branchKey])) {
      next[branchKey] = (next[branchKey] as unknown[]).map((b) =>
        transformStrictNode(b, root)
      );
    }
  }

  if (Array.isArray(next.prefixItems)) {
    next.prefixItems = (next.prefixItems as unknown[]).map((item) =>
      transformStrictNode(item, root)
    );
  }
  if (isPlainObject(next.items)) {
    next.items = transformStrictNode(next.items, root);
  }

  if (isPlainObject(next.properties)) {
    const props = next.properties as Record<string, unknown>;
    const origRequired = new Set(
      Array.isArray(next.required)
        ? (next.required as unknown[]).filter(
          (k): k is string => typeof k === "string",
        )
        : [],
    );
    const strictProps: Record<string, unknown> = {};
    for (const [key, propVal] of Object.entries(props)) {
      const transformedProp = transformStrictNode(propVal, root);
      strictProps[key] = !origRequired.has(key) &&
          isPlainObject(transformedProp)
        ? makeNullableSchema(transformedProp, root)
        : transformedProp;
    }
    next.properties = strictProps;
    next.required = Object.keys(props);
    next.additionalProperties = false;
  } else if (next.type === "object" && !("additionalProperties" in next)) {
    next.additionalProperties = false;
  }

  return next;
};

export const toOpenAiStrictSchema = (
  rawSchema: JsonSchemaNode,
): JsonSchemaNode =>
  transformStrictNode(rawSchema, rawSchema) as JsonSchemaNode;

const restoreNodeUndefineds = (
  value: unknown,
  rawNode: unknown,
  root: JsonSchemaNode,
): unknown => {
  if (!isPlainObject(rawNode) || value === null || value === undefined) {
    return value;
  }
  const node = resolveSchemaRef(rawNode, root);

  if (Array.isArray(value)) {
    const prefixItems = Array.isArray(node.prefixItems)
      ? (node.prefixItems as unknown[])
      : undefined;
    return value.map((elem, idx) =>
      restoreNodeUndefineds(
        elem,
        prefixItems && idx < prefixItems.length ? prefixItems[idx] : node.items,
        root,
      )
    );
  }

  if (isPlainObject(value) && isPlainObject(node.properties)) {
    const props = node.properties as Record<string, unknown>;
    const origRequired = new Set(
      Array.isArray(node.required)
        ? (node.required as unknown[]).filter(
          (k): k is string => typeof k === "string",
        )
        : [],
    );
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value)) {
      const propSchema = props[k];
      if (
        v === null &&
        isPlainObject(propSchema) &&
        !origRequired.has(k) &&
        !schemaAcceptsNull(propSchema, root)
      ) {
        continue;
      }
      out[k] = propSchema !== undefined
        ? restoreNodeUndefineds(v, propSchema, root)
        : v;
    }
    return out;
  }

  for (const unionKey of ["anyOf", "oneOf"] as const) {
    if (Array.isArray(node[unionKey]) && isPlainObject(value)) {
      const matchingBranch = (node[unionKey] as unknown[]).find(
        (b) => isPlainObject(b) && isPlainObject(b.properties),
      );
      if (matchingBranch) {
        return restoreNodeUndefineds(value, matchingBranch, root);
      }
    }
  }

  return value;
};

export const restoreOptionalNullsToUndefined = (
  value: unknown,
  rawSchema: JsonSchemaNode,
): unknown => restoreNodeUndefineds(value, rawSchema, rawSchema);

// deno-lint-ignore no-explicit-any
export const openAiGenJsonFromConvo = async <T extends ZodType<any, any, any>>(
  { tier = "flash" }: ModelOpts,
  messages: ChatCompletionMessageParam[],
  zodType: T,
): Promise<z.infer<T>> => {
  const cachedCall = makeCache(
    "openAiTypedCompletion",
  )((opts: ChatCompletionCreateParamsNonStreaming) =>
    new OpenAI({ apiKey: tokenInjection.access() }).chat.completions
      .parse(opts)
  );
  const rawSchema = z.toJSONSchema(zodType) as JsonSchemaNode;
  const { choices } = await cachedCall({
    model: tier === "pro" ? "gpt-5" : "gpt-5-mini",
    messages,
    response_format: {
      type: "json_schema",
      json_schema: {
        name: "structured-response",
        strict: true,
        schema: toOpenAiStrictSchema(rawSchema),
      },
    },
  });
  if (!choices[0].message.content) {
    aiRefusesToAdhereTyping();
    throw new Error("AI refused to return content");
  }
  return restoreOptionalNullsToUndefined(
    JSON.parse(choices[0].message.content),
    rawSchema,
  ) as z.infer<T>;
};

export const structuredMsgs = (
  systemMsg: string,
  userMsg: string,
): ChatCompletionMessageParam[] => [
  { role: "system", content: systemMsg },
  { role: "user", content: userMsg },
];

const numberedItem = (x: string, i: number) => `(${i}): ${x}`;

const makePromptForMatching = (
  setA: string[],
  setB: string[],
  reasoningNotes: string,
) =>
  `You need to match between the following groups.

Group A:

${setA.map(numberedItem).join("\n\n")}


Group B:

${setB.map(numberedItem).join("\n\n")}


Notes for reasoning: ${reasoningNotes}

The output should be an index from group A, then index from group B then reasoning (why you think this is a good match). please double check the indices.`;

const OutputZod = z.object({
  results: z.array(z.tuple([z.number(), z.number(), z.string()])),
});

type Output = z.infer<typeof OutputZod>;

export const openAiMatching = <X, Y>(
  opts: ModelOpts,
  reasoningNotes: string,
  xToStr: (y: X) => string,
  yToStr: (y: Y) => string,
) =>
(xs: X[], ys: Y[]): Promise<[X, Y, string][]> =>
  pipe(
    makePromptForMatching,
    genJson({ ...opts, provider: "openai" }, "", OutputZod),
    prop<Output>()("results"),
    map(([indA, indB, reasoning]: Output["results"][0]) =>
      [xs[indA], ys[indB], reasoning] satisfies [X, Y, string]
    ),
  )(xs.map(xToStr), ys.map(yToStr), reasoningNotes);
