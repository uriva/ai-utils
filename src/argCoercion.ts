// deno-lint-ignore-file no-explicit-any
// Coerces tool-call arguments toward an intended JSON Schema shape when there
// is exactly one reasonable interpretation. Returns the (possibly rewritten)
// args plus a list of human-readable correction descriptions. Empty list means
// the args were already canonical (or ambiguous, in which case nothing was
// changed).

const normalize = (s: string) => s.toLowerCase().replace(/[_\-\s]/g, "");

const isPlainObject = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v);

type Path = string[];

const schemaProperties = (node: any): Record<string, any> | undefined => {
  if (!node || typeof node !== "object") return undefined;
  if (node.properties) return node.properties;
  if (Array.isArray(node.anyOf)) {
    const merged: Record<string, any> = {};
    node.anyOf.forEach((b: any) => {
      const p = schemaProperties(b);
      if (p) Object.assign(merged, p);
    });
    return Object.keys(merged).length ? merged : undefined;
  }
  return undefined;
};

const collectPaths = (
  schema: any,
  prefix: Path,
  acc: Map<string, Path[]>,
) => {
  const props = schemaProperties(schema);
  if (!props) return;
  Object.entries(props).forEach(([key, child]) => {
    const path = [...prefix, key];
    const norm = normalize(key);
    const list = acc.get(norm) ?? [];
    list.push(path);
    acc.set(norm, list);
    collectPaths(child, path, acc);
  });
};

const propertyAt = (schema: any, path: Path): any | undefined => {
  let cursor: any = schema;
  for (const seg of path) {
    const props = schemaProperties(cursor);
    if (!props || !(seg in props)) return undefined;
    cursor = props[seg];
  }
  return cursor;
};

const hasAtPath = (root: any, path: Path): boolean => {
  let cursor: any = root;
  for (const seg of path) {
    if (!isPlainObject(cursor) || !(seg in cursor)) return false;
    cursor = cursor[seg];
  }
  return true;
};

const setAtPath = (root: any, path: Path, value: unknown): any => {
  if (path.length === 0) return value;
  const [head, ...rest] = path;
  const child = isPlainObject(root) && isPlainObject(root[head])
    ? root[head]
    : {};
  return {
    ...(isPlainObject(root) ? root : {}),
    [head]: setAtPath(child, rest, value),
  };
};

const removeAtPath = (root: any, path: Path): any => {
  if (path.length === 0 || !isPlainObject(root)) return root;
  const [head, ...rest] = path;
  if (!(head in root)) return root;
  if (rest.length === 0) {
    const next = { ...root };
    delete next[head];
    return next;
  }
  return { ...root, [head]: removeAtPath(root[head], rest) };
};

type Rewrite = { from: Path; to: Path; value: unknown };

// Finds the first key in `args` (depth-first) that is not valid at its current
// schema path and has exactly one unoccupied destination in the schema.
const findRewrite = (
  schema: any,
  rootArgs: any,
  pathIndex: Map<string, Path[]>,
  args: any,
  schemaPath: Path,
): Rewrite | null => {
  if (!isPlainObject(args)) return null;
  for (const [key, value] of Object.entries(args)) {
    const here = [...schemaPath, key];
    if (propertyAt(schema, here) !== undefined) {
      const deeper = findRewrite(schema, rootArgs, pathIndex, value, here);
      if (deeper) return deeper;
      continue;
    }
    const candidates = pathIndex.get(normalize(key)) ?? [];
    const free = candidates.filter((p) => !hasAtPath(rootArgs, p));
    if (free.length !== 1) continue;
    return { from: here, to: free[0], value };
  }
  return null;
};

const correctionFor = ({ from, to }: Rewrite): string =>
  from.join(".") === to.join(".")
    ? `renamed "${from.join(".")}" (case/style)`
    : `moved "${from.join(".")}" to "${to.join(".")}"`;

const applyRewrite = (root: any, rewrite: Rewrite): any =>
  setAtPath(removeAtPath(root, rewrite.from), rewrite.to, rewrite.value);

export const coerceArgs = (
  schema: any,
  args: unknown,
): { args: unknown; corrections: string[] } => {
  if (args === undefined || args === null) {
    if (schema?.type === "object" || schemaProperties(schema) !== undefined) {
      args = {};
    }
  }
  if (!isPlainObject(args)) return { args, corrections: [] };
  const pathIndex = new Map<string, Path[]>();
  collectPaths(schema, [], pathIndex);
  let current: any = args;
  const corrections: string[] = [];
  for (let i = 0; i < 32; i++) {
    const rewrite = findRewrite(schema, current, pathIndex, current, []);
    if (!rewrite) break;
    current = applyRewrite(current, rewrite);
    corrections.push(correctionFor(rewrite));
  }
  return { args: current, corrections };
};
