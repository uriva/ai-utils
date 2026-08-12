import { z, type ZodType } from "zod/v4";

// Fields marked with `.default()` in Zod end up in the JSON Schema `required`
// array. That's technically fine per JSON Schema semantics, but our downstream
// consumers (tool-typing strings, strict validators on prompt2bot) treat
// `required` as "caller must provide". Strip those fields from `required` so
// they're presented as optional to the model and to strict validators.
// deno-lint-ignore no-explicit-any
export const pruneDefaultsFromRequired = (schema: any) => {
  if (!schema?.properties || !Array.isArray(schema.required)) return schema;
  // deno-lint-ignore no-explicit-any
  const withDefault = (key: string) => (schema.properties[key] as any)?.default;
  return {
    ...schema,
    required: schema.required.filter((k: string) =>
      withDefault(k) === undefined
    ),
  };
};

const jsonSchemaNodeToTyping = (
  includeDescriptions: boolean,
  // deno-lint-ignore no-explicit-any
  node: any,
): string => {
  if (node.enum) return node.enum.map((v: string) => `"${v}"`).join(" | ");
  if (node.anyOf) {
    // deno-lint-ignore no-explicit-any
    return node.anyOf.map((child: any) =>
      jsonSchemaNodeToTyping(includeDescriptions, child)
    ).join(" | ");
  }
  if (node.type === "array") {
    return `${
      jsonSchemaNodeToTyping(
        includeDescriptions,
        node.items || { type: "unknown" },
      )
    }[]`;
  }
  if (node.type === "object" && node.properties) {
    return jsonSchemaObjectToTyping(includeDescriptions, node);
  }
  return node.type || "unknown";
};

const jsonSchemaObjectToTyping = (
  includeDescriptions: boolean,
  // deno-lint-ignore no-explicit-any
  schema: any,
): string => {
  const pruned = pruneDefaultsFromRequired(schema);
  const required = new Set(pruned.required || []);
  const entries = Object.entries(pruned.properties || {}).map(
    // deno-lint-ignore no-explicit-any
    ([key, prop]: [string, any]) => {
      const opt = required.has(key) ? "" : "?";
      const desc = includeDescriptions && prop.description
        ? ` /* ${prop.description} */`
        : "";
      return `${key}${opt}: ${
        jsonSchemaNodeToTyping(includeDescriptions, prop)
      }${desc}`;
    },
  );
  return `{ ${entries.join(", ")} }`;
};

export const zodToTypingString = (zodObj: ZodType): string =>
  jsonSchemaObjectToTyping(true, z.toJSONSchema(zodObj));

// Parameter names, types and optionality, without the per-parameter
// descriptions — compact enough for the always-on inactive skills listing,
// where the goal is a schema-valid first touch, not full documentation.
export const zodToCompactTypingString = (zodObj: ZodType): string =>
  jsonSchemaObjectToTyping(false, z.toJSONSchema(zodObj));
