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

// deno-lint-ignore no-explicit-any
const jsonSchemaNodeToTyping = (node: any): string => {
  if (node.enum) return node.enum.map((v: string) => `"${v}"`).join(" | ");
  if (node.anyOf) return node.anyOf.map(jsonSchemaNodeToTyping).join(" | ");
  if (node.type === "array") {
    return `${jsonSchemaNodeToTyping(node.items || { type: "unknown" })}[]`;
  }
  if (node.type === "object" && node.properties) {
    return jsonSchemaObjectToTyping(node);
  }
  return node.type || "unknown";
};

// deno-lint-ignore no-explicit-any
const jsonSchemaObjectToTyping = (schema: any): string => {
  const pruned = pruneDefaultsFromRequired(schema);
  const required = new Set(pruned.required || []);
  const entries = Object.entries(pruned.properties || {}).map(
    // deno-lint-ignore no-explicit-any
    ([key, prop]: [string, any]) => {
      const opt = required.has(key) ? "" : "?";
      const desc = prop.description ? ` /* ${prop.description} */` : "";
      return `${key}${opt}: ${jsonSchemaNodeToTyping(prop)}${desc}`;
    },
  );
  return `{ ${entries.join(", ")} }`;
};

export const zodToTypingString = (zodObj: ZodType): string =>
  jsonSchemaObjectToTyping(z.toJSONSchema(zodObj));
