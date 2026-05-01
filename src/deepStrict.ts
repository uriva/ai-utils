import z from "zod/v4";

export function deepStrict(schema: any): any {
  if (schema instanceof z.ZodObject) {
    const shape = schema.shape;
    const strictShape: any = {};
    for (const key in shape) {
      strictShape[key] = deepStrict(shape[key]);
    }
    return z.object(strictShape).strict();
  }
  if (schema instanceof z.ZodOptional) {
    return z.optional(deepStrict(schema.unwrap()));
  }
  if (schema instanceof z.ZodNullable) {
    return z.nullable(deepStrict(schema.unwrap()));
  }
  if (schema instanceof z.ZodArray) {
    return z.array(deepStrict(schema.element));
  }
  return schema;
}
