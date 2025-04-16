import Ts from "npm:typescript@5.8.3";
import { z } from "npm:zod@3.24.2";

function zodToTsType(
    schema: z.ZodTypeAny,
    isObjectProperty = false,
): Ts.TypeNode {
    let typeNode: Ts.TypeNode;

    if (schema instanceof z.ZodString) {
        typeNode = Ts.factory.createKeywordTypeNode(
            Ts.SyntaxKind.StringKeyword,
        );
    } else if (schema instanceof z.ZodNumber) {
        typeNode = Ts.factory.createKeywordTypeNode(
            Ts.SyntaxKind.NumberKeyword,
        );
    } else if (schema instanceof z.ZodBoolean) {
        typeNode = Ts.factory.createKeywordTypeNode(
            Ts.SyntaxKind.BooleanKeyword,
        );
    } else if (schema instanceof z.ZodEnum) {
        typeNode = Ts.factory.createUnionTypeNode(
            (schema.options as readonly string[]).map((option: string) =>
                Ts.factory.createLiteralTypeNode(
                    Ts.factory.createStringLiteral(option),
                )
            ),
        );
    } else if (schema instanceof z.ZodObject) {
        const properties = Object.entries(schema.shape).map(
            ([key, value]) =>
                createPropertySignature(key, value as z.ZodTypeAny),
        );
        typeNode = Ts.factory.createTypeLiteralNode(properties);
    } else if (schema instanceof z.ZodArray) {
        typeNode = Ts.factory.createArrayTypeNode(zodToTsType(schema.element));
    } else if (schema instanceof z.ZodUnion) {
        // eslint-disable-next-line ts/no-unsafe-call
        typeNode = Ts.factory.createUnionTypeNode(
            schema.options.map(zodToTsType),
        );
    } else if (schema instanceof z.ZodOptional) {
        typeNode = Ts.factory.createUnionTypeNode([
            zodToTsType(schema.unwrap()),
            Ts.factory.createKeywordTypeNode(Ts.SyntaxKind.UndefinedKeyword),
        ]);
    } else if (schema instanceof z.ZodNullable) {
        typeNode = Ts.factory.createUnionTypeNode([
            zodToTsType(schema.unwrap()),
            Ts.factory.createLiteralTypeNode(Ts.factory.createNull()),
        ]);
    } else if (schema instanceof z.ZodLiteral) {
        typeNode = Ts.factory.createLiteralTypeNode(
            typeof schema.value === "string"
                ? Ts.factory.createStringLiteral(schema.value)
                : Ts.factory.createNumericLiteral(schema.value),
        );
    } else if (schema instanceof z.ZodDate) {
        typeNode = Ts.factory.createTypeReferenceNode("Date", undefined);
    } else if ("coerce" in schema && schema.coerce instanceof z.ZodType) {
        typeNode = zodToTsType(schema.coerce);
    } else if (schema instanceof z.ZodReadonly) {
        typeNode = Ts.factory.createTypeReferenceNode(
            Ts.factory.createIdentifier("Readonly"),
            [
                zodToTsType(schema.unwrap()),
            ],
        );
    } else if (schema instanceof z.ZodDefault) {
        typeNode = zodToTsType(schema._def.innerType);
    } else {
        typeNode = Ts.factory.createKeywordTypeNode(Ts.SyntaxKind.AnyKeyword);
    }

    if (!isObjectProperty) {
        typeNode = addCommentToNode(typeNode, schema) as Ts.TypeNode;
    }

    return typeNode;
}

function createPropertySignature(
    key: string,
    schema: z.ZodTypeAny,
): Ts.PropertySignature {
    const isOptional = schema instanceof z.ZodOptional;
    const unwrappedSchema = isOptional ? schema.unwrap() : schema;
    const isReadonly = unwrappedSchema instanceof z.ZodReadonly;
    const finalSchema = isReadonly ? unwrappedSchema.unwrap() : unwrappedSchema;

    const propertySignature = Ts.factory.createPropertySignature(
        isReadonly
            ? [Ts.factory.createModifier(Ts.SyntaxKind.ReadonlyKeyword)]
            : undefined,
        key,
        isOptional
            ? Ts.factory.createToken(Ts.SyntaxKind.QuestionToken)
            : undefined,
        zodToTsType(finalSchema, true),
    );

    return addCommentToNode(propertySignature, schema) as Ts.PropertySignature;
}

function addCommentToNode(node: Ts.Node, schema: z.ZodTypeAny): Ts.Node {
    let comment = "";

    // Add description to comment if available
    if ("description" in schema && typeof schema.description === "string") {
        comment += schema.description;
    }

    // Add default value to comment if available
    if (schema instanceof z.ZodDefault) {
        const defaultValue = schema._def.defaultValue();
        const defaultValueString = typeof defaultValue === "string"
            ? `"${defaultValue}"`
            : defaultValue;
        comment += comment ? " " : "";
        comment += `(@default ${defaultValueString})`;
    }

    // Add comment to node if we have one
    if (comment) {
        return Ts.addSyntheticTrailingComment(
            node,
            Ts.SyntaxKind.SingleLineCommentTrivia,
            ` ${comment}`,
            false,
        );
    }

    return node;
}

export const zodToTs = ({
    schema,
    name,
    comment,
    printOptions,
}: {
    schema: z.ZodTypeAny;
    name: string;
    comment?: string;
    printOptions?: Ts.PrinterOptions;
}) => {
    const typeNode = Ts.factory.createTypeAliasDeclaration(
        [Ts.factory.createModifier(Ts.SyntaxKind.ExportKeyword)],
        name,
        undefined,
        zodToTsType(schema),
    );

    if (comment) {
        Ts.addSyntheticLeadingComment(
            typeNode,
            Ts.SyntaxKind.MultiLineCommentTrivia,
            `*\n * ${comment}\n `,
            true,
        );
    }

    const sourceFile = Ts.createSourceFile(
        "temp.ts",
        "",
        Ts.ScriptTarget.Latest,
        false,
        Ts.ScriptKind.TS,
    );

    const printer = Ts.createPrinter(printOptions);
    return printer.printNode(Ts.EmitHint.Unspecified, typeNode, sourceFile);
};
