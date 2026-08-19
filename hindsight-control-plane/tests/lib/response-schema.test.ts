import { describe, it, expect } from "vitest";

import {
  validateResponseSchema,
  fieldsToSchema,
  schemaToFields,
  emptyField,
  nodeForType,
  type SchemaField,
} from "@/lib/response-schema";

/**
 * Unit tests for the response_schema helpers that back the no-code schema
 * builder. These are pure functions: validation (mirroring the backend's
 * usable-shape contract) and the recursive fields<->JSON-Schema conversion.
 */

describe("validateResponseSchema", () => {
  it("accepts an object schema with properties", () => {
    expect(
      validateResponseSchema({ type: "object", properties: { a: { type: "string" } } })
    ).toBeNull();
  });

  it("allows type to be omitted", () => {
    expect(validateResponseSchema({ properties: { a: { type: "string" } } })).toBeNull();
  });

  it.each([
    [["not", "an", "object"], /JSON object/],
    ["a string", /JSON object/],
    [{ type: "array", properties: { a: { type: "string" } } }, /object schema/],
    [{ type: "object" }, /non-empty 'properties'/],
    [{ properties: {} }, /at least one property/],
    [{ properties: { a: "nope" } }, /must be an object/],
    [{ properties: { a: { type: "banana" } } }, /unsupported type/],
    [{ properties: { a: { type: "string" } }, required: "a" }, /list of property names/],
    [{ properties: { a: { type: "string" } }, required: ["b"] }, /unknown properties/],
  ])("rejects %j", (schema, pattern) => {
    expect(validateResponseSchema(schema)).toMatch(pattern as RegExp);
  });
});

describe("fieldsToSchema", () => {
  it("builds an object schema with required list", () => {
    const fields: SchemaField[] = [
      { name: "summary", required: true, description: "the gist", node: { type: "string" } },
      { name: "count", required: false, description: "", node: { type: "integer" } },
    ];
    expect(fieldsToSchema(fields)).toEqual({
      type: "object",
      properties: {
        summary: { type: "string", description: "the gist" },
        count: { type: "integer" },
      },
      required: ["summary"],
    });
  });

  it("skips unnamed fields and omits an empty required list", () => {
    const fields: SchemaField[] = [
      emptyField(), // unnamed → skipped
      { name: "a", required: false, description: "", node: { type: "string" } },
    ];
    const schema = fieldsToSchema(fields);
    expect(Object.keys(schema.properties as object)).toEqual(["a"]);
    expect(schema.required).toBeUndefined();
  });

  it("serializes nested objects and arrays of objects", () => {
    const fields: SchemaField[] = [
      {
        name: "author",
        required: true,
        description: "",
        node: {
          type: "object",
          fields: [{ name: "name", required: false, description: "", node: { type: "string" } }],
        },
      },
      {
        name: "tags",
        required: false,
        description: "",
        node: {
          type: "array",
          items: {
            type: "object",
            fields: [{ name: "label", required: false, description: "", node: { type: "string" } }],
          },
        },
      },
    ];
    expect(fieldsToSchema(fields)).toEqual({
      type: "object",
      properties: {
        author: { type: "object", properties: { name: { type: "string" } } },
        tags: {
          type: "array",
          items: { type: "object", properties: { label: { type: "string" } } },
        },
      },
      required: ["author"],
    });
  });
});

describe("schemaToFields", () => {
  it("round-trips a nested schema through fieldsToSchema", () => {
    const schema = {
      type: "object",
      properties: {
        author: { type: "object", properties: { name: { type: "string" } } },
        tags: {
          type: "array",
          items: { type: "object", properties: { label: { type: "string" } } },
        },
      },
      required: ["author"],
    };
    const fields = schemaToFields(schema);
    expect(fields).not.toBeNull();
    expect(fieldsToSchema(fields!)).toEqual(schema);
  });

  it("treats a missing/empty properties map as no fields (representable)", () => {
    expect(schemaToFields({ type: "object" })).toEqual([]);
    expect(schemaToFields({ type: "object", properties: {} })).toEqual([]);
  });

  it("returns null for schemas the flat/nested editor can't represent", () => {
    // enum is not modelled
    expect(schemaToFields({ properties: { a: { type: "string", enum: ["x"] } } })).toBeNull();
    // nested object with an unrepresentable child bubbles up
    expect(
      schemaToFields({
        properties: { a: { type: "object", properties: { b: { $ref: "#/x" } } } },
      })
    ).toBeNull();
    // tuple-form array items
    expect(
      schemaToFields({ properties: { a: { type: "array", items: [{ type: "string" }] } } })
    ).toBeNull();
  });
});

describe("nodeForType", () => {
  it("seeds object with a blank field and array with a string item", () => {
    expect(nodeForType("object")).toEqual({ type: "object", fields: [emptyField()] });
    expect(nodeForType("array")).toEqual({ type: "array", items: { type: "string" } });
    expect(nodeForType("number")).toEqual({ type: "number" });
  });
});
