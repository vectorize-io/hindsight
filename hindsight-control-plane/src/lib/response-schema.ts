/**
 * Helpers for the reflect / mental-model `response_schema` used by structured
 * output. The engine feeds the full schema (nesting included) to the model, so
 * the visual builder models a fully-recursive tree: every field has a `node`,
 * an `object` node has nested `fields`, and an `array` node has an `items` node.
 * The frontend validation mirrors the backend's usable-shape contract (an object
 * schema with a non-empty top-level `properties` map).
 */

export const SCHEMA_FIELD_TYPES = [
  "string",
  "number",
  "integer",
  "boolean",
  "array",
  "object",
] as const;

export type SchemaFieldType = (typeof SCHEMA_FIELD_TYPES)[number];

/** A type node in the schema tree. */
export interface SchemaNode {
  type: SchemaFieldType;
  /** Present when `type === "object"`. */
  fields?: SchemaField[];
  /** Present when `type === "array"`. */
  items?: SchemaNode;
}

/** A named property, as edited in the visual builder. */
export interface SchemaField {
  name: string;
  required: boolean;
  description: string;
  node: SchemaNode;
}

/** A single blank scalar field. */
export function emptyField(): SchemaField {
  return { name: "", required: false, description: "", node: { type: "string" } };
}

/** A fresh node for a chosen type (used when a field/item changes type). */
export function nodeForType(type: SchemaFieldType): SchemaNode {
  if (type === "object") return { type, fields: [emptyField()] };
  if (type === "array") return { type, items: { type: "string" } };
  return { type };
}

/**
 * Validate a parsed JSON value against the usable-schema contract (top-level,
 * matching the backend). Returns a human-readable error, or `null` when usable.
 */
export function validateResponseSchema(schema: unknown): string | null {
  if (typeof schema !== "object" || schema === null || Array.isArray(schema)) {
    return "Schema must be a JSON object.";
  }
  const obj = schema as Record<string, unknown>;

  if (obj.type !== undefined && obj.type !== "object") {
    return 'Schema must be an object schema (its "type" must be "object").';
  }

  const properties = obj.properties;
  if (typeof properties !== "object" || properties === null || Array.isArray(properties)) {
    return "Schema must define a non-empty 'properties' object.";
  }
  const propEntries = Object.entries(properties as Record<string, unknown>);
  if (propEntries.length === 0) {
    return "Schema must define at least one property.";
  }

  for (const [name, prop] of propEntries) {
    if (typeof prop !== "object" || prop === null || Array.isArray(prop)) {
      return `Property '${name}' must be an object.`;
    }
    const propType = (prop as Record<string, unknown>).type;
    if (propType !== undefined && !SCHEMA_FIELD_TYPES.includes(propType as SchemaFieldType)) {
      return `Property '${name}' has unsupported type '${String(propType)}'.`;
    }
  }

  const required = obj.required;
  if (required !== undefined) {
    if (!Array.isArray(required) || !required.every((r) => typeof r === "string")) {
      return "'required' must be a list of property names.";
    }
    const propNames = new Set(propEntries.map(([n]) => n));
    const unknown = (required as string[]).filter((r) => !propNames.has(r));
    if (unknown.length > 0) {
      return `'required' references unknown properties: ${unknown.join(", ")}.`;
    }
  }

  return null;
}

/** Serialize one node (plus its owning field's description) into a JSON Schema. */
function nodeToSchema(node: SchemaNode, description?: string): Record<string, unknown> {
  const out: Record<string, unknown> = { type: node.type };
  if (node.type === "object") {
    const inner = fieldsToSchema(node.fields ?? []);
    out.properties = inner.properties;
    if (inner.required !== undefined) out.required = inner.required;
  } else if (node.type === "array") {
    out.items = nodeToSchema(node.items ?? { type: "string" });
  }
  if (description && description.trim()) out.description = description.trim();
  return out;
}

/** Build a JSON Schema object from the visual builder's field list. */
export function fieldsToSchema(fields: SchemaField[]): Record<string, unknown> {
  const properties: Record<string, unknown> = {};
  const required: string[] = [];
  for (const field of fields) {
    const name = field.name.trim();
    if (!name) continue;
    properties[name] = nodeToSchema(field.node, field.description);
    if (field.required) required.push(name);
  }
  const schema: Record<string, unknown> = { type: "object", properties };
  if (required.length > 0) schema.required = required;
  return schema;
}

// Keys the visual editor round-trips losslessly. A schema using anything else
// (enum, oneOf, $ref, additionalProperties, format, tuple items, …) is kept in
// code mode rather than silently dropping the parts we can't render.
const ALLOWED_NODE_KEYS = new Set([
  "type",
  "description",
  "properties",
  "required",
  "items",
  "title",
]);

/** Convert one JSON Schema property into a node, or `null` if unrepresentable. */
function schemaToNode(prop: Record<string, unknown>): SchemaNode | null {
  for (const key of Object.keys(prop)) {
    if (!ALLOWED_NODE_KEYS.has(key)) return null;
  }
  const type = prop.type;
  if (typeof type !== "string" || !SCHEMA_FIELD_TYPES.includes(type as SchemaFieldType))
    return null;
  const t = type as SchemaFieldType;

  if (t === "object") {
    const inner = schemaToFields(prop);
    if (inner === null) return null;
    return { type: t, fields: inner };
  }
  if (t === "array") {
    const items = prop.items;
    if (items === undefined) return { type: t, items: { type: "string" } };
    // Tuple-form items (an array of schemas) aren't representable in the editor.
    if (typeof items !== "object" || items === null || Array.isArray(items)) return null;
    const itemNode = schemaToNode(items as Record<string, unknown>);
    if (itemNode === null) return null;
    return { type: t, items: itemNode };
  }
  return { type: t };
}

/**
 * Convert a JSON Schema object into visual builder fields. Returns `null` when
 * the schema uses features the flat/nested editor can't represent losslessly —
 * the caller keeps the user in code mode so nothing is silently dropped.
 */
export function schemaToFields(schema: Record<string, unknown>): SchemaField[] | null {
  const properties = schema.properties;
  // A missing/empty `properties` is representable — it just means "no fields yet".
  if (properties === undefined) return [];
  if (typeof properties !== "object" || properties === null || Array.isArray(properties)) {
    return null;
  }
  const requiredSet = new Set(Array.isArray(schema.required) ? (schema.required as string[]) : []);
  const fields: SchemaField[] = [];
  for (const [name, rawProp] of Object.entries(properties as Record<string, unknown>)) {
    if (typeof rawProp !== "object" || rawProp === null || Array.isArray(rawProp)) return null;
    const prop = rawProp as Record<string, unknown>;
    const node = schemaToNode(prop);
    if (node === null) return null;
    fields.push({
      name,
      required: requiredSet.has(name),
      description: typeof prop.description === "string" ? prop.description : "",
      node,
    });
  }
  return fields;
}
