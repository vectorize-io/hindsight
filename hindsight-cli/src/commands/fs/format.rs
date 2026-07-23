//! Turn a knowledge-base snapshot (folder/page tree + page contents) into a
//! concrete on-disk mirror plan: which directories to create (folders) and
//! which `.md` files to write (pages), at their nested paths.

use std::collections::HashSet;

use super::client::{KnowledgeNode, KnowledgeSnapshot};

const PAGE_PLACEHOLDER: &str = "_This page has not been generated yet._";

// ── Frontmatter ────────────────────────────────────────
//
// A tiny YAML frontmatter serializer for the flat metadata maps we emit. We
// only ever write scalars and string arrays; double-quoted scalars use JSON
// string syntax (a valid subset of YAML), giving correct escaping for free.

/// A frontmatter value: scalar or list-of-strings.
pub enum FmValue {
    Str(String),
    Num(i64),
    Null,
    List(Vec<String>),
}

/// True when a string is safe to emit unquoted as a YAML plain scalar.
fn is_plain_safe(s: &str) -> bool {
    if s.is_empty() || s != s.trim() {
        return false;
    }
    if s.chars()
        .any(|c| ":#[]{}&*!|>'\"%@`,".contains(c) || c == '\n' || c == '\r' || c == '\t')
    {
        return false;
    }
    let first = s.chars().next().unwrap();
    if first == '-' || first == '?' {
        return false;
    }
    let lower = s.to_ascii_lowercase();
    if matches!(
        lower.as_str(),
        "true" | "false" | "null" | "yes" | "no" | "on" | "off" | "~"
    ) {
        return false;
    }
    // Could parse as a number/date: optional sign then a digit.
    let after_sign = s.trim_start_matches(['-', '+']);
    if after_sign.chars().next().map(|c| c.is_ascii_digit()) == Some(true) {
        return false;
    }
    true
}

fn emit_scalar_str(s: &str) -> String {
    if is_plain_safe(s) {
        s.to_string()
    } else {
        serde_json::to_string(s).unwrap_or_else(|_| format!("\"{}\"", s))
    }
}

/// Serialize a frontmatter map to a `---`-delimited YAML block (no trailing
/// newline). Order is preserved from the input vec.
pub fn stringify_frontmatter(data: &[(&str, FmValue)]) -> String {
    let mut lines = vec!["---".to_string()];
    for (key, value) in data {
        match value {
            FmValue::List(items) => {
                if items.is_empty() {
                    lines.push(format!("{}: []", key));
                } else {
                    let joined = items
                        .iter()
                        .map(|v| emit_scalar_str(v))
                        .collect::<Vec<_>>()
                        .join(", ");
                    lines.push(format!("{}: [{}]", key, joined));
                }
            }
            FmValue::Str(s) => lines.push(format!("{}: {}", key, emit_scalar_str(s))),
            FmValue::Num(n) => lines.push(format!("{}: {}", key, n)),
            FmValue::Null => lines.push(format!("{}: null", key)),
        }
    }
    lines.push("---".to_string());
    lines.join("\n")
}

// ── Mirror planning ────────────────────────────────────

/// Map a folder/page name to a safe path segment (no slashes, lowercase).
pub fn slug(name: &str) -> String {
    let lower = name.to_lowercase();
    // Replace runs of non-[a-z0-9._-] with a single '-'.
    let mut collapsed = String::with_capacity(lower.len());
    let mut prev_dash = false;
    for c in lower.chars() {
        if c.is_ascii_lowercase() || c.is_ascii_digit() || c == '.' || c == '_' || c == '-' {
            collapsed.push(c);
            prev_dash = false;
        } else if !prev_dash {
            collapsed.push('-');
            prev_dash = true;
        }
    }
    // Trim leading/trailing dashes, then collapse runs of dots to a single dot.
    let trimmed = collapsed.trim_matches('-');
    let mut safe = String::with_capacity(trimmed.len());
    let mut prev_dot = false;
    for c in trimmed.chars() {
        if c == '.' {
            if !prev_dot {
                safe.push('.');
            }
            prev_dot = true;
        } else {
            safe.push(c);
            prev_dot = false;
        }
    }
    if safe.is_empty() {
        "untitled".to_string()
    } else {
        safe
    }
}

/// Ensure sibling nodes get distinct path segments even if their names collide.
fn unique_segment(base: &str, id: &str, used: &mut HashSet<String>) -> String {
    if !used.contains(base) {
        used.insert(base.to_string());
        return base.to_string();
    }
    let alnum: String = id.chars().filter(|c| c.is_ascii_alphanumeric()).collect();
    let suffix = if alnum.is_empty() {
        "x".to_string()
    } else {
        alnum
            .chars()
            .rev()
            .take(6)
            .collect::<String>()
            .chars()
            .rev()
            .collect()
    };
    let mut seg = format!("{}-{}", base, suffix);
    while used.contains(&seg) {
        seg = format!("{}-x", seg);
    }
    used.insert(seg.clone());
    seg
}

/// A page rendered to a relative path + full document content.
pub struct PageFile {
    /// Relative path under the mount, e.g. "engineering/runbooks/orders.md".
    pub rel_path: String,
    pub content: String,
}

/// A flattened plan of directories to create + page files to write.
pub struct MirrorPlan {
    /// Folder directories to create, in tree order (parents before children).
    pub dirs: Vec<String>,
    /// Page files to write at their nested paths.
    pub files: Vec<PageFile>,
    pub folder_count: usize,
    pub page_count: usize,
}

fn page_content(node: &KnowledgeNode, snapshot: &KnowledgeSnapshot) -> String {
    if let Some(doc) = snapshot.content.get(&node.id) {
        if !doc.trim().is_empty() {
            return if doc.ends_with('\n') {
                doc.clone()
            } else {
                format!("{}\n", doc)
            };
        }
    }
    // Fallback OKF-ish doc when the page body hasn't synthesized yet.
    let fm = stringify_frontmatter(&[
        ("id", FmValue::Str(node.id.clone())),
        ("type", FmValue::Str("knowledge-page".to_string())),
        ("title", FmValue::Str(node.name.clone())),
        ("tags", FmValue::List(node.tags.clone())),
        (
            "timestamp",
            match &node.timestamp {
                Some(t) => FmValue::Str(t.clone()),
                None => FmValue::Null,
            },
        ),
    ]);
    format!("{}\n\n{}\n", fm, PAGE_PLACEHOLDER)
}

fn sorted_by_name(nodes: &[KnowledgeNode]) -> Vec<&KnowledgeNode> {
    let mut ordered: Vec<&KnowledgeNode> = nodes.iter().collect();
    ordered.sort_by(|a, b| a.name.cmp(&b.name));
    ordered
}

/// Walk the tree into a flat list of directories + page files at nested paths.
pub fn plan_mirror(snapshot: &KnowledgeSnapshot) -> MirrorPlan {
    let mut plan = MirrorPlan {
        dirs: Vec::new(),
        files: Vec::new(),
        folder_count: 0,
        page_count: 0,
    };
    walk_plan(&snapshot.roots, "", snapshot, &mut plan);
    plan
}

fn walk_plan(
    nodes: &[KnowledgeNode],
    parent_dir: &str,
    snapshot: &KnowledgeSnapshot,
    plan: &mut MirrorPlan,
) {
    let mut used = HashSet::new();
    for node in sorted_by_name(nodes) {
        let seg = unique_segment(&slug(&node.name), &node.id, &mut used);
        let rel = if parent_dir.is_empty() {
            seg
        } else {
            format!("{}/{}", parent_dir, seg)
        };
        if node.is_folder() {
            plan.folder_count += 1;
            plan.dirs.push(rel.clone());
            walk_plan(&node.children, &rel, snapshot, plan);
        } else {
            plan.page_count += 1;
            plan.files.push(PageFile {
                rel_path: format!("{}.md", rel),
                content: page_content(node, snapshot),
            });
        }
    }
}

/// Build the `index.md` overview written into the control directory.
pub fn render_index(snapshot: &KnowledgeSnapshot, bank_id: &str, api_url: &str) -> String {
    let plan = plan_mirror(snapshot);
    let fm = stringify_frontmatter(&[
        ("bank", FmValue::Str(bank_id.to_string())),
        ("api_url", FmValue::Str(api_url.to_string())),
        ("folders", FmValue::Num(plan.folder_count as i64)),
        ("pages", FmValue::Num(plan.page_count as i64)),
    ]);

    let mut lines: Vec<String> = Vec::new();
    walk_index(&snapshot.roots, 0, "", &mut lines);
    let body = if lines.is_empty() {
        "_This bank has no knowledge base yet._".to_string()
    } else {
        lines.join("\n")
    };
    format!("{}\n\n# Knowledge base — `{}`\n\n{}\n", fm, bank_id, body)
}

fn walk_index(nodes: &[KnowledgeNode], depth: usize, parent_dir: &str, lines: &mut Vec<String>) {
    let mut used = HashSet::new();
    for node in sorted_by_name(nodes) {
        let seg = unique_segment(&slug(&node.name), &node.id, &mut used);
        let rel = if parent_dir.is_empty() {
            seg
        } else {
            format!("{}/{}", parent_dir, seg)
        };
        let indent = "  ".repeat(depth);
        if node.is_folder() {
            let mission = node
                .mission
                .as_deref()
                .filter(|m| !m.is_empty())
                .map(|m| format!(" — _{}_", m))
                .unwrap_or_default();
            lines.push(format!("{}- **{}/**{}", indent, node.name, mission));
            walk_index(&node.children, depth + 1, &rel, lines);
        } else {
            let auto = if node.managed { " ·auto" } else { "" };
            lines.push(format!(
                "{}- [`{}`](../{}.md){}",
                indent, node.name, rel, auto
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn page(id: &str, name: &str) -> KnowledgeNode {
        KnowledgeNode {
            id: id.to_string(),
            kind: "page".to_string(),
            name: name.to_string(),
            mission: None,
            managed: false,
            tags: Vec::new(),
            timestamp: None,
            children: Vec::new(),
        }
    }

    fn folder(id: &str, name: &str, children: Vec<KnowledgeNode>) -> KnowledgeNode {
        KnowledgeNode {
            id: id.to_string(),
            kind: "folder".to_string(),
            name: name.to_string(),
            mission: None,
            managed: false,
            tags: Vec::new(),
            timestamp: None,
            children,
        }
    }

    #[test]
    fn slug_lowercases_and_sanitizes() {
        assert_eq!(slug("Billing Policy"), "billing-policy");
        assert_eq!(slug("Net-30 / Terms!"), "net-30-terms");
        assert_eq!(slug("!!!"), "untitled"); // all-unsafe → fallback
        assert_eq!(slug("A..B"), "a.b"); // dot runs collapse
    }

    #[test]
    fn frontmatter_quotes_only_when_needed() {
        let s = stringify_frontmatter(&[
            ("id", FmValue::Str("kp-123".to_string())),
            ("title", FmValue::Str("Has: colon".to_string())),
            ("plain", FmValue::Str("simple".to_string())),
            (
                "tags",
                FmValue::List(vec!["a".to_string(), "b,c".to_string()]),
            ),
            ("count", FmValue::Num(3)),
            ("ts", FmValue::Null),
        ]);
        assert!(s.starts_with("---\n"));
        assert!(s.ends_with("\n---"));
        assert!(s.contains("id: kp-123")); // plain-safe, unquoted
        assert!(s.contains("title: \"Has: colon\"")); // colon forces quoting
        assert!(s.contains("plain: simple"));
        assert!(s.contains("tags: [a, \"b,c\"]")); // comma forces item quoting
        assert!(s.contains("count: 3"));
        assert!(s.contains("ts: null"));
    }

    #[test]
    fn is_plain_safe_rejects_ambiguous_scalars() {
        assert!(is_plain_safe("hello"));
        assert!(!is_plain_safe("true"));
        assert!(!is_plain_safe("123"));
        assert!(!is_plain_safe("-leading"));
        assert!(!is_plain_safe("a: b"));
        assert!(!is_plain_safe(""));
        assert!(!is_plain_safe(" padded "));
    }

    #[test]
    fn plan_mirror_nests_and_sorts() {
        let snapshot = KnowledgeSnapshot {
            roots: vec![
                folder("f1", "Policies", vec![page("p1", "Billing")]),
                page("p2", "Overview"),
            ],
            content: {
                let mut m = HashMap::new();
                m.insert("p1".to_string(), "# Billing\n".to_string());
                m
            },
        };
        let plan = plan_mirror(&snapshot);
        assert_eq!(plan.folder_count, 1);
        assert_eq!(plan.page_count, 2);
        assert_eq!(plan.dirs, vec!["policies"]);
        let paths: Vec<&str> = plan.files.iter().map(|f| f.rel_path.as_str()).collect();
        assert!(paths.contains(&"policies/billing.md"));
        assert!(paths.contains(&"overview.md"));
    }

    #[test]
    fn plan_mirror_disambiguates_colliding_names() {
        let snapshot = KnowledgeSnapshot {
            roots: vec![page("aaaaaa", "Same"), page("bbbbbb", "Same")],
            content: HashMap::new(),
        };
        let plan = plan_mirror(&snapshot);
        let paths: Vec<String> = plan.files.iter().map(|f| f.rel_path.clone()).collect();
        assert_eq!(paths.len(), 2);
        assert_ne!(
            paths[0], paths[1],
            "colliding names must get distinct paths"
        );
    }

    #[test]
    fn missing_page_body_falls_back_to_placeholder_doc() {
        let snapshot = KnowledgeSnapshot {
            roots: vec![page("p1", "Empty Page")],
            content: HashMap::new(),
        };
        let plan = plan_mirror(&snapshot);
        let doc = &plan.files[0].content;
        assert!(doc.contains("type: knowledge-page"));
        assert!(doc.contains("title: Empty Page"));
        assert!(doc.contains(PAGE_PLACEHOLDER));
    }
}
