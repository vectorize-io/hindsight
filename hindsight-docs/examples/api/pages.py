#!/usr/bin/env python3
"""
Pages API examples for Hindsight.
Run: python examples/api/pages.py
"""
import os
import time

HINDSIGHT_URL = os.getenv("HINDSIGHT_API_URL", "http://localhost:8888")
BANK_ID = "pages-demo-bank"

# =============================================================================
# Setup (not shown in docs)
# =============================================================================
from hindsight_client import Hindsight

client = Hindsight(base_url=HINDSIGHT_URL)

client.create_bank(bank_id=BANK_ID, name="Pages Demo")
client.retain(bank_id=BANK_ID, content="The API is deployed to Kubernetes with a rolling update")
client.retain(bank_id=BANK_ID, content="Deploys run from the main branch after CI passes")
client.retain(bank_id=BANK_ID, content="A failed deploy is rolled back by redeploying the previous tag")

time.sleep(2)

# =============================================================================
# Doc Examples
# =============================================================================

# [docs:create-folder]
# Create a folder (omit parent_id, or pass None, to create it at the root)
folder = client.create_page_folder(BANK_ID, name="Operations")

print(f"Folder ID: {folder.id}")
# [/docs:create-folder]

# [docs:create-page]
# Create a page — content is generated in the background
page = client.create_page(
    BANK_ID,
    name="Deploying the API",
    source_query="How is the API deployed?",
    parent_id=folder.id,
    tags=["ops", "type:runbook"],
)

# Poll the operation to know when the first build has finished
print(f"Page ID: {page.page_id}, operation: {page.operation_id}")
# [/docs:create-page]

# Wait for the page's first build
time.sleep(20)

# [docs:get-tree]
# Fetch all the bank's pages as a nested folder/page tree (no page bodies)
tree = client.get_page_tree(BANK_ID)

for root in tree.roots:
    print(f"{root.kind}: {root.name}")
    for child in root.children:
        print(f"  {child.kind}: {child.name} (stale: {child.is_stale})")
# [/docs:get-tree]

# [docs:get-page]
# Read a page as a markdown document
document = client.get_page(BANK_ID, page.page_id)

print(document.type)      # "runbook" — from the type:runbook tag
print(document.body)      # the synthesized markdown body
print(document.markdown)  # YAML frontmatter + body
# [/docs:get-page]

# [docs:search-pages]
# Hybrid search (full-text + vector) over whole pages
results = client.search_pages(BANK_ID, q="how do we deploy", limit=5)

for hit in results.results:
    print(f"{hit.score:.3f}  {hit.name}: {hit.snippet}")
# [/docs:search-pages]

# [docs:filter-by-tags]
# Narrow the tree and search to pages carrying a tag (recall's tags / tags_match / tag_groups)
tree = client.get_page_tree(BANK_ID, tags=["type:runbook"], tags_match="any_strict")
ops_hits = client.search_pages(BANK_ID, q="how do we deploy", tags=["ops"], tags_match="all_strict")
print(f"{len(tree.roots)} runbook roots, {len(ops_hits.results)} ops hits")

# Compound filters use tag_groups: here, runbooks that are not drafts
not_drafts = client.search_pages(
    BANK_ID,
    q="how do we deploy",
    tag_groups=[{"and": [{"tags": ["type:runbook"]}, {"not": {"tags": ["draft"]}}]}],
)
print(f"{len(not_drafts.results)} non-draft runbooks")
# [/docs:filter-by-tags]

# [docs:update-node]
# Rename a node, move it, and/or update a page's options.
# Changing source_query rebuilds the page against the new question.
client.update_page_node(
    BANK_ID,
    page.page_id,
    name="Deploying the API (v2)",
    tags=["ops", "type:runbook", "reviewed"],
)
# [/docs:update-node]

# [docs:export]
# Export the bank's pages as a portable markdown bundle
bundle = client.export_pages(BANK_ID)

for file in bundle.files:
    print(file.path)  # index.md, <page-id>.md, <page-id>.log.md
# [/docs:export]

# [docs:delete-node]
# Delete a folder or page — deleting a folder removes its whole subtree
client.delete_page_node(BANK_ID, folder.id)
# [/docs:delete-node]

# =============================================================================
# Cleanup (not shown in docs)
# =============================================================================
client.delete_bank(bank_id=BANK_ID)

print("pages.py: All examples passed")
