#!/usr/bin/env python3
"""
Import knowledge into Hindsight using MCP tools.
This script coordinates the import from:
1. Memlord MCP (via subprocess calls)
2. Graphify knowledge graph
3. Documentation markdown files
"""

import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any


class KnowledgeImporter:
    """Orchestrate knowledge import to Hindsight."""
    
    def __init__(self):
        self.stats = {
            "memlord_imported": 0,
            "graphify_imported": 0,
            "docs_imported": 0,
            "errors": []
        }
        
        # These will be populated with actual data
        self.memlord_memories = []
        self.graphify_data = None
        self.doc_files = []
    
    def load_memlord_export(self, json_path: str):
        """Load memlord memories from exported JSON."""
        print(f"📚 Loading memlord export from {json_path}...")
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, dict) and 'items' in data:
                self.memlord_memories = data['items']
            elif isinstance(data, list):
                self.memlord_memories = data
            else:
                self.memlord_memories = [data]
            
            print(f"✅ Loaded {len(self.memlord_memories)} memlord memories")
        except Exception as e:
            print(f"❌ Error loading memlord export: {e}")
            self.stats["errors"].append(str(e))
    
    def load_graphify_graph(self, graph_path: str):
        """Load graphify knowledge graph."""
        print(f"🕸️  Loading graphify graph from {graph_path}...")
        try:
            with open(graph_path, 'r') as f:
                self.graphify_data = json.load(f)
            
            nodes = len(self.graphify_data.get("nodes", []))
            edges = len(self.graphify_data.get("links", []))
            print(f"✅ Loaded graphify graph: {nodes} nodes, {edges} edges")
        except Exception as e:
            print(f"❌ Error loading graphify: {e}")
            self.stats["errors"].append(str(e))
    
    def find_documentation(self, docs_root: str):
        """Find all markdown documentation files."""
        print(f"📄 Scanning for documentation in {docs_root}...")
        try:
            md_files = []
            for path in Path(docs_root).rglob("*.md"):
                # Skip node_modules, .git, etc.
                if any(p.startswith('.') or p == 'node_modules' for p in path.parts):
                    continue
                md_files.append(path)
            
            self.doc_files = md_files
            print(f"✅ Found {len(self.doc_files)} documentation files")
        except Exception as e:
            print(f"❌ Error scanning docs: {e}")
            self.stats["errors"].append(str(e))
    
    def generate_hindsight_commands(self, output_file: str):
        """Generate shell script with Hindsight MCP commands."""
        print(f"\n🔧 Generating Hindsight import commands...")
        
        commands = []
        commands.append("#!/bin/bash")
        commands.append("# Auto-generated Hindsight import script")
        commands.append("# Run this to import knowledge into Hindsight banks")
        commands.append("")
        commands.append("set -e  # Exit on error")
        commands.append("")
        
        # Create banks
        commands.append("echo '📦 Creating Hindsight banks...'")
        commands.append("")
        
        # Import memlord memories
        if self.memlord_memories:
            commands.append(f"echo '📚 Importing {len(self.memlord_memories)} memlord memories...'")
            for i, mem in enumerate(self.memlord_memories[:100]):  # Limit to 100
                try:
                    # Build content
                    content_parts = [f"[{mem.get('memory_type', 'unknown').upper()}] {mem.get('name', 'Untitled')}"]
                    
                    if mem.get('metadata'):
                        meta_str = json.dumps(mem['metadata']).replace('"', '\\"')
                        content_parts.append(f"Metadata: {meta_str}")
                    
                    content = "\\n\\n".join(content_parts)
                    content = content.replace('"', '\\"').replace('$', '\\$')
                    
                    # Build tags
                    tags = mem.get('tags', [])
                    tags.append("source:memlord")
                    tags.append(f"type:{mem.get('memory_type', 'unknown')}")
                    tags_str = ",".join(tags)
                    
                    # Build metadata
                    metadata = {
                        "source": "memlord",
                        "name": mem.get('name'),
                        "workspace": mem.get('workspace')
                    }
                    metadata_str = json.dumps(metadata).replace('"', '\\"')
                    
                    commands.append(f'# Memory {i+1}: {mem.get("name", "Untitled")[:50]}')
                    commands.append(f'echo "  Importing: {mem.get("name", "Untitled")[:50]}..."')
                    
                    # Use hindsight_sync_retain for immediate storage
                    commands.append(f'# hindsight_sync_retain --bank_id collabmind-platform --content "{content}" --context "{mem.get("memory_type", "general")}" --tags "{tags_str}"')
                    commands.append("")
                    
                    self.stats["memlord_imported"] += 1
                except Exception as e:
                    self.stats["errors"].append(f"Memlord command gen failed: {mem.get('name', 'unknown')}: {e}")
        
        # Import graphify nodes
        if self.graphify_data:
            nodes = self.graphify_data.get("nodes", [])[:200]  # Limit to 200 nodes
            commands.append(f"echo '🕸️  Importing {len(nodes)} graphify nodes...'")
            
            for i, node in enumerate(nodes):
                try:
                    file_type = node.get("file_type", "unknown")
                    label = node.get("label", "")
                    source_file = node.get("source_file", "")
                    
                    content = f"[{file_type.upper()}] {label}"
                    if source_file:
                        content += f"\\nFile: {source_file}"
                    
                    content = content.replace('"', '\\"').replace('$', '\\$')
                    
                    tags = [
                        "source:graphify",
                        f"type:{file_type}"
                    ]
                    tags_str = ",".join(tags)
                    
                    if i % 50 == 0:
                        commands.append(f'echo "  Progress: {i}/{len(nodes)} nodes..."')
                    
                    commands.append(f'# hindsight_sync_retain --bank_id codebase-structure --content "{content}" --context code --tags "{tags_str}"')
                    
                    self.stats["graphify_imported"] += 1
                except Exception as e:
                    self.stats["errors"].append(f"Graphify command gen failed: {node.get('id', 'unknown')}: {e}")
            
            commands.append("")
        
        # Import documentation
        if self.doc_files:
            commands.append(f"echo '📄 Importing {len(self.doc_files[:50])} documentation files...'")
            
            for i, doc_path in enumerate(self.doc_files[:50]):  # Limit to 50 docs
                try:
                    with open(doc_path, 'r', encoding='utf-8', errors='ignore') as f:
                        doc_content = f.read()[:2000]  # Limit content size
                    
                    if len(doc_content.strip()) < 50:
                        continue
                    
                    title = doc_path.stem.replace('-', ' ').replace('_', ' ').title()
                    
                    content = f"# {title}\\n\\n{doc_content}"
                    content = content.replace('"', '\\"').replace('$', '\\$').replace('\n', '\\n')
                    
                    tags = ["source:docs", "type:documentation"]
                    tags_str = ",".join(tags)
                    
                    commands.append(f'echo "  Importing: {title[:50]}..."')
                    commands.append(f'# hindsight_sync_retain --bank_id documentation --content "{content}" --context documentation --tags "{tags_str}"')
                    commands.append("")
                    
                    self.stats["docs_imported"] += 1
                except Exception as e:
                    self.stats["errors"].append(f"Doc command gen failed: {doc_path}: {e}")
        
        commands.append("echo '✅ Import complete!'")
        commands.append("echo ''")
        commands.append("echo 'Imported:'")
        commands.append(f"echo '  • Memlord memories: {self.stats['memlord_imported']}'")
        commands.append(f"echo '  • Graphify nodes: {self.stats['graphify_imported']}'")
        commands.append(f"echo '  • Documentation: {self.stats['docs_imported']}'")
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(commands))
        
        import os
        os.chmod(output_file, 0o755)
        
        print(f"✅ Generated import script: {output_file}")
        print(f"   Uncomment the hindsight_sync_retain lines to actually import")
    
    def generate_import_json(self, output_file: str):
        """Generate structured JSON for manual MCP import."""
        print(f"\n📝 Generating structured import data...")
        
        import_data = {
            "banks": [
                {
                    "bank_id": "collabmind-platform",
                    "name": "CollabMind Platform Knowledge",
                    "mission": "Store architectural decisions, deployment patterns, and platform evolution"
                },
                {
                    "bank_id": "codebase-structure",
                    "name": "Codebase Structure",
                    "mission": "Store code relationships and implementation patterns from graphify"
                },
                {
                    "bank_id": "documentation",
                    "name": "Technical Documentation",
                    "mission": "Store deployment guides, API specs, and operational runbooks"
                }
            ],
            "memories": []
        }
        
        # Add memlord memories
        for mem in self.memlord_memories[:100]:
            import_data["memories"].append({
                "bank_id": "collabmind-platform",
                "content": f"[{mem.get('memory_type', 'unknown').upper()}] {mem.get('name', 'Untitled')}",
                "context": mem.get('memory_type', 'general'),
                "tags": mem.get('tags', []) + ["source:memlord"],
                "metadata": {
                    "source": "memlord",
                    "name": mem.get('name'),
                    "workspace": mem.get('workspace')
                }
            })
        
        # Add graphify nodes (sample)
        if self.graphify_data:
            for node in self.graphify_data.get("nodes", [])[:100]:
                import_data["memories"].append({
                    "bank_id": "codebase-structure",
                    "content": f"[{node.get('file_type', 'unknown').upper()}] {node.get('label', '')}",
                    "context": "code",
                    "tags": ["source:graphify", f"type:{node.get('file_type', 'unknown')}"],
                    "metadata": {
                        "source": "graphify",
                        "node_id": node.get('id'),
                        "source_file": node.get('source_file')
                    }
                })
        
        # Write JSON
        with open(output_file, 'w') as f:
            json.dump(import_data, f, indent=2)
        
        print(f"✅ Generated import JSON: {output_file}")
    
    def print_summary(self):
        """Print summary."""
        print("\n" + "="*60)
        print("📊 IMPORT PREPARATION SUMMARY")
        print("="*60)
        print(f"Memlord memories ready: {len(self.memlord_memories)}")
        print(f"Graphify nodes available: {len(self.graphify_data.get('nodes', [])) if self.graphify_data else 0}")
        print(f"Documentation files found: {len(self.doc_files)}")
        
        if self.stats['errors']:
            print(f"\n⚠️  Errors: {len(self.stats['errors'])}")
            for err in self.stats['errors'][:5]:
                print(f"  • {err}")
        
        print("="*60)


def main():
    """Main workflow."""
    print("🔄 Hindsight Knowledge Import Preparation")
    print("="*60)
    
    importer = KnowledgeImporter()
    
    # Load data sources
    # Note: You'll need to export memlord data first
    # For now, using the list we already have
    MEMLORD_EXPORT = "/tmp/memlord_export.json"
    GRAPHIFY_PATH = "/Users/oliververmeulen/collabmind-stack-live/graphify-out/graph.json"
    DOCS_ROOT = "/Users/oliververmeulen/collabmind-stack-live"
    
    # Check if memlord export exists, if not, create placeholder
    if not Path(MEMLORD_EXPORT).exists():
        print(f"⚠️  Memlord export not found at {MEMLORD_EXPORT}")
        print("   Will generate template - you need to populate it with actual data")
    
    importer.load_graphify_graph(GRAPHIFY_PATH)
    importer.find_documentation(DOCS_ROOT)
    
    # Generate import artifacts
    importer.generate_import_json("/Users/oliververmeulen/hindsight/import_data.json")
    importer.generate_hindsight_commands("/Users/oliververmeulen/hindsight/import_to_hindsight.sh")
    
    importer.print_summary()
    
    print("\n✅ Next steps:")
    print("   1. Review /Users/oliververmeulen/hindsight/import_data.json")
    print("   2. Edit /Users/oliververmeulen/hindsight/import_to_hindsight.sh")
    print("   3. Uncomment the hindsight_sync_retain lines")
    print("   4. Run: ./import_to_hindsight.sh")


if __name__ == "__main__":
    main()
