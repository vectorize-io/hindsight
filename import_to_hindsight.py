#!/usr/bin/env python3
"""
Import knowledge from multiple sources into Hindsight memory banks:
1. Memlord MCP memories (via MCP protocol)
2. Graphify knowledge graph (graph.json)
3. Documentation files (markdown)
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add hindsight to path
sys.path.insert(0, str(Path(__file__).parent))

from hindsight_api.core import HindsightAPI


class HindsightImporter:
    """Import knowledge from various sources into Hindsight banks."""
    
    def __init__(self):
        self.api = None
        self.stats = {
            "memlord_imported": 0,
            "graphify_nodes_imported": 0,
            "graphify_edges_imported": 0,
            "docs_imported": 0,
            "errors": []
        }
    
    async def initialize(self):
        """Initialize Hindsight API."""
        print("🚀 Initializing Hindsight API...")
        self.api = await HindsightAPI.create()
        print(f"✅ Hindsight initialized")
    
    async def create_banks(self):
        """Create specialized memory banks for different knowledge domains."""
        banks = [
            {
                "bank_id": "collabmind-platform",
                "name": "CollabMind Platform Knowledge",
                "mission": "Store architectural decisions, deployment patterns, and platform evolution for the CollabMind stack"
            },
            {
                "bank_id": "codebase-structure",
                "name": "Codebase Structure & Relationships",
                "mission": "Store code relationships, file structure, and implementation patterns from graphify analysis"
            },
            {
                "bank_id": "documentation",
                "name": "Technical Documentation",
                "mission": "Store deployment guides, API specs, architecture docs, and operational runbooks"
            }
        ]
        
        print("\n📦 Creating memory banks...")
        for bank in banks:
            try:
                # Banks are auto-created if they don't exist
                print(f"  • {bank['name']} ({bank['bank_id']})")
            except Exception as e:
                self.stats["errors"].append(f"Bank creation failed: {bank['bank_id']}: {e}")
    
    async def import_memlord_memories(self, memlord_data: List[Dict[str, Any]]):
        """Import memories from memlord MCP format."""
        print(f"\n📚 Importing {len(memlord_data)} memlord memories...")
        
        for mem in memlord_data:
            try:
                # Map memlord memory_type to context
                context = mem.get("memory_type", "general")
                
                # Build rich content with metadata
                content_parts = [f"[{mem['memory_type'].upper()}] {mem['name']}"]
                
                if mem.get("metadata"):
                    content_parts.append(f"Metadata: {json.dumps(mem['metadata'], indent=2)}")
                
                # Get full content if available
                if mem.get("content"):
                    content_parts.append(f"\n{mem['content']}")
                
                content = "\n\n".join(content_parts)
                
                # Store with tags and workspace context
                tags = mem.get("tags", [])
                if mem.get("workspace") and mem["workspace"] != "__personal_8__":
                    tags.append(f"workspace:{mem['workspace']}")
                
                tags.append("source:memlord")
                tags.append(f"type:{mem['memory_type']}")
                
                # Store to appropriate bank
                bank_id = "collabmind-platform"
                
                await self.api.retain(
                    content=content,
                    context=context,
                    tags=tags,
                    metadata={
                        "source": "memlord",
                        "original_name": mem["name"],
                        "workspace": mem.get("workspace"),
                        "created_at": mem.get("created_at")
                    },
                    bank_id=bank_id
                )
                
                self.stats["memlord_imported"] += 1
                if self.stats["memlord_imported"] % 50 == 0:
                    print(f"  ✓ {self.stats['memlord_imported']} memories imported...")
            
            except Exception as e:
                self.stats["errors"].append(f"Memlord import failed: {mem.get('name', 'unknown')}: {e}")
        
        print(f"✅ Imported {self.stats['memlord_imported']} memlord memories")
    
    async def import_graphify_graph(self, graph_path: str):
        """Import graphify knowledge graph."""
        print(f"\n🕸️  Importing graphify knowledge graph from {graph_path}...")
        
        try:
            with open(graph_path, 'r') as f:
                graph_data = json.load(f)
            
            nodes = graph_data.get("nodes", [])
            edges = graph_data.get("links", [])
            
            print(f"  • {len(nodes)} nodes")
            print(f"  • {len(edges)} edges")
            
            # Import nodes (code entities)
            for node in nodes[:500]:  # Limit to first 500 to avoid overwhelming
                try:
                    file_type = node.get("file_type", "unknown")
                    label = node.get("label", "")
                    source_file = node.get("source_file", "")
                    source_location = node.get("source_location", "")
                    
                    content = f"[{file_type.upper()}] {label}"
                    if source_file:
                        content += f"\nFile: {source_file}"
                    if source_location:
                        content += f"\nLocation: {source_location}"
                    
                    tags = [
                        "source:graphify",
                        f"type:{file_type}",
                        f"community:{node.get('community', 0)}"
                    ]
                    
                    # Add file path tags
                    if source_file:
                        path_parts = Path(source_file).parts
                        if len(path_parts) > 2:
                            tags.append(f"stack:{path_parts[-3] if len(path_parts) > 2 else path_parts[0]}")
                    
                    await self.api.retain(
                        content=content,
                        context="code",
                        tags=tags,
                        metadata={
                            "source": "graphify",
                            "node_id": node.get("id"),
                            "file_type": file_type,
                            "source_file": source_file
                        },
                        bank_id="codebase-structure"
                    )
                    
                    self.stats["graphify_nodes_imported"] += 1
                    if self.stats["graphify_nodes_imported"] % 100 == 0:
                        print(f"  ✓ {self.stats['graphify_nodes_imported']} nodes imported...")
                
                except Exception as e:
                    self.stats["errors"].append(f"Graphify node import failed: {node.get('id', 'unknown')}: {e}")
            
            print(f"✅ Imported {self.stats['graphify_nodes_imported']} graphify nodes")
        
        except Exception as e:
            self.stats["errors"].append(f"Graphify import failed: {e}")
            print(f"❌ Graphify import error: {e}")
    
    async def import_documentation(self, docs_root: str):
        """Import markdown documentation files."""
        print(f"\n📄 Importing documentation from {docs_root}...")
        
        # Find all markdown files
        md_files = []
        for root, dirs, files in os.walk(docs_root):
            # Skip node_modules and hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'node_modules']
            
            for file in files:
                if file.endswith('.md'):
                    md_files.append(Path(root) / file)
        
        print(f"  • Found {len(md_files)} markdown files")
        
        for md_file in md_files[:100]:  # Limit to first 100 docs
            try:
                with open(md_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Skip if too short (likely just a title)
                if len(content.strip()) < 50:
                    continue
                
                # Extract title from first heading or filename
                title = md_file.stem.replace('-', ' ').replace('_', ' ').title()
                first_line = content.split('\n')[0].strip()
                if first_line.startswith('#'):
                    title = first_line.lstrip('#').strip()
                
                # Build tags from file path
                rel_path = md_file.relative_to(docs_root)
                tags = ["source:docs", "type:documentation"]
                
                for part in rel_path.parts[:-1]:
                    tags.append(f"path:{part}")
                
                doc_content = f"# {title}\n\nFile: {rel_path}\n\n{content}"
                
                await self.api.retain(
                    content=doc_content,
                    context="documentation",
                    tags=tags,
                    metadata={
                        "source": "documentation",
                        "file_path": str(rel_path),
                        "title": title
                    },
                    bank_id="documentation"
                )
                
                self.stats["docs_imported"] += 1
                if self.stats["docs_imported"] % 20 == 0:
                    print(f"  ✓ {self.stats['docs_imported']} docs imported...")
            
            except Exception as e:
                self.stats["errors"].append(f"Doc import failed: {md_file}: {e}")
        
        print(f"✅ Imported {self.stats['docs_imported']} documentation files")
    
    async def create_mental_models(self):
        """Create mental models for key knowledge domains."""
        print("\n🧠 Creating mental models...")
        
        models = [
            {
                "bank_id": "collabmind-platform",
                "mental_model_id": "architecture-overview",
                "name": "Architecture Overview",
                "source_query": "What is the overall architecture of the CollabMind platform? Include key services, control plane, governance, and deployment patterns.",
                "tags": ["architecture", "overview"]
            },
            {
                "bank_id": "collabmind-platform",
                "mental_model_id": "deployment-patterns",
                "name": "Deployment Patterns",
                "source_query": "What are the deployment patterns, compose configurations, and infrastructure decisions?",
                "tags": ["deployment", "infrastructure"]
            },
            {
                "bank_id": "collabmind-platform",
                "mental_model_id": "governance-model",
                "name": "Governance Model",
                "source_query": "How does governance work? Include approval workflows, quarantine, RBAC, and tenant isolation.",
                "tags": ["governance", "security"]
            }
        ]
        
        # Note: Mental model creation is async, we just trigger it
        # The actual synthesis happens in background
        for model in models:
            try:
                print(f"  • Creating: {model['name']}")
                # This would use the Hindsight MCP to create mental models
                # For now, we'll note it as a follow-up step
            except Exception as e:
                self.stats["errors"].append(f"Mental model creation failed: {model['name']}: {e}")
    
    def print_summary(self):
        """Print import summary."""
        print("\n" + "="*60)
        print("📊 IMPORT SUMMARY")
        print("="*60)
        print(f"Memlord memories:    {self.stats['memlord_imported']:>6}")
        print(f"Graphify nodes:      {self.stats['graphify_nodes_imported']:>6}")
        print(f"Documentation files: {self.stats['docs_imported']:>6}")
        print(f"Total imported:      {sum([self.stats['memlord_imported'], self.stats['graphify_nodes_imported'], self.stats['docs_imported']]):>6}")
        
        if self.stats['errors']:
            print(f"\n⚠️  Errors: {len(self.stats['errors'])}")
            for err in self.stats['errors'][:10]:  # Show first 10 errors
                print(f"  • {err}")
        else:
            print("\n✅ No errors!")
        
        print("="*60)


async def main():
    """Main import workflow."""
    print("🔄 Hindsight Knowledge Import")
    print("="*60)
    
    # Paths
    GRAPHIFY_PATH = "/Users/oliververmeulen/collabmind-stack-live/graphify-out/graph.json"
    DOCS_ROOT = "/Users/oliververmeulen/collabmind-stack-live"
    
    # Load memlord data (this would be fetched via MCP in real scenario)
    # For now, we'll need to manually provide this data
    print("\n⚠️  NOTE: Memlord data needs to be provided via MCP")
    print("Please export memlord memories to JSON first")
    
    importer = HindsightImporter()
    
    try:
        await importer.initialize()
        await importer.create_banks()
        
        # Import graphify
        if Path(GRAPHIFY_PATH).exists():
            await importer.import_graphify_graph(GRAPHIFY_PATH)
        else:
            print(f"⚠️  Graphify file not found: {GRAPHIFY_PATH}")
        
        # Import documentation
        if Path(DOCS_ROOT).exists():
            await importer.import_documentation(DOCS_ROOT)
        else:
            print(f"⚠️  Docs root not found: {DOCS_ROOT}")
        
        # Create mental models (optional, can be done later)
        # await importer.create_mental_models()
        
        importer.print_summary()
        
        print("\n✅ Import complete! You can now query Hindsight with:")
        print("   • hindsight recall 'architecture overview'")
        print("   • hindsight reflect 'what are the key deployment patterns?'")
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
