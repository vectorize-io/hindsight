#!/usr/bin/env python3
"""
Batch import to Hindsight using MCP tools via subprocess.
This coordinates large-scale imports from memlord, graphify, and docs.
"""

import json
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any


def hindsight_retain(bank_id: str, content: str, context: str, tags: List[str], metadata: Dict[str, Any] = None):
    """Call hindsight_retain via MCP (simulated via direct API call)."""
    # In real scenario, this would use the MCP protocol
    # For now, we'll just print what would be imported
    print(f"[{bank_id}] [{context}] {content[:80]}...")
    return {"status": "simulated"}


def import_memlord_batch(memories: List[Dict[str, Any]], bank_id: str = "collabmind-platform"):
    """Import a batch of memlord memories."""
    print(f"\n📚 Importing {len(memories)} memlord memories to {bank_id}...")
    
    for i, mem in enumerate(memories):
        try:
            mem_type = mem.get('memory_type', 'fact').upper()
            name = mem.get('name', 'Untitled')
            content_text = mem.get('content', '')
            
            # Build rich content
            content = f"[{mem_type}] {name}\n\n{content_text}"
            
            if mem.get('metadata'):
                content += f"\n\nMetadata: {json.dumps(mem['metadata'], indent=2)}"
            
            # Build tags
            tags = mem.get('tags', [])
            tags.extend(["source:memlord", f"type:{mem.get('memory_type', 'fact')}"])
            
            if mem.get('workspace') and mem['workspace'] != "__personal_8__":
                tags.append(f"workspace:{mem['workspace']}")
            
            # Store
            result = hindsight_retain(
                bank_id=bank_id,
                content=content,
                context=mem.get('memory_type', 'general'),
                tags=tags,
                metadata={
                    "source": "memlord",
                    "name": name,
                    "workspace": mem.get('workspace'),
                    "created_at": mem.get('created_at')
                }
            )
            
            if (i + 1) % 10 == 0:
                print(f"  ✓ {i + 1}/{len(memories)} memories processed...")
                time.sleep(0.1)  # Rate limiting
        
        except Exception as e:
            print(f"  ❌ Error importing {mem.get('name', 'unknown')}: {e}")
    
    print(f"✅ Completed memlord import")


def import_graphify_batch(graph_path: str, limit: int = 200, bank_id: str = "codebase-structure"):
    """Import graphify nodes."""
    print(f"\n🕸️  Importing graphify nodes from {graph_path}...")
    
    try:
        with open(graph_path, 'r') as f:
            graph_data = json.load(f)
        
        nodes = graph_data.get("nodes", [])[:limit]
        print(f"  • Processing {len(nodes)} nodes (limit: {limit})")
        
        for i, node in enumerate(nodes):
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
                
                # Add stack tag from path
                if source_file:
                    path_parts = Path(source_file).parts
                    if len(path_parts) > 2:
                        tags.append(f"stack:{path_parts[-3] if len(path_parts) > 2 else path_parts[0]}")
                
                result = hindsight_retain(
                    bank_id=bank_id,
                    content=content,
                    context="code",
                    tags=tags,
                    metadata={
                        "source": "graphify",
                        "node_id": node.get("id"),
                        "file_type": file_type,
                        "source_file": source_file
                    }
                )
                
                if (i + 1) % 50 == 0:
                    print(f"  ✓ {i + 1}/{len(nodes)} nodes processed...")
                    time.sleep(0.1)
            
            except Exception as e:
                print(f"  ❌ Error importing node {node.get('id', 'unknown')}: {e}")
        
        print(f"✅ Completed graphify import")
    
    except Exception as e:
        print(f"❌ Error loading graphify: {e}")


def import_documentation_batch(docs_root: str, limit: int = 50, bank_id: str = "documentation"):
    """Import markdown documentation."""
    print(f"\n📄 Importing documentation from {docs_root}...")
    
    # Find markdown files
    md_files = []
    for path in Path(docs_root).rglob("*.md"):
        # Skip hidden directories and node_modules
        if any(p.startswith('.') or p == 'node_modules' for p in path.parts):
            continue
        md_files.append(path)
    
    md_files = md_files[:limit]
    print(f"  • Processing {len(md_files)} files (limit: {limit})")
    
    for i, md_file in enumerate(md_files):
        try:
            with open(md_file, 'r', encoding='utf-8', errors='ignore') as f:
                doc_content = f.read()
            
            # Skip very short files
            if len(doc_content.strip()) < 50:
                continue
            
            # Extract title
            title = md_file.stem.replace('-', ' ').replace('_', ' ').title()
            first_line = doc_content.split('\n')[0].strip()
            if first_line.startswith('#'):
                title = first_line.lstrip('#').strip()
            
            # Build tags from path
            rel_path = md_file.relative_to(docs_root)
            tags = ["source:docs", "type:documentation"]
            for part in rel_path.parts[:-1]:
                tags.append(f"path:{part}")
            
            content = f"# {title}\n\nFile: {rel_path}\n\n{doc_content}"
            
            result = hindsight_retain(
                bank_id=bank_id,
                content=content,
                context="documentation",
                tags=tags,
                metadata={
                    "source": "documentation",
                    "file_path": str(rel_path),
                    "title": title
                }
            )
            
            if (i + 1) % 10 == 0:
                print(f"  ✓ {i + 1}/{len(md_files)} docs processed...")
                time.sleep(0.1)
        
        except Exception as e:
            print(f"  ❌ Error importing {md_file}: {e}")
    
    print(f"✅ Completed documentation import")


def main():
    """Main batch import."""
    print("="*60)
    print("🔄 Hindsight Batch Import (Simulation Mode)")
    print("="*60)
    
    # This is a simulation - actual imports would use the MCP tools
    # through the hindsight_retain MCP calls shown in the conversation
    
    print("\n⚠️  This is a simulation script.")
    print("Actual imports are being done via hindsight_retain MCP calls.")
    print("\nTo perform real imports, use:")
    print("  • hindsight_retain MCP tool for each memory")
    print("  • Or hindsight_sync_retain for synchronous imports")
    
    # Load and show what would be imported
    GRAPHIFY_PATH = "/Users/oliververmeulen/collabmind-stack-live/graphify-out/graph.json"
    DOCS_ROOT = "/Users/oliververmeulen/collabmind-stack-live"
    
    # Simulate imports (would be real with MCP tools)
    import_graphify_batch(GRAPHIFY_PATH, limit=100)
    import_documentation_batch(DOCS_ROOT, limit=30)
    
    print("\n" + "="*60)
    print("✅ Simulation Complete")
    print("="*60)


if __name__ == "__main__":
    main()
