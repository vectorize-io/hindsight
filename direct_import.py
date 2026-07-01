#!/usr/bin/env python3
"""
Direct import to Hindsight using Python and available MCP tools.
This script directly calls hindsight_sync_retain to import knowledge.
"""

import json
import sys
from pathlib import Path


def import_from_json(json_path: str):
    """
    Import memories from prepared JSON file.
    
    This generates a Python script that you can run to import via MCP.
    Since we're in a scripting context, we'll use the hindsight Python MCP client.
    """
    print(f"Loading import data from {json_path}...")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"Found {len(data['memories'])} memories to import")
    print(f"Target banks: {[b['bank_id'] for b in data['banks']]}")
    
    # Generate Python code that uses hindsight MCP
    print("\n" + "="*60)
    print("IMPORT SCRIPT (copy to use with Hindsight MCP)")
    print("="*60)
    print()
    
    print("# Import via Hindsight MCP")
    print("# Run this in a context that has hindsight MCP tools available")
    print()
    
    for i, mem in enumerate(data['memories'][:10]):  # Show first 10 as example
        content = mem['content'].replace('"', '\\"').replace('\n', '\\n')
        tags = ','.join(mem.get('tags', []))
        
        tags_list = ', '.join([f"'{t}'" for t in mem.get('tags', [])])
        
        print(f"# Memory {i+1}")
        print("hindsight_sync_retain(")
        print(f"    bank_id='{mem['bank_id']}',")
        print(f"    content='{content[:200]}...',")
        print(f"    context='{mem.get('context', 'general')}',")
        print(f"    tags=[{tags_list}]")
        print(")")
        print()
    
    print(f"# ... and {len(data['memories']) - 10} more memories")
    print()


def main():
    json_path = "/Users/oliververmeulen/hindsight/import_data.json"
    
    if not Path(json_path).exists():
        print(f"Error: {json_path} not found")
        print("Run import_knowledge.py first to generate import data")
        sys.exit(1)
    
    import_from_json(json_path)
    
    print("\n✅ Import script generated above")
    print("\nTo actually import, you need to:")
    print("1. Use an MCP client that has hindsight tools")
    print("2. Or start Hindsight server and use the API directly")
    print("3. Or use the OpenCode/Claude interface with hindsight MCP")


if __name__ == "__main__":
    main()
