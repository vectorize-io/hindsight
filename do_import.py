#!/usr/bin/env python3
"""
Direct import script using Hindsight's sync retain functionality.
This bypasses server issues by using the API directly.
"""
import asyncio
import json
import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent / "hindsight-api-slim"))

async def main():
    """Main import workflow."""
    print("🚀 Starting Hindsight Knowledge Import")
    print("="*60)
    
    # Load the prepared import data
    import_data_path = Path(__file__).parent / "import_data.json"
    
    if not import_data_path.exists():
        print(f"❌ Import data not found: {import_data_path}")
        return
    
    with open(import_data_path) as f:
        import_data = json.load(f)
    
    print(f"📚 Loaded {len(import_data['memories'])} memories to import")
    print(f"🏦 Target banks: {', '.join([b['bank_id'] for b in import_data['banks']])}")
    print()
    
    # Import using direct API calls
    try:
        from hindsight_api.core import HindsightAPI
        
        print("🔧 Initializing Hindsight API...")
        api = await HindsightAPI.create()
        print("✅ API initialized")
        print()
        
        # Import memories
        imported = 0
        errors = []
        
        for i, mem in enumerate(import_data['memories']):
            try:
                bank_id = mem['bank_id']
                content = mem['content']
                context = mem.get('context', 'general')
                tags = mem.get('tags', [])
                metadata = mem.get('metadata', {})
                
                # Import using retain
                await api.retain(
                    content=content,
                    context=context,
                    tags=tags,
                    metadata=metadata,
                    bank_id=bank_id
                )
                
                imported += 1
                if imported % 10 == 0:
                    print(f"  ✓ Imported {imported} memories...")
            
            except Exception as e:
                errors.append(f"Memory {i}: {str(e)[:100]}")
                if len(errors) <= 5:  # Only print first 5 errors
                    print(f"  ⚠️  Error importing memory {i}: {str(e)[:100]}")
        
        print()
        print("="*60)
        print(f"✅ Import complete!")
        print(f"  • Imported: {imported}")
        print(f"  • Errors: {len(errors)}")
        print("="*60)
        
        if errors and len(errors) <= 10:
            print("\nFirst errors:")
            for err in errors[:10]:
                print(f"  • {err}")
    
    except ImportError as e:
        print(f"❌ Cannot import Hindsight API: {e}")
        print("Using fallback method...")
        print("\nYou'll need to import manually using the MCP tools.")
        print("See IMPORT_STATUS.md for instructions.")
    
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
