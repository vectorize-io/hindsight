#!/usr/bin/env python3
"""
Check database status and embeddings count.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "hindsight-api-slim"))

from hindsight_api.config import _get_raw_config
from hindsight_api.pg0 import resolve_database_url
import asyncpg


async def check_database():
    """Check database status."""
    print("=" * 70)
    print("HINDSIGHT DATABASE CHECK")
    print("=" * 70)
    
    # Get database URL from config
    raw_config = _get_raw_config()
    db_url_raw = raw_config.database_url
    
    # Resolve database URL (handles embedded pg0)
    db_url = await resolve_database_url(db_url_raw)
    print(f"\n📌 Database URL: {db_url[:50]}...")
    
    # Connect to database
    conn = await asyncpg.connect(db_url)
    
    try:
        # Check if memory_units table exists
        table_check = await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'memory_units')"
        )
        
        if not table_check:
            print("\n❌ memory_units table does not exist")
            print("   Run Hindsight migrations first")
            return False
        
        # Count rows with embeddings
        count = await conn.fetchval(
            "SELECT COUNT(*) FROM public.memory_units WHERE embedding IS NOT NULL"
        )
        
        print(f"\n✅ memory_units table exists")
        print(f"   Rows with embeddings: {count}")
        
        if count == 0:
            print("\n✅ Database is ready for new embedding dimension")
        else:
            # Show first row dimension
            first_dim = await conn.fetchval(
                "SELECT array_length(embedding, 1) FROM public.memory_units WHERE embedding IS NOT NULL LIMIT 1"
            )
            print(f"\n⚠️  Database contains {count} rows with {first_dim}-dimensional embeddings")
            print("   You may need to clear embeddings if changing dimension")
        
    finally:
        await conn.close()
    
    print("\n" + "=" * 70)
    return True


if __name__ == "__main__":
    success = asyncio.run(check_database())
    sys.exit(0 if success else 1)
