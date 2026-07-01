#!/usr/bin/env python3
"""
Clear existing embeddings to allow dimension migration.

This script deletes rows from memory_units to enable switching
from 384-dim to 768-dim embeddings (ONNX -> nomic-embed-text).
"""

import asyncio
import sys

async def clear_memory_units_async():
    """Delete all rows from memory_units table."""
    
    # Import directly from installed hindsight_api package
    try:
        from sqlalchemy import create_engine, text
        from sqlalchemy.pool import NullPool
        from hindsight_api.config import _get_raw_config
        from hindsight_api.db_url import to_libpq_url
        from hindsight_api.pg0 import resolve_database_url
        
        # Get database URL from config
        raw_config = _get_raw_config()
        db_url = raw_config.database_url
        schema_name = raw_config.database_schema or "public"
        
        # Resolve pg0:// URLs to actual postgresql:// URLs
        resolved_url = await resolve_database_url(db_url)
        
        print(f"Connecting to database (schema: {schema_name})...")
        
        # Create engine
        engine = create_engine(to_libpq_url(resolved_url), poolclass=NullPool)
        
        with engine.connect() as conn:
            # Check current count
            count_result = conn.execute(
                text("SELECT COUNT(*) FROM public.memory_units WHERE embedding IS NOT NULL")
            )
            count = count_result.scalar()
            print(f"Found {count} rows with embeddings in memory_units table")
            
            if count == 0:
                print("✓ No rows to delete. You're ready to proceed!")
                return
            
            # Delete rows
            print(f"Deleting {count} rows...")
            result = conn.execute(text("DELETE FROM public.memory_units;"))
            conn.commit()
            
            print(f"✓ Successfully deleted {count} rows")
            print("✓ You can now restart Hindsight with 768-dim embeddings")
            
    except ImportError as e:
        print(f"Error importing hindsight_api: {e}")
        print("\nPlease ensure hindsight-api is installed:")
        print("  pip install -e ./hindsight-api-slim")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def clear_memory_units():
    """Wrapper to run async function."""
    asyncio.run(clear_memory_units_async())

if __name__ == "__main__":
    print("=" * 60)
    print("Hindsight Embeddings Migration Helper")
    print("=" * 60)
    print()
    clear_memory_units()
    print()
    print("=" * 60)
