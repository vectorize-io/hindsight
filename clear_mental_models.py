#!/usr/bin/env python3
"""Clear mental_models table to allow embedding dimension migration."""
import asyncio
import asyncpg
import os
from hindsight_api.pg0 import resolve_database_url

async def clear_mental_models():
    # Get the database URL (handles embedded pg0)
    db_url_raw = os.getenv("HINDSIGHT_API_DATABASE_URL", "pg0://hindsight")
    db_url = await resolve_database_url(db_url_raw)
    
    # Connect and clear
    conn = await asyncpg.connect(db_url)
    
    try:
        result = await conn.execute("DELETE FROM public.mental_models")
        print(f"✓ Cleared mental_models table: {result}")
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(clear_mental_models())
