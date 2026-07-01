#!/usr/bin/env python3
"""
Test script to validate embedding configurations for Hindsight.
Tests initialization, dimension detection, and actual embedding generation.
"""
import asyncio
import os
import sys
from pathlib import Path

# Add hindsight-api-slim to path
sys.path.insert(0, str(Path(__file__).parent / "hindsight-api-slim"))

from hindsight_api.engine.embeddings import create_embeddings_from_env


async def test_embeddings():
    """Test current embeddings configuration."""
    print("=" * 70)
    print("HINDSIGHT EMBEDDINGS TEST")
    print("=" * 70)
    
    # Show current config
    provider = os.getenv("HINDSIGHT_API_EMBEDDINGS_PROVIDER", "NOT SET")
    print(f"\n📌 Provider: {provider}")
    
    if provider == "openai":
        print(f"   Model: {os.getenv('HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL')}")
        print(f"   Base URL: {os.getenv('HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL')}")
    elif provider == "litellm":
        print(f"   Model: {os.getenv('HINDSIGHT_API_EMBEDDINGS_LITELLM_MODEL')}")
        print(f"   Base URL: {os.getenv('HINDSIGHT_API_EMBEDDINGS_LITELLM_API_BASE')}")
    elif provider == "onnx":
        print(f"   Model: {os.getenv('HINDSIGHT_API_EMBEDDINGS_ONNX_MODEL_ID')}")
        print(f"   Dimensions: {os.getenv('HINDSIGHT_API_EMBEDDINGS_ONNX_DIMENSIONS')}")
    elif provider == "litellm-sdk":
        print(f"   Model: {os.getenv('HINDSIGHT_API_EMBEDDINGS_LITELLM_SDK_MODEL')}")
        print(f"   Base URL: {os.getenv('HINDSIGHT_API_EMBEDDINGS_LITELLM_SDK_API_BASE')}")
    
    print("\n" + "-" * 70)
    print("STEP 1: Creating embeddings instance...")
    print("-" * 70)
    
    try:
        embeddings = create_embeddings_from_env()
        print(f"✅ Created embeddings instance: {embeddings.__class__.__name__}")
    except Exception as e:
        print(f"❌ Failed to create embeddings: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "-" * 70)
    print("STEP 2: Initializing embeddings...")
    print("-" * 70)
    
    try:
        await embeddings.initialize()
        dimension = embeddings.dimension
        print(f"✅ Initialized successfully")
        print(f"   Dimension: {dimension}")
        print(f"   Provider: {embeddings.provider_name}")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "-" * 70)
    print("STEP 3: Generating test embeddings...")
    print("-" * 70)
    
    test_texts = [
        "Hindsight is a memory system for AI agents",
        "PostgreSQL with pgvector stores embeddings",
        "Text embeddings convert words to vectors"
    ]
    
    try:
        vectors = embeddings.encode(test_texts)
        print(f"✅ Generated {len(vectors)} embeddings")
        for i, vec in enumerate(vectors):
            print(f"   Text {i+1}: {len(vec)} dimensions")
            print(f"           First 5 values: {[round(v, 4) for v in vec[:5]]}")
        
        # Verify all have same dimension
        if len(set(len(v) for v in vectors)) == 1:
            print(f"\n✅ All vectors have consistent dimension: {len(vectors[0])}")
        else:
            print(f"\n❌ Inconsistent dimensions: {[len(v) for v in vectors]}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to generate embeddings: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 70)
    print("🎉 ALL TESTS PASSED")
    print("=" * 70)
    print(f"\n✅ Provider: {embeddings.provider_name}")
    print(f"✅ Dimension: {dimension}")
    print(f"✅ Model: {provider}")
    return True


if __name__ == "__main__":
    success = asyncio.run(test_embeddings())
    sys.exit(0 if success else 1)
