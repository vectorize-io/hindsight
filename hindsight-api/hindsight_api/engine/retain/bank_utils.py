"""
bank profile utilities for disposition and background management.
"""

import json
import logging
import re
from typing import TypedDict

from pydantic import BaseModel, Field

from ..db_utils import acquire_with_retry
from ..memory_engine import fq_table
from ..response_models import DispositionTraits

logger = logging.getLogger(__name__)

DEFAULT_DISPOSITION = {
    "skepticism": 3,
    "literalism": 3,
    "empathy": 3,
}


class BankProfile(TypedDict):
    """Type for bank profile data."""

    name: str
    disposition: DispositionTraits
    background: str


class BackgroundMergeResponse(BaseModel):
    """LLM response for background merge with disposition inference."""

    background: str = Field(description="Merged background in first person perspective")
    disposition: DispositionTraits = Field(description="Inferred disposition traits (skepticism, literalism, empathy)")


async def get_bank_profile(pool, bank_id: str) -> BankProfile:
    """
    Get bank profile (name, disposition + background).
    Auto-creates bank with default values if not exists.

    Args:
        pool: Database connection pool
        bank_id: bank IDentifier

    Returns:
        BankProfile with name, typed DispositionTraits, and background
    """
    async with acquire_with_retry(pool) as conn:
        # Try to get existing bank
        row = await conn.fetchrow(
            f"""
            SELECT name, disposition, background
            FROM {fq_table("banks")} WHERE bank_id = $1
            """,
            bank_id,
        )

        if row:
            # asyncpg returns JSONB as a string, so parse it
            disposition_data = row["disposition"]
            if isinstance(disposition_data, str):
                disposition_data = json.loads(disposition_data)

            return BankProfile(
                name=row["name"], disposition=DispositionTraits(**disposition_data), background=row["background"]
            )

        # Bank doesn't exist, create with defaults
        await conn.execute(
            f"""
            INSERT INTO {fq_table("banks")} (bank_id, name, disposition, background)
            VALUES ($1, $2, $3::jsonb, $4)
            ON CONFLICT (bank_id) DO NOTHING
            """,
            bank_id,
            bank_id,  # Default name is the bank_id
            json.dumps(DEFAULT_DISPOSITION),
            "",
        )

        return BankProfile(name=bank_id, disposition=DispositionTraits(**DEFAULT_DISPOSITION), background="")


async def update_bank_disposition(pool, bank_id: str, disposition: dict[str, int]) -> None:
    """
    Update bank disposition traits.

    Args:
        pool: Database connection pool
        bank_id: bank IDentifier
        disposition: Dict with skepticism, literalism, empathy (all 1-5)
    """
    # Ensure bank exists first
    await get_bank_profile(pool, bank_id)

    async with acquire_with_retry(pool) as conn:
        await conn.execute(
            f"""
            UPDATE {fq_table("banks")}
            SET disposition = $2::jsonb,
                updated_at = NOW()
            WHERE bank_id = $1
            """,
            bank_id,
            json.dumps(disposition),
        )


async def merge_bank_background(pool, llm_config, bank_id: str, new_info: str, update_disposition: bool = True) -> dict:
    """
    Merge new background information with existing background using LLM.
    Normalizes to first person ("I") and resolves conflicts.
    Optionally infers disposition traits from the merged background.

    Args:
        pool: Database connection pool
        llm_config: LLM configuration for background merging
        bank_id: bank IDentifier
        new_info: New background information to add/merge
        update_disposition: If True, infer Big Five traits from background (default: True)

    Returns:
        Dict with 'background' (str) and optionally 'disposition' (dict) keys
    """
    # Get current profile
    profile = await get_bank_profile(pool, bank_id)
    current_background = profile["background"]

    # Use LLM to merge backgrounds and optionally infer disposition
    result = await _llm_merge_background(llm_config, current_background, new_info, infer_disposition=update_disposition)

    merged_background = result["background"]
    inferred_disposition = result.get("disposition")

    # Update in database
    async with acquire_with_retry(pool) as conn:
        if inferred_disposition:
            # Update both background and disposition
            await conn.execute(
                f"""
                UPDATE {fq_table("banks")}
                SET background = $2,
                    disposition = $3::jsonb,
                    updated_at = NOW()
                WHERE bank_id = $1
                """,
                bank_id,
                merged_background,
                json.dumps(inferred_disposition),
            )
        else:
            # Update only background
            await conn.execute(
                f"""
                UPDATE {fq_table("banks")}
                SET background = $2,
                    updated_at = NOW()
                WHERE bank_id = $1
                """,
                bank_id,
                merged_background,
            )

    response = {"background": merged_background}
    if inferred_disposition:
        response["disposition"] = inferred_disposition

    return response


async def _llm_merge_background(llm_config, current: str, new_info: str, infer_disposition: bool = False) -> dict:
    """
    Use LLM to intelligently merge background information.
    Optionally infer Big Five disposition traits from the merged background.

    Args:
        llm_config: LLM configuration to use
        current: Current background text
        new_info: New information to merge
        infer_disposition: If True, also infer disposition traits

    Returns:
        Dict with 'background' (str) and optionally 'disposition' (dict) keys
    """
    if infer_disposition:
        prompt = f"""You are helping maintain a memory bank's background/profile and infer their disposition. You MUST respond with ONLY valid JSON.

Current background: {current if current else "(empty)"}

New information to add: {new_info}

Instructions:
1. Merge the new information with the current background
2. If there are conflicts (e.g., different birthplaces), the NEW information overwrites the old
3. Keep additions that don't conflict
4. Output in FIRST PERSON ("I") perspective
5. Be concise - keep merged background under 500 characters
6. Infer disposition traits from the merged background (each 1-5 integer):
   - Skepticism: 1-5 (1=trusting, takes things at face value; 5=skeptical, questions everything)
   - Literalism: 1-5 (1=flexible interpretation, reads between lines; 5=literal, exact interpretation)
   - Empathy: 1-5 (1=detached, focuses on facts; 5=empathetic, considers emotional context)

CRITICAL: You MUST respond with ONLY a valid JSON object. No markdown, no code blocks, no explanations. Just the JSON.

Format:
{{
  "background": "the merged background text in first person",
  "disposition": {{
    "skepticism": 3,
    "literalism": 3,
    "empathy": 3
  }}
}}

Trait inference examples:
- "I'm a lawyer" → skepticism: 4, literalism: 5, empathy: 2
- "I'm a therapist" → skepticism: 2, literalism: 2, empathy: 5
- "I'm an engineer" → skepticism: 3, literalism: 4, empathy: 3
- "I've been burned before by trusting people" → skepticism: 5, literalism: 3, empathy: 3
- "I try to understand what people really mean" → skepticism: 3, literalism: 2, empathy: 4
- "I take contracts very seriously" → skepticism: 4, literalism: 5, empathy: 2"""
    else:
        prompt = f"""You are helping maintain a memory bank's background/profile.

Current background: {current if current else "(empty)"}

New information to add: {new_info}

Instructions:
1. Merge the new information with the current background
2. If there are conflicts (e.g., different birthplaces), the NEW information overwrites the old
3. Keep additions that don't conflict
4. Output in FIRST PERSON ("I") perspective
5. Be concise - keep it under 500 characters
6. Return ONLY the merged background text, no explanations

Merged background:"""

    try:
        # Prepare messages
        messages = [{"role": "user", "content": prompt}]

        if infer_disposition:
            # Use structured output with Pydantic model for disposition inference
            try:
                parsed = await llm_config.call(
                    messages=messages,
                    response_format=BackgroundMergeResponse,
                    scope="bank_background",
                    temperature=0.3,
                    max_completion_tokens=8192,
                )
                logger.info(f"Successfully got structured response: background={parsed.background[:100]}")

                # Convert Pydantic model to dict format
                return {"background": parsed.background, "disposition": parsed.disposition.model_dump()}
            except Exception as e:
                logger.warning(f"Structured output failed, falling back to manual parsing: {e}")
                # Fall through to manual parsing below

        # Manual parsing fallback or non-disposition merge
        content = await llm_config.call(
            messages=messages, scope="bank_background", temperature=0.3, max_completion_tokens=8192
        )

        logger.info(f"LLM response for background merge (first 500 chars): {content[:500]}")

        if infer_disposition:
            # Parse JSON response - try multiple extraction methods
            result = None

            # Method 1: Direct parse
            try:
                result = json.loads(content)
                logger.info("Successfully parsed JSON directly")
            except json.JSONDecodeError:
                pass

            # Method 2: Extract from markdown code blocks
            if result is None:
                # Remove markdown code blocks
                code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
                if code_block_match:
                    try:
                        result = json.loads(code_block_match.group(1))
                        logger.info("Successfully extracted JSON from markdown code block")
                    except json.JSONDecodeError:
                        pass

            # Method 3: Find nested JSON structure
            if result is None:
                # Look for JSON object with nested structure
                json_match = re.search(
                    r'\{[^{}]*"background"[^{}]*"disposition"[^{}]*\{[^{}]*\}[^{}]*\}', content, re.DOTALL
                )
                if json_match:
                    try:
                        result = json.loads(json_match.group())
                        logger.info("Successfully extracted JSON using nested pattern")
                    except json.JSONDecodeError:
                        pass

            # All parsing methods failed - use fallback
            if result is None:
                logger.warning(f"Failed to extract JSON from LLM response. Raw content: {content[:200]}")
                # Fallback: use new_info as background with default disposition
                return {
                    "background": new_info if new_info else current if current else "",
                    "disposition": DEFAULT_DISPOSITION.copy(),
                }

            # Validate disposition values
            disposition = result.get("disposition", {})
            for key in ["skepticism", "literalism", "empathy"]:
                if key not in disposition:
                    disposition[key] = 3  # Default to neutral
                else:
                    # Clamp to [1, 5] and convert to int
                    disposition[key] = max(1, min(5, int(disposition[key])))

            result["disposition"] = disposition

            # Ensure background exists
            if "background" not in result or not result["background"]:
                result["background"] = new_info if new_info else ""

            return result
        else:
            # Just background merge
            merged = content
            if not merged or merged.lower() in ["(empty)", "none", "n/a"]:
                merged = new_info if new_info else ""
            return {"background": merged}

    except Exception as e:
        logger.error(f"Error merging background with LLM: {e}")
        # Fallback: just append new info
        if current:
            merged = f"{current} {new_info}".strip()
        else:
            merged = new_info

        result = {"background": merged}
        if infer_disposition:
            result["disposition"] = DEFAULT_DISPOSITION.copy()
        return result


async def list_banks(pool) -> list:
    """
    List all banks in the system.

    Args:
        pool: Database connection pool

    Returns:
        List of dicts with bank_id, name, disposition, background, created_at, updated_at
    """
    async with acquire_with_retry(pool) as conn:
        rows = await conn.fetch(
            f"""
            SELECT bank_id, name, disposition, background, created_at, updated_at
            FROM {fq_table("banks")}
            ORDER BY updated_at DESC
            """
        )

        result = []
        for row in rows:
            # asyncpg returns JSONB as a string, so parse it
            disposition_data = row["disposition"]
            if isinstance(disposition_data, str):
                disposition_data = json.loads(disposition_data)

            result.append(
                {
                    "bank_id": row["bank_id"],
                    "name": row["name"],
                    "disposition": disposition_data,
                    "background": row["background"],
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                    "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
                }
            )

        return result
