"""
Optimized Hindsight Memory API Client.

Features:
- Connection pooling via requests.Session
- Exponential backoff retry logic
- Proper timeout handling
- Singleton pattern for efficiency
- Multi-bank support
"""

import logging
import time
from functools import lru_cache
from typing import Optional, Dict, Any, List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class HindsightClientError(Exception):
    """Custom exception for Hindsight API errors."""
    pass


class HindsightClient:
    """
    High-performance client for Hindsight Memory API.
    
    Uses connection pooling and retry logic for optimal performance.
    Thread-safe singleton per base_url.
    """
    
    _instances: Dict[str, "HindsightClient"] = {}
    
    def __new__(cls, base_url: str, default_bank_id: str = "hindsight_agent_bank"):
        """Singleton pattern - reuse client instances per base_url."""
        if base_url not in cls._instances:
            instance = super().__new__(cls)
            instance._initialized = False
            cls._instances[base_url] = instance
        return cls._instances[base_url]
    
    def __init__(self, base_url: str, default_bank_id: str = "hindsight_agent_bank"):
        if self._initialized:
            # Update default bank if needed
            self.default_bank_id = default_bank_id
            return
            
        self.base_url = base_url.rstrip('/')
        self.default_bank_id = default_bank_id
        
        # Create session with connection pooling
        self.session = requests.Session()
        
        # Configure retry strategy with exponential backoff
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,  # 0.5, 1.0, 2.0 seconds
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "DELETE"],
            raise_on_status=False
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=20
        )
        
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Default headers
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "HindsightAgent/2.0"
        })
        
        # Default timeouts
        self.default_timeout = 30
        self.reflect_timeout = 120  # Reflect operations take longer
        
        self._initialized = True
        logger.info(f"HindsightClient initialized: {self.base_url}")
    
    def _get_bank_url(self, bank_id: Optional[str] = None) -> str:
        """Get the base URL for a memory bank."""
        target_bank = bank_id or self.default_bank_id
        return f"{self.base_url}/v1/default/banks/{target_bank}"
    
    def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        payload: Optional[Dict] = None,
        timeout: Optional[int] = None,
        bank_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Make an HTTP request with proper error handling.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (e.g., /memories/recall)
            payload: JSON payload for POST requests
            timeout: Request timeout in seconds
            bank_id: Target memory bank
            
        Returns:
            JSON response as dict
            
        Raises:
            HindsightClientError: On API errors
        """
        url = f"{self._get_bank_url(bank_id)}{endpoint}"
        timeout = timeout or self.default_timeout
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url, timeout=timeout)
            elif method.upper() == "POST":
                response = self.session.post(url, json=payload, timeout=timeout)
            elif method.upper() == "DELETE":
                response = self.session.delete(url, timeout=timeout)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            # Check for errors
            if response.status_code >= 400:
                error_msg = f"API error {response.status_code}: {response.text[:200]}"
                logger.error(error_msg)
                return {"error": error_msg, "status_code": response.status_code}
            
            return response.json()
            
        except requests.exceptions.Timeout:
            error_msg = f"Request timed out after {timeout}s: {url}"
            logger.error(error_msg)
            return {"error": error_msg, "timeout": True}
            
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection error: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg, "connection_error": True}
            
        except json.JSONDecodeError:
            error_msg = "Invalid JSON response from API"
            logger.error(error_msg)
            return {"error": error_msg, "response_text": response.text[:200]}
            
        except Exception as e:
            # Don't leak internal details
            logger.exception(f"Unexpected error calling {endpoint}")
            return {"error": "An unexpected error occurred."}
    
    def recall(
        self, 
        query: str, 
        max_tokens: int = 4096, 
        budget: str = "mid",
        bank_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search and retrieve memories from long-term storage.
        
        Args:
            query: Natural language search query
            max_tokens: Maximum tokens to return
            budget: Search budget (low/mid/high) - affects thoroughness
            bank_id: Specific memory bank to query
            
        Returns:
            Dict with 'results' list of matching memories
        """
        payload = {
            "query": query,
            "max_tokens": max_tokens,
            "budget": budget
        }
        
        result = self._make_request(
            "POST", 
            "/memories/recall", 
            payload=payload,
            bank_id=bank_id
        )
        
        # Ensure results key exists
        if "error" not in result and "results" not in result:
            result["results"] = []
            
        return result
    
    def retain(
        self, 
        content: str, 
        context: str = "general",
        bank_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Store new information to long-term memory.
        
        Args:
            content: The fact/memory to store
            context: Category for the memory
            bank_id: Specific memory bank to store in
            
        Returns:
            Dict with storage confirmation
        """
        payload = {
            "items": [
                {"content": content, "context": context}
            ]
        }
        
        return self._make_request(
            "POST",
            "/memories",
            payload=payload,
            bank_id=bank_id
        )
    
    def retain_batch(
        self,
        items: List[Dict[str, str]],
        bank_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Store multiple memories in a single request.
        
        Args:
            items: List of dicts with 'content' and optional 'context'
            bank_id: Specific memory bank to store in
            
        Returns:
            Dict with storage confirmation
        """
        # Ensure each item has context
        normalized_items = [
            {"content": item["content"], "context": item.get("context", "general")}
            for item in items
        ]
        
        payload = {"items": normalized_items}
        
        return self._make_request(
            "POST",
            "/memories",
            payload=payload,
            bank_id=bank_id
        )
    
    def reflect(
        self, 
        query: str, 
        budget: str = "low",
        bank_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Synthesize memories to form opinions or answer complex questions.
        
        Args:
            query: Topic to reflect on
            budget: Reflection budget (affects depth)
            bank_id: Specific memory bank to use
            
        Returns:
            Dict with reflection result
        """
        payload = {
            "query": query,
            "budget": budget
        }
        
        return self._make_request(
            "POST",
            "/memories/reflect",
            payload=payload,
            timeout=self.reflect_timeout,
            bank_id=bank_id
        )
    
    def health_check(self) -> bool:
        """
        Check if the Hindsight API is reachable.
        
        Returns:
            True if API is healthy, False otherwise
        """
        try:
            # 1. Simple reachability check
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=5
            )
            if response.status_code != 200:
                logger.warning(f"Health check failed with status {response.status_code}")
                return False

            # 2. Verify basic auth/connectivity if possible (optional, depending on API)
            # For Hindsight, listing banks is a good lightweight check
            # response = self.session.get(f"{self.base_url}/v1/default/banks", timeout=5)
            # return response.status_code == 200
            
            return True
        except Exception:
            return False
    
    def list_banks(self) -> Dict[str, Any]:
        """
        List all available memory banks.
        
        Returns:
            Dict with list of banks
        """
        try:
            response = self.session.get(
                f"{self.base_url}/v1/default/banks",
                timeout=self.default_timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e), "banks": []}
    
    def close(self):
        """Close the HTTP session."""
        self.session.close()
        # Remove from instances cache
        if self.base_url in self._instances:
            del self._instances[self.base_url]
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


@lru_cache(maxsize=1)
def get_default_client() -> HindsightClient:
    """
    Get a cached default HindsightClient instance.
    
    Uses configuration from config.py.
    """
    from config import get_config
    settings = get_config()
    return HindsightClient(
        base_url=settings.mcp_base_url,
        default_bank_id=settings.default_bank_id
    )


# Memory bank constants for type safety
class MemoryBanks:
    """Available memory banks."""
    USER_PREFERENCES = "user_preferences"
    PROJECT_CONTEXT = "project_context"
    KNOWLEDGE_BASE = "knowledge_base"
    HINDSIGHT_AGENT = "hindsight_agent_bank"


if __name__ == "__main__":
    # Test the client
    from config import get_config
    
    settings = get_config()
    client = HindsightClient(settings.mcp_base_url, settings.default_bank_id)
    
    print(f"Testing HindsightClient...")
    print(f"Base URL: {client.base_url}")
    print(f"Default Bank: {client.default_bank_id}")
    
    # Health check
    if client.health_check():
        print("✓ API is healthy")
    else:
        print("✗ API health check failed")
    
    # Test recall
    print("\nTesting recall...")
    result = client.recall("test query", max_tokens=100)
    print(f"Recall result: {json.dumps(result, indent=2)[:200]}")
