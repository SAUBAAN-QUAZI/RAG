"""
External Integration Module
--------------------------
This module handles integrations with external services and APIs.

Primary integrations include:
- Ragie.ai - A managed RAG service with document processing and retrieval capabilities
"""

from typing import Dict, Any, Optional, List

# Import all integrations for easier access
try:
    from .ragie import RagieClient, RAGIE_AVAILABLE
except ImportError:
    # This allows the module to load even if some integrations aren't available
    RAGIE_AVAILABLE = False

__all__ = ["RagieClient", "RAGIE_AVAILABLE", "create_ragie_client"]

def create_ragie_client(api_key: Optional[str] = None, default_partition: Optional[str] = None) -> Optional["RagieClient"]:
    """
    Factory function to create a Ragie client if available.
    
    Args:
        api_key: Optional API key for Ragie (uses env var if not provided)
        default_partition: Optional default partition to use
        
    Returns:
        RagieClient instance or None if Ragie integration is not available
    """
    if not RAGIE_AVAILABLE:
        return None
    
    try:
        from .ragie import RagieClient
        return RagieClient(api_key=api_key, default_partition=default_partition)
    except (ImportError, Exception) as e:
        import logging
        logging.getLogger(__name__).error(f"Failed to create Ragie client: {e}")
        return None 