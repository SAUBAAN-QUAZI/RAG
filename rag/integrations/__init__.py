"""
External Integration Module
--------------------------
This module handles integrations with external services and APIs.
"""

from typing import Dict, Any

# Import all integrations for easier access
try:
    from .ragie import RagieClient
except ImportError:
    # This allows the module to load even if some integrations aren't available
    pass

__all__ = ["RagieClient"] 