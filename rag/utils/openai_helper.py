"""
OpenAI Helper
-----------
Helper functions for safely initializing and using OpenAI clients.
"""

import os
import json
from openai import OpenAI
from rag.utils import logger

def create_openai_client(api_key):
    """
    Create an OpenAI client with safety measures to prevent proxy issues.
    
    This function completely isolates the OpenAI client initialization from any
    system environment variables that might cause problems.
    
    Args:
        api_key: OpenAI API key to use
        
    Returns:
        OpenAI: Initialized OpenAI client
    """
    # Store all environment variables that might affect the OpenAI client
    env_backup = {}
    proxy_vars = [
        'http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY',
        'OPENAI_PROXY', 'openai_proxy'
    ]
    
    # Backup and remove all proxy-related environment variables
    for var in proxy_vars:
        if var in os.environ:
            env_backup[var] = os.environ[var]
            del os.environ[var]
    
    try:
        # Create minimal configuration with just the API key
        client_kwargs = {'api_key': api_key}
        
        # Log attempt (but don't log API key)
        logger.info("Initializing OpenAI client with minimal configuration")
        
        # Create the client
        client = OpenAI(**client_kwargs)
        logger.info("Successfully initialized OpenAI client")
        return client
    except Exception as e:
        # If that still doesn't work, try even more minimal approach
        logger.warning(f"Error initializing OpenAI client: {e}, trying alternate method")
        try:
            # Try importing the class directly and using minimal args
            from openai._client import OpenAI as DirectOpenAI
            client = DirectOpenAI(api_key=api_key)
            logger.info("Successfully initialized OpenAI client using alternate method")
            return client
        except Exception as e2:
            logger.error(f"Failed to initialize OpenAI client with alternate method: {e2}")
            raise
    finally:
        # Restore environment variables
        for var, value in env_backup.items():
            os.environ[var] = value 