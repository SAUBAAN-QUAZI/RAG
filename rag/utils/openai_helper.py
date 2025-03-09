"""
OpenAI Helper
-----------
Helper functions for safely initializing and using OpenAI clients.
"""

import os
import json
import httpx
from typing import Dict, Any, Optional, Union, Mapping, Sequence
from openai import OpenAI
from openai._base_client import SyncAPIClient, make_request_options
from openai._client import OpenAI as BaseOpenAI
from openai._types import NOT_GIVEN, Headers, Query, Body
from rag.utils.utils import logger

class ProxyFreeHTTPTransport(httpx.HTTPTransport):
    """Custom HTTP Transport that explicitly ignores proxies."""
    def __init__(self, *args, **kwargs):
        # Remove any proxy-related arguments
        if 'proxy' in kwargs:
            del kwargs['proxy']
        if 'proxies' in kwargs:
            del kwargs['proxies']
        # Initialize with explicit no-proxy setting
        super().__init__(*args, **kwargs)

class ProxyFreeHTTPClient(httpx.Client):
    """Custom HTTP Client that uses the proxy-free transport."""
    def __init__(self, *args, **kwargs):
        # Remove any proxy settings
        if 'proxy' in kwargs:
            del kwargs['proxy']
        if 'proxies' in kwargs:
            del kwargs['proxies']
        
        # Use our custom transport
        kwargs['transport'] = ProxyFreeHTTPTransport()
        super().__init__(*args, **kwargs)

class CustomSyncHttpxClientWrapper:
    """A custom wrapper for httpx that ignores all proxies."""
    
    def __init__(
        self,
        timeout: float = 60.0,
        max_retries: int = 2,
        limits: Optional[httpx.Limits] = None,
        **kwargs: Any,
    ) -> None:
        # Remove any proxy settings
        if 'proxy' in kwargs:
            del kwargs['proxy']
        if 'proxies' in kwargs:
            del kwargs['proxies']

        self._timeout = timeout
        self._client = ProxyFreeHTTPClient(
            timeout=timeout,
            max_retries=max_retries,
            limits=limits,
            **kwargs,
        )

    def close(self) -> None:
        self._client.close()

    def request(
        self,
        method: str,
        url: str,
        *,
        params: Optional[Union[Mapping[str, Any], Sequence[tuple[str, Any]]]] = None,
        headers: Optional[Mapping[str, str]] = None,
        json_data: Any = None,
        content: Optional[bytes] = None,
    ) -> httpx.Response:
        kwargs = make_request_options(
            params=params,
            headers=headers,
            json=json_data,
            content=content,
        )
        return self._client.request(method, url, **kwargs)

class ProxyFreeOpenAI(BaseOpenAI):
    """A version of the OpenAI client that ignores proxy settings."""
    
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        timeout: Union[float, httpx.Timeout, None] = 60.0,
        max_retries: int = 2,
        **kwargs: Any,
    ):
        # Remove any proxy settings
        if 'proxy' in kwargs:
            del kwargs['proxy']
        if 'proxies' in kwargs:
            del kwargs['proxies']
        
        # Create a custom HTTP client
        http_client = CustomSyncHttpxClientWrapper(
            timeout=timeout,
            max_retries=max_retries,
            follow_redirects=True,
        )
        
        # Initialize the base client with our custom HTTP client
        super().__init__(
            api_key=api_key,
            http_client=http_client,
            **kwargs
        )

def create_openai_client(api_key: str) -> OpenAI:
    """
    Create an OpenAI client with safety measures to prevent proxy issues.
    
    This function creates a custom OpenAI client that completely ignores 
    proxy settings, avoiding the 'proxies' parameter error.
    
    Args:
        api_key: OpenAI API key to use
        
    Returns:
        OpenAI: Initialized OpenAI client
    """
    # Store and remove all environment variables that might affect the OpenAI client
    env_backup = {}
    proxy_vars = [
        'http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY',
        'OPENAI_PROXY', 'openai_proxy', 'OPENAI_HTTP_PROXY', 'OPENAI_HTTPS_PROXY',
        'all_proxy', 'ALL_PROXY', 'no_proxy', 'NO_PROXY'
    ]
    
    # Backup and remove all proxy-related environment variables
    for var in proxy_vars:
        if var in os.environ:
            env_backup[var] = os.environ[var]
            del os.environ[var]
    
    # Create an explicit no-proxy environment
    os.environ['no_proxy'] = '*'
    os.environ['NO_PROXY'] = '*'
    
    try:
        # Try creating our proxy-free client
        logger.info("Initializing ProxyFreeOpenAI client")
        client = ProxyFreeOpenAI(api_key=api_key)
        logger.info("Successfully initialized OpenAI client with proxy safeguards")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        # As a last resort, try the simplest possible approach
        try:
            from openai import Client
            logger.warning("Trying alternate initialization method: direct Client import")
            client = Client(api_key=api_key)
            return client
        except Exception as e2:
            logger.error(f"All client initialization methods failed: {e2}")
            raise
    finally:
        # Restore environment variables
        for var, value in env_backup.items():
            os.environ[var] = value
        
        # Remove our temporary settings
        if 'no_proxy' not in env_backup and 'no_proxy' in os.environ:
            del os.environ['no_proxy']
        if 'NO_PROXY' not in env_backup and 'NO_PROXY' in os.environ:
            del os.environ['NO_PROXY'] 