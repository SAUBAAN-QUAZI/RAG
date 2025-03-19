"""
Webhook Handlers
---------------
This module implements webhook handlers for Ragie events.
"""

import logging
import hmac
import hashlib
import json
from typing import Dict, Any, Optional

from fastapi import APIRouter, Request, Response, HTTPException, Depends, Header

from rag.config import RAGIE_WEBHOOK_SECRET

# Set up logger
logger = logging.getLogger(__name__)

# Create router
webhooks_router = APIRouter(prefix="/webhooks", tags=["webhooks"])

def verify_ragie_signature(
    request_body: bytes,
    signature_header: Optional[str] = Header(None, alias="X-Signature")
) -> bool:
    """
    Verify the signature from Ragie webhooks.
    
    Args:
        request_body: The raw request body
        signature_header: The X-Signature header from Ragie
        
    Returns:
        True if signature is valid, False otherwise
    """
    if not RAGIE_WEBHOOK_SECRET or not signature_header:
        return False
    
    computed_signature = hmac.new(
        RAGIE_WEBHOOK_SECRET.encode(),
        request_body,
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(signature_header, computed_signature)

@webhooks_router.post("/ragie")
async def ragie_webhook(request: Request):
    """
    Webhook handler for Ragie events.
    
    Events:
    - document_status_updated: Document processing status changed
    """
    # Get the raw request body
    body = await request.body()
    
    # Verify the signature if webhook secret is configured
    if RAGIE_WEBHOOK_SECRET:
        signature = request.headers.get("X-Signature")
        if not signature:
            logger.warning("Received webhook without signature")
            return Response(status_code=400, content="Missing signature")
        
        if not verify_ragie_signature(body, signature):
            logger.warning("Received webhook with invalid signature")
            return Response(status_code=401, content="Invalid signature")
    
    # Parse the event
    try:
        event = json.loads(body)
        event_type = event.get("type")
        
        # Process based on event type
        if event_type == "document_status_updated":
            document_id = event.get("document_id")
            status = event.get("status")
            
            logger.info(f"Document {document_id} status updated to {status}")
            
            # Update local status tracking
            # This would typically update a database or cache
            # For now, we just log the event
            
        elif event_type == "summary_created":
            document_id = event.get("document_id")
            logger.info(f"Summary created for document {document_id}")
            
        else:
            logger.warning(f"Received unknown event type: {event_type}")
        
        # Return success - always acknowledge webhooks promptly
        return {"status": "success"}
        
    except json.JSONDecodeError:
        logger.error("Failed to parse webhook payload as JSON")
        return Response(status_code=400, content="Invalid JSON")
    except Exception as e:
        logger.exception(f"Error processing webhook: {e}")
        return Response(status_code=500, content="Internal server error") 