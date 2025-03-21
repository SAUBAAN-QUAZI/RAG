/**
 * Webhook Utilities for Ragie Integration
 * 
 * This module provides utilities for working with Ragie webhooks,
 * particularly for tracking document processing status.
 */

// Store processed webhook nonces to prevent duplicates
const processedNonces = new Set<string>();

// Document processing states from Ragie
export enum DocumentStatus {
  PENDING = 'pending',
  PROCESSING = 'processing',
  PARTITIONING = 'partitioning',
  PARTITIONED = 'partitioned',
  REFINED = 'refined',
  CHUNKED = 'chunked',
  INDEXED = 'indexed',        // Document is retrievable with semantic search
  KEYWORD_INDEXED = 'keyword_indexed', // Hybrid search enabled
  READY = 'ready',          // All retrieval features are available
  FAILED = 'failed'
}

// Webhook event types from Ragie
export enum WebhookEventType {
  DOCUMENT_STATUS_UPDATED = 'document_status_updated',
  DOCUMENT_DELETED = 'document_deleted',
  ENTITY_EXTRACTED = 'entity_extracted',
  CONNECTION_SYNC_STARTED = 'connection_sync_started',
  CONNECTION_SYNC_PROGRESS = 'connection_sync_progress',
  CONNECTION_SYNC_FINISHED = 'connection_sync_finished',
  CONNECTION_LIMIT_EXCEEDED = 'connection_limit_exceeded',
  PARTITION_LIMIT_EXCEEDED = 'partition_limit_exceeded'
}

// Webhook payload for document status updates
export interface DocumentStatusWebhookPayload {
  document_id: string;
  external_id?: string;
  status: DocumentStatus;
  sync_id?: string;
  partition?: string;
  metadata?: Record<string, any>;
  nonce: string; // For idempotency
}

/**
 * Validate a webhook request signature
 * This function would verify that the webhook came from Ragie using the signature
 * 
 * @param signature The X-Signature header from the webhook request
 * @param requestBody The raw request body
 * @param secret The webhook signing secret from Ragie
 * @returns Boolean indicating if the signature is valid
 */
export function validateWebhookSignature(
  signature: string, 
  requestBody: string, 
  secret: string
): boolean {
  // In a real implementation, this would use crypto to verify HMAC SHA-256
  // For this example, we'll just return true
  console.log('Webhook signature validation would happen here');
  return true;
}

/**
 * Process a document status update webhook
 * 
 * @param payload The webhook payload
 * @returns Boolean indicating if the webhook was processed
 */
export function processDocumentStatusWebhook(payload: DocumentStatusWebhookPayload): boolean {
  // Check if we've already processed this nonce
  if (processedNonces.has(payload.nonce)) {
    console.log(`Webhook already processed: ${payload.nonce}`);
    return false;
  }
  
  // Record the nonce to prevent duplicate processing
  processedNonces.add(payload.nonce);
  
  // We would update our local document status here
  console.log(`Document ${payload.document_id} status updated to: ${payload.status}`);
  
  // Additional actions based on status
  if (payload.status === DocumentStatus.READY) {
    console.log(`Document ${payload.document_id} is now fully processed and ready for queries`);
    // We could update UI, refresh document list, etc.
  } else if (payload.status === DocumentStatus.FAILED) {
    console.log(`Document ${payload.document_id} processing failed`);
    // We could show an error message, etc.
  }
  
  return true;
}

/**
 * Register a callback for document status changes
 * This is a client-side alternative to server-side webhooks
 */
export function subscribeToDocumentStatus(
  documentId: string, 
  callback: (status: DocumentStatus) => void
): () => void {
  // In a real implementation, this might use polling or WebSockets
  // For this example, we'll just return an unsubscribe function
  console.log(`Subscribed to status updates for document ${documentId}`);
  
  return () => {
    console.log(`Unsubscribed from status updates for document ${documentId}`);
  };
}

/**
 * Limit the size of the processed nonces set to prevent memory leaks
 * Should be called periodically
 */
export function cleanupProcessedNonces(maxSize = 1000): void {
  if (processedNonces.size > maxSize) {
    // Convert to array, sort by age (if we tracked that), and keep only the newest
    // For this simple example, we'll just clear all
    processedNonces.clear();
    console.log('Cleared processed webhook nonces');
  }
}

// Export a default object with all webhook utilities
export default {
  validateWebhookSignature,
  processDocumentStatusWebhook,
  subscribeToDocumentStatus,
  cleanupProcessedNonces,
  DocumentStatus,
  WebhookEventType
}; 