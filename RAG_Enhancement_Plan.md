 # RAG System Enhancement Plan

## Overview

This document outlines the implementation plan for four key enhancements to our Ragie-powered RAG system. These improvements will enhance user experience, provide more relevant results, and optimize performance.

## 1. Metadata Filtering

### Description
Add UI controls that allow users to filter documents by metadata fields (e.g., document type, creation date, author, department) when querying the system.

### Value Proposition
- **Enhanced Precision**: Users can narrow searches to specific document subsets
- **Reduced Noise**: Exclude irrelevant documents from consideration
- **Domain-Specific Queries**: Easily target queries to particular departments or content types

### Technical Approach

#### Frontend Changes
1. Add a collapsible "Advanced Filters" section to the Chat settings drawer
2. Dynamically generate filter controls based on available metadata fields
3. Create a metadata filter builder UI component

#### Backend Changes
1. Enhance the `/api/query` endpoint to accept and process metadata filters
2. Map frontend filter structure to Ragie's filter format
3. Add validation for filter parameters

### Implementation Steps

1. **Backend API Enhancement** (3 days)
   - Update the `QueryRequest` model to include metadata filters
   - Modify the Ragie integration to translate filters to Ragie's format
   - Add validation for filter parameters
   - Test filter translation with various query patterns

2. **Frontend Filter UI** (4 days)
   - Create a `MetadataFilterBuilder` component
   - Implement UI controls for different metadata types (strings, dates, numbers, booleans)
   - Add operator selection (equals, contains, greater than, etc.)
   - Design clear visual indicators for active filters

3. **Document Metadata Discovery** (2 days)
   - Implement API endpoint to retrieve available metadata fields and their types
   - Create a discovery mechanism to identify common metadata across documents
   - Cache metadata schema to avoid repeated API calls

4. **Testing & Refinement** (2 days)
   - Test with various document collections and metadata patterns
   - Optimize filter UI for mobile devices
   - Add user-friendly tooltips and guidance

#### Example API Request
```json
{
  "query": "budget forecasts",
  "document_ids": ["doc1", "doc2"],
  "metadata_filter": {
    "department": {"$eq": "finance"},
    "date": {"$gte": "2023-01-01"},
    "$or": [
      {"document_type": {"$eq": "spreadsheet"}},
      {"document_type": {"$eq": "report"}}
    ]
  },
  "rerank": true,
  "top_k": 5
}
```

#### Example UI Component Structure
```tsx
// MetadataFilterBuilder.tsx
<VStack spacing={4}>
  <HStack>
    <Select value={field} onChange={handleFieldChange}>
      {availableFields.map(f => <option key={f.name} value={f.name}>{f.displayName}</option>)}
    </Select>
    <Select value={operator} onChange={handleOperatorChange}>
      {operators.map(op => <option key={op.value} value={op.value}>{op.label}</option>)}
    </Select>
    <Input value={filterValue} onChange={handleValueChange} />
    <IconButton icon={<FiPlus />} onClick={addFilter} />
  </HStack>
  
  <Box p={2} borderWidth={1} borderRadius="md">
    <Text fontSize="sm" fontWeight="bold">Active Filters:</Text>
    {filters.map((filter, index) => (
      <Tag key={index} m={1}>
        {filter.field} {filter.operator} {filter.value}
        <TagCloseButton onClick={() => removeFilter(index)} />
      </Tag>
    ))}
  </Box>
</VStack>
```

## 2. Relevance Feedback

### Description
Allow users to provide feedback on response quality, which can be used to improve retrieval algorithms and potentially fine-tune the underlying models.

### Value Proposition
- **Continuous Improvement**: System learns from user feedback to deliver more relevant results
- **User Engagement**: Users feel empowered to improve system performance
- **Quantifiable Metrics**: Provides data for measuring and improving RAG quality

### Technical Approach

#### Frontend Changes
1. Add thumbs up/down buttons after each AI response
2. Implement detailed feedback modal for negative ratings
3. Track feedback statistics in the UI

#### Backend Changes
1. Create a feedback collection API endpoint
2. Store feedback data with query/response pairs
3. Implement feedback analysis tools for system improvement

### Implementation Steps

1. **Feedback UI Components** (3 days)
   - Add simple thumbs up/down controls after each response
   - Create expandable feedback form for detailed input
   - Implement feedback categories (irrelevant, incomplete, incorrect, etc.)
   - Add optional free-text comment field

2. **Feedback API & Storage** (2 days)
   - Design and implement feedback data model
   - Create API endpoint for submitting feedback
   - Set up secure storage for feedback data
   - Implement basic analytics queries

3. **Feedback Integration** (3 days)
   - Link feedback to specific query/response pairs
   - Store relevant context (chunks, document IDs, etc.)
   - Implement mechanism to track improvement over time
   - Create admin dashboard for feedback review

4. **Long-term Improvement Loop** (2 days)
   - Design process for regular feedback review
   - Create tools to identify patterns in negative feedback
   - Implement mechanism to adjust retrieval parameters based on feedback
   - Set up reporting for feedback-driven improvements

#### Example Feedback Data Model
```typescript
interface RelevanceFeedback {
  id: string;
  timestamp: Date;
  queryId: string;
  query: string;
  response: string;
  rating: 'positive' | 'negative';
  feedbackCategory?: 'irrelevant' | 'incomplete' | 'incorrect' | 'other';
  comment?: string;
  retrievedChunks: Array<{
    chunkId: string;
    documentId: string;
    text: string;
    score: number;
  }>;
  userId?: string;
  sessionId: string;
}
```

#### Example Feedback Component
```tsx
// ResponseFeedback.tsx
const ResponseFeedback = ({ messageId, query, response, chunks }) => {
  const [rating, setRating] = useState<'positive' | 'negative' | null>(null);
  const [showDetails, setShowDetails] = useState(false);
  const [category, setCategory] = useState('');
  const [comment, setComment] = useState('');
  
  const handleSubmitFeedback = async () => {
    await ragApi.submitFeedback({
      messageId,
      query,
      response,
      rating,
      category,
      comment,
      chunks
    });
    
    toast({
      title: 'Feedback submitted',
      description: 'Thank you for helping improve the system',
      status: 'success'
    });
    
    setShowDetails(false);
  };
  
  return (
    <Box mt={2}>
      <HStack spacing={2} justifyContent="flex-end">
        <Text fontSize="xs">Was this response helpful?</Text>
        <IconButton 
          icon={<FiThumbsUp />} 
          size="xs" 
          colorScheme={rating === 'positive' ? 'green' : 'gray'}
          onClick={() => setRating('positive')}
        />
        <IconButton 
          icon={<FiThumbsDown />} 
          size="xs"
          colorScheme={rating === 'negative' ? 'red' : 'gray'}
          onClick={() => {
            setRating('negative');
            setShowDetails(true);
          }}
        />
      </HStack>
      
      <Collapse in={showDetails}>
        <Box mt={2} p={3} borderWidth={1} borderRadius="md">
          <FormControl mb={2}>
            <FormLabel fontSize="sm">What was wrong with the response?</FormLabel>
            <Select value={category} onChange={(e) => setCategory(e.target.value)}>
              <option value="irrelevant">Not relevant to my question</option>
              <option value="incomplete">Information was incomplete</option>
              <option value="incorrect">Information was incorrect</option>
              <option value="other">Other issue</option>
            </Select>
          </FormControl>
          <FormControl>
            <FormLabel fontSize="sm">Additional comments</FormLabel>
            <Textarea 
              value={comment} 
              onChange={(e) => setComment(e.target.value)}
              placeholder="Please provide any additional details..."
              size="sm"
            />
          </FormControl>
          <Button size="sm" mt={2} colorScheme="blue" onClick={handleSubmitFeedback}>
            Submit Feedback
          </Button>
        </Box>
      </Collapse>
    </Box>
  );
};
```

## 3. Result Highlighting

### Description
Visually highlight specific sections of source documents that contributed most significantly to the generated response, helping users understand how the system arrived at its answer.

### Value Proposition
- **Increased Transparency**: Users understand which parts of documents influenced the response
- **Enhanced Trust**: Clear connection between source documents and generated answers
- **Easier Verification**: Users can quickly check source material for accuracy

### Technical Approach

#### Frontend Changes
1. Enhance source document display with highlighted text sections
2. Implement collapsible context view showing text before/after the highlighted section
3. Add visual indicators of relevance score within highlighted sections

#### Backend Changes
1. Update retrieval mechanism to return exact text spans that contributed to the answer
2. Implement chunk-to-response mapping to track which chunks influenced which parts of the response
3. Add position metadata to chunk responses

### Implementation Steps

1. **Enhanced Chunk Metadata** (2 days)
   - Modify the Ragie integration to preserve exact position information
   - Adapt the chunk processing pipeline to maintain text span details
   - Update the API response format to include position data

2. **Highlighting Component** (3 days)
   - Create a `HighlightedText` component to render text with highlights
   - Implement collapsible context for highlighted passages
   - Add visual indicators for relevance scores
   - Support keyboard navigation between highlights

3. **Response-to-Source Mapping** (4 days)
   - Modify the OpenAI prompt to include citation markers
   - Implement citation extraction from the generated response
   - Create a mapping between response sections and source chunks
   - Add bidirectional navigation (click on response to see source, and vice versa)

4. **UI Integration** (2 days)
   - Update the Chat interface to include the new highlighting components
   - Add a toggle to enable/disable advanced highlighting
   - Implement smooth scrolling to highlighted sections
   - Optimize rendering for large documents

#### Example Highlighted Text Component
```tsx
// HighlightedText.tsx
interface HighlightSpan {
  startIndex: number;
  endIndex: number;
  score: number;
  citationIndex?: number;
}

interface HighlightedTextProps {
  text: string;
  highlights: HighlightSpan[];
  showFullContext?: boolean;
  onHighlightClick?: (highlight: HighlightSpan) => void;
}

const HighlightedText: React.FC<HighlightedTextProps> = ({ 
  text, 
  highlights,
  showFullContext = false,
  onHighlightClick
}) => {
  const sortedHighlights = useMemo(() => 
    [...highlights].sort((a, b) => a.startIndex - b.startIndex),
    [highlights]
  );
  
  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'green.100';
    if (score >= 0.6) return 'yellow.100';
    return 'orange.100';
  };
  
  const renderText = () => {
    if (!sortedHighlights.length) return text;
    
    let lastIndex = 0;
    const elements: JSX.Element[] = [];
    
    sortedHighlights.forEach((highlight, idx) => {
      // Add text before highlight
      if (lastIndex < highlight.startIndex) {
        const preText = text.substring(lastIndex, highlight.startIndex);
        elements.push(
          <Text as="span" key={`pre-${idx}`} 
            display={!showFullContext && preText.length > 30 ? 'none' : 'inline'}>
            {preText}
          </Text>
        );
      }
      
      // Add highlighted text
      const highlightedText = text.substring(highlight.startIndex, highlight.endIndex);
      elements.push(
        <Box
          as="span"
          key={`highlight-${idx}`}
          bg={getScoreColor(highlight.score)}
          px={1}
          borderRadius="sm"
          cursor="pointer"
          onClick={() => onHighlightClick?.(highlight)}
          data-citation={highlight.citationIndex}
        >
          {highlightedText}
          {highlight.citationIndex && 
            <sup>[{highlight.citationIndex}]</sup>
          }
        </Box>
      );
      
      lastIndex = highlight.endIndex;
    });
    
    // Add remaining text
    if (lastIndex < text.length) {
      const postText = text.substring(lastIndex);
      elements.push(
        <Text as="span" key="post" 
          display={!showFullContext && postText.length > 30 ? 'none' : 'inline'}>
          {postText}
        </Text>
      );
    }
    
    return elements;
  };
  
  return <Box fontSize="sm">{renderText()}</Box>;
};
```

## 4. Performance Optimization (Caching)

### Description
Implement a caching system for frequently asked questions and their responses to reduce latency and API costs while improving user experience.

### Value Proposition
- **Reduced Latency**: Instant responses for previously asked questions
- **Lower API Costs**: Decreased LLM API usage for repetitive queries
- **Improved Reliability**: System can provide responses even during API disruptions

### Technical Approach

#### Frontend Changes
1. Implement client-side caching for responses
2. Add UI indicators to show when a response is from cache
3. Provide option to force refresh for cached responses

#### Backend Changes
1. Develop a robust caching system with Redis or similar technology
2. Implement cache management API endpoints
3. Create cache invalidation strategies for document updates

### Implementation Steps

1. **Backend Cache Implementation** (4 days)
   - Set up Redis or another distributed caching system
   - Design cache key structure based on query + filters + document IDs
   - Implement TTL (Time-To-Live) for cached responses
   - Add cache hit/miss metrics

2. **Cache Management** (3 days)
   - Create admin API for cache inspection and management
   - Implement automatic cache invalidation when documents change
   - Design partial cache invalidation for specific document updates
   - Add cache warming for common queries

3. **Frontend Cache Integration** (2 days)
   - Add indicators for cached responses
   - Implement "refresh" option to bypass cache
   - Create session-level caching for better performance
   - Add loading state differences for cached vs. new queries

4. **Analytics & Optimization** (2 days)
   - Track cache hit rates and response times
   - Implement adaptive TTL based on query frequency
   - Create cache usage dashboard
   - Set up alerts for low cache performance

#### Example Backend Cache Implementation
```python
# cache_manager.py
import redis
import hashlib
import json
from typing import Dict, Any, Optional
from datetime import timedelta

class RagCache:
    def __init__(self, redis_url: str, default_ttl: int = 3600):
        self.redis = redis.from_url(redis_url)
        self.default_ttl = default_ttl
        
    def _generate_key(self, query: str, filters: Dict[str, Any], doc_ids: list) -> str:
        """Generate a unique cache key for the query and its parameters"""
        key_data = {
            "query": query.lower().strip(),
            "filters": filters or {},
            "doc_ids": sorted(doc_ids) if doc_ids else []
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return f"rag:query:{hashlib.md5(key_str.encode()).hexdigest()}"
    
    def get_response(self, query: str, filters: Optional[Dict[str, Any]] = None, 
                    doc_ids: Optional[list] = None) -> Optional[Dict[str, Any]]:
        """Get cached response if available"""
        key = self._generate_key(query, filters, doc_ids)
        cached_data = self.redis.get(key)
        
        if cached_data:
            self.redis.incr(f"{key}:hits")
            return json.loads(cached_data)
        return None
    
    def cache_response(self, query: str, response: Dict[str, Any], 
                       filters: Optional[Dict[str, Any]] = None, 
                       doc_ids: Optional[list] = None,
                       ttl: Optional[int] = None) -> None:
        """Store response in cache"""
        key = self._generate_key(query, filters, doc_ids)
        self.redis.set(key, json.dumps(response), ex=ttl or self.default_ttl)
        self.redis.incr(f"{key}:miss")
    
    def invalidate_for_documents(self, document_ids: list) -> int:
        """Invalidate cache entries that used specific documents"""
        pattern = "rag:query:*"
        invalidated = 0
        
        for key in self.redis.scan_iter(match=pattern):
            cached_data = self.redis.get(key)
            if not cached_data:
                continue
                
            data = json.loads(cached_data)
            # Check if any chunks came from the invalidated documents
            has_invalidated_doc = False
            for chunk in data.get("chunks", []):
                if chunk.get("document_id") in document_ids:
                    has_invalidated_doc = True
                    break
                    
            if has_invalidated_doc:
                self.redis.delete(key)
                invalidated += 1
                
        return invalidated
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        pattern = "rag:query:*"
        total_keys = 0
        hits = 0
        misses = 0
        
        for key in self.redis.scan_iter(match=pattern):
            if not key.endswith(":hits") and not key.endswith(":miss"):
                total_keys += 1
                hits += int(self.redis.get(f"{key}:hits") or 0)
                misses += int(self.redis.get(f"{key}:miss") or 0)
                
        return {
            "total_cached": total_keys,
            "hits": hits,
            "misses": misses,
            "hit_ratio": hits / (hits + misses) if (hits + misses) > 0 else 0
        }
```

#### Example API Integration
```python
# main.py
@app.post("/api/query", response_model=Dict)
async def query_documents(request: QueryRequest):
    """Query the RAG system with caching support"""
    try:
        # Check cache first
        cache_key = {
            "query": request.query,
            "filters": request.metadata_filter,
            "doc_ids": request.document_ids
        }
        
        # Skip cache if explicitly requested
        if not request.bypass_cache:
            cached_response = rag_cache.get_response(
                request.query, 
                request.metadata_filter, 
                request.document_ids
            )
            if cached_response:
                # Add cache indicator
                cached_response["from_cache"] = True
                return cached_response
        
        # Process query as normal if not in cache
        ragie_agent = RagieRAGAgent()
        result = ragie_agent.query(
            query=request.query,
            document_ids=request.document_ids,
            filter_metadata=request.metadata_filter,
            top_k=request.top_k,
            show_timings=request.show_timings
        )
        
        # Cache the result
        rag_cache.cache_response(
            request.query,
            result,
            request.metadata_filter,
            request.document_ids
        )
        
        return result
    except Exception as e:
        logger.exception(f"Error processing query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )
```

## Implementation Timeline

The entire enhancement project is estimated to take approximately 4-5 weeks, with the following timeline:

1. **Week 1**: Metadata Filtering implementation
2. **Week 2**: Relevance Feedback implementation
3. **Week 3**: Result Highlighting implementation
4. **Week 4**: Caching & Performance Optimization
5. **Week 5**: Testing, refinement, and deployment

## Resource Requirements

- 1 Frontend Developer (4-5 weeks)
- 1 Backend Developer (4-5 weeks)
- 1 UX Designer (2 weeks for component design)
- 1 DevOps Engineer (1 week for cache infrastructure)
- Testing resources (2 weeks)

## Milestones & Deliverables

1. **Milestone 1** (End of Week 1): Metadata filtering fully implemented
2. **Milestone 2** (End of Week 2): Relevance feedback system operational
3. **Milestone 3** (End of Week 3): Result highlighting features complete
4. **Milestone 4** (End of Week 4): Caching and performance optimizations implemented
5. **Final Delivery** (End of Week 5): All features tested, refined, and deployed

## Success Metrics

The success of these enhancements will be measured by:

1. **User Engagement**: Increase in system usage and query volume
2. **Response Quality**: Improvement in relevance feedback scores
3. **Performance**: Reduced average response time
4. **Cost Efficiency**: Decreased API costs per query
5. **User Satisfaction**: Improvement in overall user feedback