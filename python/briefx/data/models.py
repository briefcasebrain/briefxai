"""Data models for BriefX - Python equivalents of Rust structs"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

@dataclass
class Message:
    """Equivalent to Rust Message struct"""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class ConversationData:
    """Equivalent to Rust ConversationData struct"""
    messages: List[Message]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.messages)
    
    @property
    def content(self) -> str:
        """Alias for get_text(), used by clio.py."""
        return self.get_text()

    def get_text(self) -> str:
        """Get combined text from all messages"""
        return " ".join(msg.content for msg in self.messages)
    
    def get_user_messages(self) -> List[Message]:
        """Get only user messages"""
        return [msg for msg in self.messages if msg.role == "user"]
    
    def get_assistant_messages(self) -> List[Message]:
        """Get only assistant messages"""
        return [msg for msg in self.messages if msg.role == "assistant"]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ConversationData to dict for JSON serialization"""
        return {
            "messages": [{"role": msg.role, "content": msg.content} for msg in self.messages],
            "metadata": self.metadata
        }

@dataclass
class ConversationCluster:
    """Equivalent to Rust ConversationCluster struct"""
    id: int
    name: str
    description: str
    count: int
    conversations: List[ConversationData] = field(default_factory=list)
    representative_messages: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'count': self.count,
            'keywords': self.keywords,
            'representative_messages': self.representative_messages
        }

@dataclass
class FacetValue:
    """Equivalent to Rust FacetValue struct"""
    name: str
    value: str
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'value': self.value,
            'confidence': self.confidence
        }

@dataclass
class ConversationAnalysis:
    """Analysis results for a conversation"""
    conversation: ConversationData
    cluster_id: Optional[int] = None
    cluster_name: Optional[str] = None
    sentiment: float = 0.0  # -1.0 to 1.0
    category: Optional[str] = None
    facets: List[FacetValue] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    visualization_embedding: Optional[List[float]] = None
    topics: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': getattr(self.conversation, 'id', None),
            'text': self.conversation.get_text()[:200],  # First 200 chars
            'category': self.category,
            'cluster_name': self.cluster_name,
            'sentiment': self.sentiment,
            'topics': self.topics,
            'facets': [f.to_dict() for f in self.facets],
            'messages': len(self.conversation.messages),
            'timestamp': datetime.now().isoformat()
        }

@dataclass
class AnalysisResults:
    """Complete analysis results"""
    conversations: List[ConversationAnalysis]
    clusters: List[ConversationCluster]
    total_conversations: int
    total_messages: int
    processing_time: float
    session_id: str
    
    @property
    def avg_messages_per_conversation(self) -> float:
        if self.total_conversations == 0:
            return 0.0
        return self.total_messages / self.total_conversations
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'conversations': [conv.to_dict() for conv in self.conversations],
            'clusters': [cluster.to_dict() for cluster in self.clusters],
            'statistics': {
                'total_conversations': self.total_conversations,
                'total_messages': self.total_messages,
                'avg_messages': self.avg_messages_per_conversation,
                'processing_time': self.processing_time
            },
            'session_id': self.session_id
        }