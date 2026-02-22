"""
Smart preprocessing system that orchestrates validation, normalization, and batching
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

from ..data.models import ConversationData
from .validators import CompositeValidator, ValidationResult
from .normalizers import CompositeNormalizer
from .language_detector import LanguageDetector, LanguageStats

logger = logging.getLogger(__name__)


class BatchingMethod(Enum):
    """Methods for batching conversations"""
    FIXED_SIZE = "fixed_size"
    TOKEN_BUDGET = "token_budget"
    SIMILARITY = "similarity"
    LANGUAGE = "language"
    MIXED = "mixed"


@dataclass
class TokenStats:
    """Token statistics for conversations"""
    total_tokens: int = 0
    avg_tokens_per_conversation: float = 0.0
    max_tokens: int = 0
    min_tokens: int = 0
    conversations_over_limit: List[int] = field(default_factory=list)
    token_distribution: Dict[str, int] = field(default_factory=dict)


@dataclass
class BatchingStrategy:
    """Strategy for batching conversations"""
    batch_size: int
    batching_method: BatchingMethod
    estimated_batches: int
    token_budget_per_batch: int = 4000


@dataclass
class DataQualityReport:
    """Comprehensive data quality report"""
    total_conversations: int
    valid_conversations: int
    validation_results: ValidationResult
    language_stats: LanguageStats
    token_stats: TokenStats
    recommendations: List[str]
    auto_fixable_issues: int
    estimated_processing_time: float
    overall_quality_score: float


@dataclass
class PreprocessingOptions:
    """Options for preprocessing"""
    remove_duplicates: bool = True
    remove_pii: bool = False
    fix_encoding: bool = True
    normalize_unicode: bool = True
    truncate_long: bool = True
    remove_empty: bool = True
    min_quality_score: float = 0.3
    target_language: Optional[str] = None
    max_tokens_per_conversation: int = 4000


class SmartPreprocessor:
    """Smart preprocessing orchestrator"""
    
    def __init__(
        self,
        options: Optional[PreprocessingOptions] = None,
        tokenizer_model: str = "cl100k_base"
    ):
        self.options = options or PreprocessingOptions()
        
        # Initialize components
        self.validator = CompositeValidator()
        self.normalizer = CompositeNormalizer()
        self.language_detector = LanguageDetector()
        
        # Initialize tokenizer
        self.tokenizer = None
        if HAS_TIKTOKEN:
            try:
                self.tokenizer = tiktoken.get_encoding(tokenizer_model)
            except Exception:
                logger.warning(f"Failed to load tokenizer {tokenizer_model}, using default")
                try:
                    self.tokenizer = tiktoken.get_encoding("cl100k_base")
                except Exception:
                    logger.warning("tiktoken encoding not available, using simple token counting")
        else:
            logger.info("tiktoken not installed, using simple token counting")
    
    def analyze_data_quality(self, conversations: List[ConversationData]) -> DataQualityReport:
        """Analyze data quality and generate report"""
        
        start_time = time.time()
        
        # Validate conversations
        validation_results = self.validator.validate(conversations)
        
        # Analyze languages
        language_stats = self.language_detector.analyze_conversations(conversations)
        
        # Calculate token statistics
        token_stats = self._calculate_token_stats(conversations)
        
        # Count valid conversations
        valid_conversations = sum(
            1 for i in range(len(conversations))
            if not any(
                issue.conversation_index == i and issue.severity.value == "error"
                for issue in validation_results.issues
            )
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            validation_results,
            language_stats,
            token_stats
        )
        
        # Count auto-fixable issues
        auto_fixable = sum(
            1 for issue in validation_results.issues
            if issue.issue_type.value in ["encoding_issue", "duplicate", "too_long"]
        )
        
        # Calculate overall quality score
        quality_score = (
            validation_results.quality_score * 0.5 +
            (valid_conversations / len(conversations) if conversations else 0) * 0.3 +
            language_stats.detection_confidence * 0.2
        )
        
        processing_time = time.time() - start_time
        
        return DataQualityReport(
            total_conversations=len(conversations),
            valid_conversations=valid_conversations,
            validation_results=validation_results,
            language_stats=language_stats,
            token_stats=token_stats,
            recommendations=recommendations,
            auto_fixable_issues=auto_fixable,
            estimated_processing_time=processing_time,
            overall_quality_score=quality_score
        )
    
    def preprocess(
        self,
        conversations: List[ConversationData],
        options: Optional[PreprocessingOptions] = None
    ) -> Tuple[List[ConversationData], DataQualityReport]:
        """Preprocess conversations"""
        
        options = options or self.options
        
        # Analyze quality first
        quality_report = self.analyze_data_quality(conversations)
        
        # Start preprocessing
        processed = conversations.copy()
        
        # Remove empty conversations
        if options.remove_empty:
            processed = [c for c in processed if c.messages]
            logger.info(f"Removed {len(conversations) - len(processed)} empty conversations")
        
        # Normalize
        if options.normalize_unicode or options.fix_encoding:
            processed = [self.normalizer.normalize(c) for c in processed]
            logger.info("Applied normalization")
        
        # Remove duplicates
        if options.remove_duplicates:
            processed = self._remove_duplicates(processed)
            logger.info(f"Removed duplicates, {len(processed)} conversations remaining")
        
        # Filter by language
        if options.target_language:
            processed = self.language_detector.filter_by_language(
                processed,
                options.target_language
            )
            logger.info(f"Filtered to {len(processed)} {options.target_language} conversations")
        
        # Truncate long conversations
        if options.truncate_long:
            processed = self._truncate_long_conversations(
                processed,
                options.max_tokens_per_conversation
            )
            logger.info("Truncated long conversations")
        
        # Filter by quality score
        if options.min_quality_score > 0:
            processed = self._filter_by_quality(processed, options.min_quality_score)
            logger.info(f"Filtered by quality, {len(processed)} conversations remaining")
        
        return processed, quality_report
    
    def create_batching_strategy(
        self,
        conversations: List[ConversationData],
        method: BatchingMethod = BatchingMethod.TOKEN_BUDGET,
        target_batch_size: int = 10,
        token_budget: int = 4000
    ) -> BatchingStrategy:
        """Create optimal batching strategy"""
        
        if method == BatchingMethod.FIXED_SIZE:
            estimated_batches = (len(conversations) + target_batch_size - 1) // target_batch_size
            
            return BatchingStrategy(
                batch_size=target_batch_size,
                batching_method=method,
                estimated_batches=estimated_batches,
                token_budget_per_batch=token_budget
            )
        
        elif method == BatchingMethod.TOKEN_BUDGET:
            # Calculate tokens per conversation
            token_counts = [self._count_tokens(c) for c in conversations]
            
            # Estimate batches based on token budget
            current_batch_tokens = 0
            batch_count = 1
            
            for tokens in token_counts:
                if current_batch_tokens + tokens > token_budget:
                    batch_count += 1
                    current_batch_tokens = tokens
                else:
                    current_batch_tokens += tokens
            
            avg_batch_size = len(conversations) // batch_count if batch_count > 0 else len(conversations)
            
            return BatchingStrategy(
                batch_size=avg_batch_size,
                batching_method=method,
                estimated_batches=batch_count,
                token_budget_per_batch=token_budget
            )
        
        elif method == BatchingMethod.LANGUAGE:
            # Group by language
            language_groups = {}
            for conv in conversations:
                lang = self.language_detector.detect_conversation_language(conv).language
                if lang not in language_groups:
                    language_groups[lang] = []
                language_groups[lang].append(conv)
            
            estimated_batches = len(language_groups)
            avg_batch_size = len(conversations) // estimated_batches if estimated_batches > 0 else len(conversations)
            
            return BatchingStrategy(
                batch_size=avg_batch_size,
                batching_method=method,
                estimated_batches=estimated_batches,
                token_budget_per_batch=token_budget
            )
        
        else:
            # Default to fixed size
            return self.create_batching_strategy(
                conversations,
                BatchingMethod.FIXED_SIZE,
                target_batch_size,
                token_budget
            )
    
    def create_batches(
        self,
        conversations: List[ConversationData],
        strategy: BatchingStrategy
    ) -> List[List[ConversationData]]:
        """Create batches based on strategy"""
        
        batches = []
        
        if strategy.batching_method == BatchingMethod.FIXED_SIZE:
            # Simple fixed-size batching
            for i in range(0, len(conversations), strategy.batch_size):
                batch = conversations[i:i + strategy.batch_size]
                batches.append(batch)
        
        elif strategy.batching_method == BatchingMethod.TOKEN_BUDGET:
            # Token-based batching
            current_batch = []
            current_tokens = 0
            
            for conv in conversations:
                conv_tokens = self._count_tokens(conv)
                
                if current_tokens + conv_tokens > strategy.token_budget_per_batch and current_batch:
                    batches.append(current_batch)
                    current_batch = [conv]
                    current_tokens = conv_tokens
                else:
                    current_batch.append(conv)
                    current_tokens += conv_tokens
            
            if current_batch:
                batches.append(current_batch)
        
        elif strategy.batching_method == BatchingMethod.LANGUAGE:
            # Language-based batching
            language_groups = {}
            for conv in conversations:
                lang = self.language_detector.detect_conversation_language(conv).language
                if lang not in language_groups:
                    language_groups[lang] = []
                language_groups[lang].append(conv)
            
            for lang, group in language_groups.items():
                # Further split by size if needed
                for i in range(0, len(group), strategy.batch_size):
                    batch = group[i:i + strategy.batch_size]
                    batches.append(batch)
        
        logger.info(f"Created {len(batches)} batches using {strategy.batching_method.value} method")
        
        return batches
    
    def _calculate_token_stats(self, conversations: List[ConversationData]) -> TokenStats:
        """Calculate token statistics"""
        
        if not conversations:
            return TokenStats()
        
        token_counts = []
        conversations_over_limit = []
        
        for i, conv in enumerate(conversations):
            tokens = self._count_tokens(conv)
            token_counts.append(tokens)
            
            if tokens > self.options.max_tokens_per_conversation:
                conversations_over_limit.append(i)
        
        # Calculate distribution
        distribution = {
            "0-500": sum(1 for t in token_counts if t <= 500),
            "501-1000": sum(1 for t in token_counts if 500 < t <= 1000),
            "1001-2000": sum(1 for t in token_counts if 1000 < t <= 2000),
            "2001-4000": sum(1 for t in token_counts if 2000 < t <= 4000),
            "4000+": sum(1 for t in token_counts if t > 4000)
        }
        
        return TokenStats(
            total_tokens=sum(token_counts),
            avg_tokens_per_conversation=sum(token_counts) / len(token_counts),
            max_tokens=max(token_counts),
            min_tokens=min(token_counts),
            conversations_over_limit=conversations_over_limit,
            token_distribution=distribution
        )
    
    def _count_tokens(self, conversation: ConversationData) -> int:
        """Count tokens in conversation"""
        text = " ".join([msg.content for msg in conversation.messages])
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text))
        # Fallback: approximate token count by splitting on whitespace
        return len(text.split())
    
    def _remove_duplicates(self, conversations: List[ConversationData]) -> List[ConversationData]:
        """Remove duplicate conversations"""
        
        seen_hashes = set()
        unique = []
        
        for conv in conversations:
            conv_hash = self._get_conversation_hash(conv)
            if conv_hash not in seen_hashes:
                seen_hashes.add(conv_hash)
                unique.append(conv)
        
        return unique
    
    def _get_conversation_hash(self, conversation: ConversationData) -> str:
        """Generate hash for conversation"""
        
        content = "".join([msg.content for msg in conversation.messages])
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _truncate_long_conversations(
        self,
        conversations: List[ConversationData],
        max_tokens: int
    ) -> List[ConversationData]:
        """Truncate conversations that exceed token limit"""
        
        truncated = []
        
        for conv in conversations:
            tokens = self._count_tokens(conv)
            
            if tokens <= max_tokens:
                truncated.append(conv)
            else:
                # Truncate from the middle to preserve context
                messages = conv.messages.copy()
                
                # Keep first and last messages
                if len(messages) > 2:
                    # Calculate how many messages to keep
                    keep_ratio = max_tokens / tokens
                    keep_count = max(2, int(len(messages) * keep_ratio))
                    
                    # Keep first half and last half
                    keep_start = keep_count // 2
                    keep_end = keep_count - keep_start
                    
                    truncated_messages = messages[:keep_start] + messages[-keep_end:]
                else:
                    truncated_messages = messages
                
                truncated.append(ConversationData(
                    messages=truncated_messages,
                    metadata=conv.metadata
                ))
        
        return truncated
    
    def _filter_by_quality(
        self,
        conversations: List[ConversationData],
        min_score: float
    ) -> List[ConversationData]:
        """Filter conversations by quality score"""
        
        filtered = []
        
        for conv in conversations:
            # Simple quality heuristics
            score = 1.0
            
            # Penalize very short conversations
            if len(conv.messages) < 2:
                score *= 0.5
            
            # Penalize conversations with very short messages
            avg_msg_length = sum(len(msg.content) for msg in conv.messages) / len(conv.messages)
            if avg_msg_length < 50:
                score *= 0.7
            
            # Penalize unbalanced conversations
            user_msgs = sum(1 for msg in conv.messages if msg.role == "user")
            assistant_msgs = sum(1 for msg in conv.messages if msg.role == "assistant")
            if user_msgs == 0 or assistant_msgs == 0:
                score *= 0.5
            
            if score >= min_score:
                filtered.append(conv)
        
        return filtered
    
    def _generate_recommendations(
        self,
        validation_results: ValidationResult,
        language_stats: LanguageStats,
        token_stats: TokenStats
    ) -> List[str]:
        """Generate preprocessing recommendations"""
        
        recommendations = []
        
        # Add validation-based recommendations
        recommendations.extend(validation_results.suggestions)
        
        # Language recommendations
        if len(language_stats.mixed_language_conversations) > len(language_stats.language_distribution) * 0.1:
            recommendations.append("Consider filtering conversations to a single language")
        
        # Token recommendations
        if token_stats.conversations_over_limit:
            recommendations.append(
                f"Consider truncating {len(token_stats.conversations_over_limit)} "
                f"conversations that exceed token limit"
            )
        
        if token_stats.avg_tokens_per_conversation > 3000:
            recommendations.append("Consider using smaller batch sizes due to high token counts")
        
        return recommendations