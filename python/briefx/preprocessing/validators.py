"""
Data validation system for preprocessing conversations
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set, Dict, Any

from ..data.models import ConversationData

logger = logging.getLogger(__name__)


class IssueSeverity(Enum):
    """Severity levels for validation issues"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class IssueType(Enum):
    """Types of validation issues"""
    EMPTY = "empty"
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    DUPLICATE = "duplicate"
    MISSING_ROLE = "missing_role"
    INVALID_FORMAT = "invalid_format"
    PII = "pii"
    PROFANITY = "profanity"
    LOW_QUALITY = "low_quality"
    TRUNCATED = "truncated"
    MIXED_LANGUAGES = "mixed_languages"
    ENCODING_ISSUE = "encoding_issue"


@dataclass
class ValidationIssue:
    """Represents a validation issue"""
    severity: IssueSeverity
    issue_type: IssueType
    message: str
    conversation_index: Optional[int] = None
    message_index: Optional[int] = None


@dataclass
class ValidationResult:
    """Result of validation process"""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    quality_score: float = 1.0


class Validator:
    """Base validator class"""
    
    def validate(self, conversations: List[ConversationData]) -> ValidationResult:
        """Validate conversations"""
        raise NotImplementedError
    
    @property
    def name(self) -> str:
        """Get validator name"""
        return self.__class__.__name__


class FormatValidator(Validator):
    """Validates conversation format and structure"""
    
    def __init__(
        self,
        min_messages: int = 2,
        max_messages: int = 1000,
        required_roles: Optional[Set[str]] = None
    ):
        self.min_messages = min_messages
        self.max_messages = max_messages
        self.required_roles = required_roles or {"user", "assistant"}
    
    def validate(self, conversations: List[ConversationData]) -> ValidationResult:
        """Validate conversation format"""
        
        issues = []
        valid_count = 0
        
        for i, conversation in enumerate(conversations):
            # Check if empty
            if not conversation.messages:
                issues.append(ValidationIssue(
                    severity=IssueSeverity.ERROR,
                    issue_type=IssueType.EMPTY,
                    message="Conversation is empty",
                    conversation_index=i
                ))
                continue
            
            # Check message count
            msg_count = len(conversation.messages)
            if msg_count < self.min_messages:
                issues.append(ValidationIssue(
                    severity=IssueSeverity.WARNING,
                    issue_type=IssueType.TOO_SHORT,
                    message=f"Conversation has only {msg_count} messages",
                    conversation_index=i
                ))
            elif msg_count > self.max_messages:
                issues.append(ValidationIssue(
                    severity=IssueSeverity.WARNING,
                    issue_type=IssueType.TOO_LONG,
                    message=f"Conversation has {msg_count} messages, may be truncated",
                    conversation_index=i
                ))
            
            # Check roles
            roles = {msg.role for msg in conversation.messages}
            for required_role in self.required_roles:
                if required_role not in roles:
                    issues.append(ValidationIssue(
                        severity=IssueSeverity.WARNING,
                        issue_type=IssueType.MISSING_ROLE,
                        message=f"Missing '{required_role}' role in conversation",
                        conversation_index=i
                    ))
            
            # Check for alternating roles
            prev_role = ""
            for j, message in enumerate(conversation.messages):
                if message.role == prev_role and message.role != "system":
                    issues.append(ValidationIssue(
                        severity=IssueSeverity.INFO,
                        issue_type=IssueType.INVALID_FORMAT,
                        message=f"Consecutive messages from same role: {message.role}",
                        conversation_index=i,
                        message_index=j
                    ))
                prev_role = message.role
            
            if not any(issue.severity == IssueSeverity.ERROR for issue in issues if issue.conversation_index == i):
                valid_count += 1
        
        # Calculate quality score
        quality_score = valid_count / len(conversations) if conversations else 0.0
        
        # Generate suggestions
        suggestions = []
        if any(issue.issue_type == IssueType.TOO_SHORT for issue in issues):
            suggestions.append("Consider filtering out very short conversations")
        if any(issue.issue_type == IssueType.TOO_LONG for issue in issues):
            suggestions.append("Consider truncating very long conversations")
        if any(issue.issue_type == IssueType.MISSING_ROLE for issue in issues):
            suggestions.append("Ensure all conversations have both user and assistant messages")
        
        return ValidationResult(
            is_valid=len([i for i in issues if i.severity == IssueSeverity.ERROR]) == 0,
            issues=issues,
            suggestions=suggestions,
            quality_score=quality_score
        )


class ContentValidator(Validator):
    """Validates conversation content quality"""
    
    def __init__(
        self,
        min_content_length: int = 10,
        max_content_length: int = 50000,
        check_pii: bool = True,
        check_profanity: bool = False
    ):
        self.min_content_length = min_content_length
        self.max_content_length = max_content_length
        self.check_pii = check_pii
        self.check_profanity = check_profanity
        
        # PII patterns
        self.pii_patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
            'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            'credit_card': re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'),
            'ip_address': re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')
        }
    
    def validate(self, conversations: List[ConversationData]) -> ValidationResult:
        """Validate conversation content"""
        
        issues = []
        
        for i, conversation in enumerate(conversations):
            for j, message in enumerate(conversation.messages):
                content = message.content
                
                # Check content length
                if len(content) < self.min_content_length:
                    issues.append(ValidationIssue(
                        severity=IssueSeverity.WARNING,
                        issue_type=IssueType.LOW_QUALITY,
                        message=f"Message content too short ({len(content)} chars)",
                        conversation_index=i,
                        message_index=j
                    ))
                elif len(content) > self.max_content_length:
                    issues.append(ValidationIssue(
                        severity=IssueSeverity.WARNING,
                        issue_type=IssueType.TRUNCATED,
                        message=f"Message content too long ({len(content)} chars)",
                        conversation_index=i,
                        message_index=j
                    ))
                
                # Check for PII
                if self.check_pii:
                    for pii_type, pattern in self.pii_patterns.items():
                        if pattern.search(content):
                            issues.append(ValidationIssue(
                                severity=IssueSeverity.WARNING,
                                issue_type=IssueType.PII,
                                message=f"Potential {pii_type} detected",
                                conversation_index=i,
                                message_index=j
                            ))
                
                # Check encoding issues
                try:
                    content.encode('utf-8').decode('utf-8')
                except UnicodeError:
                    issues.append(ValidationIssue(
                        severity=IssueSeverity.ERROR,
                        issue_type=IssueType.ENCODING_ISSUE,
                        message="Encoding issue detected",
                        conversation_index=i,
                        message_index=j
                    ))
        
        quality_score = 1.0 - (len(issues) / (len(conversations) * 3)) if conversations else 0.0
        quality_score = max(0.0, min(1.0, quality_score))
        
        suggestions = []
        if any(issue.issue_type == IssueType.PII for issue in issues):
            suggestions.append("Consider removing or masking PII before processing")
        if any(issue.issue_type == IssueType.ENCODING_ISSUE for issue in issues):
            suggestions.append("Fix encoding issues before processing")
        
        return ValidationResult(
            is_valid=len([i for i in issues if i.severity == IssueSeverity.ERROR]) == 0,
            issues=issues,
            suggestions=suggestions,
            quality_score=quality_score
        )


class DuplicateDetector(Validator):
    """Detects duplicate conversations"""
    
    def __init__(self, similarity_threshold: float = 0.95):
        self.similarity_threshold = similarity_threshold
    
    def _get_conversation_hash(self, conversation: ConversationData) -> str:
        """Generate hash for conversation"""
        
        content = "".join([msg.content for msg in conversation.messages])
        return hashlib.sha256(content.encode()).hexdigest()
    
    def validate(self, conversations: List[ConversationData]) -> ValidationResult:
        """Detect duplicate conversations"""
        
        issues = []
        seen_hashes = {}
        duplicates = set()
        
        for i, conversation in enumerate(conversations):
            conv_hash = self._get_conversation_hash(conversation)
            
            if conv_hash in seen_hashes:
                issues.append(ValidationIssue(
                    severity=IssueSeverity.WARNING,
                    issue_type=IssueType.DUPLICATE,
                    message=f"Duplicate of conversation {seen_hashes[conv_hash]}",
                    conversation_index=i
                ))
                duplicates.add(i)
            else:
                seen_hashes[conv_hash] = i
        
        quality_score = 1.0 - (len(duplicates) / len(conversations)) if conversations else 1.0
        
        suggestions = []
        if duplicates:
            suggestions.append(f"Remove {len(duplicates)} duplicate conversations")
        
        return ValidationResult(
            is_valid=True,  # Duplicates are not errors
            issues=issues,
            suggestions=suggestions,
            quality_score=quality_score
        )


class CompositeValidator(Validator):
    """Combines multiple validators"""
    
    def __init__(self, validators: Optional[List[Validator]] = None):
        self.validators = validators or [
            FormatValidator(),
            ContentValidator(),
            DuplicateDetector()
        ]
    
    def validate(self, conversations: List[ConversationData]) -> ValidationResult:
        """Run all validators"""
        
        all_issues = []
        all_suggestions = []
        quality_scores = []
        is_valid = True
        
        for validator in self.validators:
            result = validator.validate(conversations)
            all_issues.extend(result.issues)
            all_suggestions.extend(result.suggestions)
            quality_scores.append(result.quality_score)
            is_valid = is_valid and result.is_valid
            
            logger.info(f"{validator.name}: {len(result.issues)} issues found")
        
        # Remove duplicate suggestions
        unique_suggestions = list(set(all_suggestions))
        
        # Calculate overall quality score
        overall_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        return ValidationResult(
            is_valid=is_valid,
            issues=all_issues,
            suggestions=unique_suggestions,
            quality_score=overall_quality
        )