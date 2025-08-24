"""
Language detection for conversations
"""

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    from langdetect import detect_langs, LangDetectException
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False
    logging.warning("langdetect not installed, language detection will be limited")

from ..data.models import ConversationData

logger = logging.getLogger(__name__)


@dataclass
class DetectedLanguage:
    """Detected language information"""
    language: str
    confidence: float
    message_count: int = 0


@dataclass
class LanguageStats:
    """Language statistics for conversations"""
    primary_language: str
    language_distribution: Dict[str, int]
    mixed_language_conversations: List[int]
    detection_confidence: float


class LanguageDetector:
    """Detects languages in conversations"""
    
    def __init__(
        self,
        min_confidence: float = 0.5,
        fallback_language: str = "en"
    ):
        self.min_confidence = min_confidence
        self.fallback_language = fallback_language
        
        # Common language codes mapping
        self.language_names = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'ja': 'Japanese',
            'ko': 'Korean',
            'zh-cn': 'Chinese (Simplified)',
            'zh-tw': 'Chinese (Traditional)',
            'ar': 'Arabic',
            'hi': 'Hindi',
            'nl': 'Dutch',
            'sv': 'Swedish',
            'no': 'Norwegian',
            'da': 'Danish',
            'fi': 'Finnish',
            'pl': 'Polish',
            'tr': 'Turkish',
            'he': 'Hebrew',
            'id': 'Indonesian',
            'vi': 'Vietnamese',
            'th': 'Thai'
        }
    
    def detect_language(self, text: str) -> DetectedLanguage:
        """Detect language of text"""
        
        if not text or len(text.strip()) < 10:
            return DetectedLanguage(
                language=self.fallback_language,
                confidence=0.0
            )
        
        if HAS_LANGDETECT:
            try:
                # Use langdetect library
                detected = detect_langs(text)
                if detected and len(detected) > 0:
                    best = detected[0]
                    return DetectedLanguage(
                        language=best.lang,
                        confidence=best.prob
                    )
            except LangDetectException:
                logger.debug("Language detection failed, using fallback")
        
        # Fallback to simple heuristics
        return self._detect_with_heuristics(text)
    
    def _detect_with_heuristics(self, text: str) -> DetectedLanguage:
        """Simple heuristic-based language detection"""
        
        # Count character types
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
        has_japanese = any('\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff' for char in text)
        has_korean = any('\uac00' <= char <= '\ud7af' for char in text)
        has_arabic = any('\u0600' <= char <= '\u06ff' for char in text)
        has_cyrillic = any('\u0400' <= char <= '\u04ff' for char in text)
        has_hebrew = any('\u0590' <= char <= '\u05ff' for char in text)
        
        # Simple detection based on character sets
        if has_chinese:
            return DetectedLanguage(language='zh-cn', confidence=0.7)
        elif has_japanese:
            return DetectedLanguage(language='ja', confidence=0.7)
        elif has_korean:
            return DetectedLanguage(language='ko', confidence=0.7)
        elif has_arabic:
            return DetectedLanguage(language='ar', confidence=0.7)
        elif has_cyrillic:
            return DetectedLanguage(language='ru', confidence=0.6)
        elif has_hebrew:
            return DetectedLanguage(language='he', confidence=0.7)
        else:
            # Default to English for Latin script
            return DetectedLanguage(language='en', confidence=0.5)
    
    def detect_conversation_language(self, conversation: ConversationData) -> DetectedLanguage:
        """Detect primary language of conversation"""
        
        # Combine all message content
        full_text = " ".join([msg.content for msg in conversation.messages])
        
        if not full_text:
            return DetectedLanguage(
                language=self.fallback_language,
                confidence=0.0,
                message_count=0
            )
        
        detected = self.detect_language(full_text)
        detected.message_count = len(conversation.messages)
        
        return detected
    
    def analyze_conversations(self, conversations: List[ConversationData]) -> LanguageStats:
        """Analyze language distribution across conversations"""
        
        language_counts = Counter()
        mixed_language_convs = []
        total_confidence = 0.0
        
        for i, conversation in enumerate(conversations):
            # Detect language for each message
            message_languages = []
            for message in conversation.messages:
                if len(message.content) > 10:
                    detected = self.detect_language(message.content)
                    if detected.confidence >= self.min_confidence:
                        message_languages.append(detected.language)
            
            if message_languages:
                # Check if conversation has mixed languages
                unique_langs = set(message_languages)
                if len(unique_langs) > 1:
                    mixed_language_convs.append(i)
                
                # Count most common language in conversation
                most_common = Counter(message_languages).most_common(1)[0][0]
                language_counts[most_common] += 1
                
                # Track confidence
                conv_detected = self.detect_conversation_language(conversation)
                total_confidence += conv_detected.confidence
        
        # Determine primary language
        if language_counts:
            primary_language = language_counts.most_common(1)[0][0]
        else:
            primary_language = self.fallback_language
        
        # Calculate average confidence
        avg_confidence = total_confidence / len(conversations) if conversations else 0.0
        
        return LanguageStats(
            primary_language=primary_language,
            language_distribution=dict(language_counts),
            mixed_language_conversations=mixed_language_convs,
            detection_confidence=avg_confidence
        )
    
    def get_language_name(self, code: str) -> str:
        """Get human-readable language name from code"""
        
        return self.language_names.get(code, code.upper())
    
    def filter_by_language(
        self,
        conversations: List[ConversationData],
        target_language: str,
        min_confidence: Optional[float] = None
    ) -> List[ConversationData]:
        """Filter conversations by language"""
        
        min_conf = min_confidence or self.min_confidence
        filtered = []
        
        for conversation in conversations:
            detected = self.detect_conversation_language(conversation)
            if detected.language == target_language and detected.confidence >= min_conf:
                filtered.append(conversation)
        
        logger.info(
            f"Filtered {len(filtered)}/{len(conversations)} conversations "
            f"for language '{target_language}'"
        )
        
        return filtered