"""
Data normalization utilities for preprocessing
"""

import logging
import re
import unicodedata
from typing import List, Optional

from ..data.models import ConversationData, Message

logger = logging.getLogger(__name__)


class Normalizer:
    """Base normalizer class"""
    
    def normalize(self, conversation: ConversationData) -> ConversationData:
        """Normalize a conversation"""
        raise NotImplementedError
    
    @property
    def name(self) -> str:
        """Get normalizer name"""
        return self.__class__.__name__


class UnicodeNormalizer(Normalizer):
    """Normalizes Unicode characters"""
    
    def __init__(self, form: str = 'NFC'):
        """
        Initialize Unicode normalizer
        
        Args:
            form: Unicode normalization form ('NFC', 'NFD', 'NFKC', 'NFKD')
        """
        self.form = form
    
    def normalize(self, conversation: ConversationData) -> ConversationData:
        """Normalize Unicode in conversation"""
        
        normalized_messages = []
        for message in conversation.messages:
            normalized_content = unicodedata.normalize(self.form, message.content)
            normalized_messages.append(Message(
                role=message.role,
                content=normalized_content
            ))
        
        return ConversationData(
            messages=normalized_messages,
            metadata=conversation.metadata
        )


class WhitespaceNormalizer(Normalizer):
    """Normalizes whitespace in text"""
    
    def __init__(
        self,
        remove_extra_spaces: bool = True,
        trim_lines: bool = True,
        remove_empty_lines: bool = True,
        normalize_newlines: bool = True
    ):
        self.remove_extra_spaces = remove_extra_spaces
        self.trim_lines = trim_lines
        self.remove_empty_lines = remove_empty_lines
        self.normalize_newlines = normalize_newlines
    
    def normalize(self, conversation: ConversationData) -> ConversationData:
        """Normalize whitespace in conversation"""
        
        normalized_messages = []
        
        for message in conversation.messages:
            content = message.content
            
            # Normalize newlines
            if self.normalize_newlines:
                content = content.replace('\r\n', '\n').replace('\r', '\n')
            
            # Process lines
            if self.trim_lines or self.remove_empty_lines:
                lines = content.split('\n')
                
                if self.trim_lines:
                    lines = [line.strip() for line in lines]
                
                if self.remove_empty_lines:
                    lines = [line for line in lines if line]
                
                content = '\n'.join(lines)
            
            # Remove extra spaces
            if self.remove_extra_spaces:
                content = re.sub(r'\s+', ' ', content)
                content = re.sub(r'\n\s+', '\n', content)
                content = re.sub(r'\s+\n', '\n', content)
            
            normalized_messages.append(Message(
                role=message.role,
                content=content.strip()
            ))
        
        return ConversationData(
            messages=normalized_messages,
            metadata=conversation.metadata
        )


class EncodingFixer(Normalizer):
    """Fixes common encoding issues"""
    
    def __init__(self):
        # Common encoding fixes
        self.replacements = {
            'â€™': "'",
            'â€œ': '"',
            'â€': '"',
            'â€"': '—',
            'â€"': '–',
            'â€¦': '...',
            'Ã©': 'é',
            'Ã¨': 'è',
            'Ã ': 'à',
            'Ã¢': 'â',
            'Ã§': 'ç',
            'Ã´': 'ô',
            'Ã®': 'î',
            'Ã¯': 'ï',
            'Ã«': 'ë',
            'Ã¼': 'ü',
            'Ã¶': 'ö',
            'Ã¤': 'ä',
            'â€‹': '',  # Zero-width space
            '\u200b': '',  # Zero-width space
            '\ufeff': '',  # BOM
        }
    
    def normalize(self, conversation: ConversationData) -> ConversationData:
        """Fix encoding issues in conversation"""
        
        normalized_messages = []
        
        for message in conversation.messages:
            content = message.content
            
            # Apply replacements
            for old, new in self.replacements.items():
                content = content.replace(old, new)
            
            # Try to fix mojibake
            try:
                # Try to decode as latin-1 and re-encode as utf-8
                content_bytes = content.encode('latin-1', errors='ignore')
                content = content_bytes.decode('utf-8', errors='ignore')
            except:
                pass  # Keep original if conversion fails
            
            # Remove non-printable characters
            content = ''.join(char for char in content if char.isprintable() or char.isspace())
            
            normalized_messages.append(Message(
                role=message.role,
                content=content
            ))
        
        return ConversationData(
            messages=normalized_messages,
            metadata=conversation.metadata
        )


class TextCleaner(Normalizer):
    """Cleans text content"""
    
    def __init__(
        self,
        remove_urls: bool = False,
        remove_emails: bool = False,
        remove_html: bool = True,
        remove_special_chars: bool = False,
        lowercase: bool = False
    ):
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_html = remove_html
        self.remove_special_chars = remove_special_chars
        self.lowercase = lowercase
    
    def normalize(self, conversation: ConversationData) -> ConversationData:
        """Clean text in conversation"""
        
        normalized_messages = []
        
        for message in conversation.messages:
            content = message.content
            
            # Remove URLs
            if self.remove_urls:
                content = re.sub(r'https?://\S+|www\.\S+', '[URL]', content)
            
            # Remove emails
            if self.remove_emails:
                content = re.sub(r'\S+@\S+', '[EMAIL]', content)
            
            # Remove HTML tags
            if self.remove_html:
                content = re.sub(r'<[^>]+>', '', content)
                # Unescape HTML entities
                content = content.replace('&lt;', '<')
                content = content.replace('&gt;', '>')
                content = content.replace('&amp;', '&')
                content = content.replace('&quot;', '"')
                content = content.replace('&#39;', "'")
            
            # Remove special characters
            if self.remove_special_chars:
                content = re.sub(r'[^\w\s-]', '', content)
            
            # Lowercase
            if self.lowercase:
                content = content.lower()
            
            normalized_messages.append(Message(
                role=message.role,
                content=content
            ))
        
        return ConversationData(
            messages=normalized_messages,
            metadata=conversation.metadata
        )


class CompositeNormalizer(Normalizer):
    """Combines multiple normalizers"""
    
    def __init__(self, normalizers: Optional[List[Normalizer]] = None):
        self.normalizers = normalizers or [
            UnicodeNormalizer(),
            WhitespaceNormalizer(),
            EncodingFixer()
        ]
    
    def normalize(self, conversation: ConversationData) -> ConversationData:
        """Apply all normalizers in sequence"""
        
        result = conversation
        for normalizer in self.normalizers:
            result = normalizer.normalize(result)
            logger.debug(f"Applied {normalizer.name}")
        
        return result