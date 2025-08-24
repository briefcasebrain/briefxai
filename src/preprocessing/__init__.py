"""
Data preprocessing and validation module for BriefX
"""

from .validators import (
    Validator,
    FormatValidator,
    ContentValidator,
    DuplicateDetector,
    CompositeValidator,
    ValidationResult,
    ValidationIssue,
    IssueSeverity,
    IssueType
)

from .normalizers import (
    Normalizer,
    UnicodeNormalizer,
    WhitespaceNormalizer,
    EncodingFixer,
    CompositeNormalizer
)

from .smart_preprocessor import (
    SmartPreprocessor,
    DataQualityReport,
    BatchingStrategy,
    BatchingMethod,
    PreprocessingOptions,
    TokenStats
)

from .language_detector import (
    LanguageDetector,
    DetectedLanguage,
    LanguageStats
)

__all__ = [
    # Validators
    'Validator',
    'FormatValidator',
    'ContentValidator',
    'DuplicateDetector',
    'CompositeValidator',
    'ValidationResult',
    'ValidationIssue',
    'IssueSeverity',
    'IssueType',
    
    # Normalizers
    'Normalizer',
    'UnicodeNormalizer',
    'WhitespaceNormalizer',
    'EncodingFixer',
    'CompositeNormalizer',
    
    # Smart preprocessor
    'SmartPreprocessor',
    'DataQualityReport',
    'BatchingStrategy',
    'BatchingMethod',
    'PreprocessingOptions',
    'TokenStats',
    
    # Language detection
    'LanguageDetector',
    'DetectedLanguage',
    'LanguageStats'
]