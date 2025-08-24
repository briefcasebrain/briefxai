"""
Persistence Layer

Provides comprehensive data persistence with:
- Enhanced persistence manager with SQLite/PostgreSQL support
- Session management and batch tracking
- Result caching and compression
- Migration system
- Template and project management
"""

from .enhanced import (
    # Core persistence manager
    EnhancedPersistenceManager,
    get_persistence_manager,
    initialize_persistence,
    
    # Data models
    AnalysisSession,
    BatchProgress,
    AnalysisTemplate,
    Project,
    ProviderConfig,
    CacheEntry,
    
    # Enums
    SessionStatus,
    BatchStatus,
    TemplateCategory,
    ProviderType,
    CacheType,
    ResultType,
    
    # Migration system
    Migration,
    MigrationManager,
)

__all__ = [
    # Core persistence
    'EnhancedPersistenceManager',
    'get_persistence_manager',
    'initialize_persistence',
    
    # Data models
    'AnalysisSession',
    'BatchProgress',
    'AnalysisTemplate',
    'Project',
    'ProviderConfig',
    'CacheEntry',
    
    # Enums
    'SessionStatus',
    'BatchStatus',
    'TemplateCategory',
    'ProviderType',
    'CacheType',
    'ResultType',
    
    # Migration system
    'Migration',
    'MigrationManager',
]