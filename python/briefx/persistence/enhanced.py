"""
Enhanced Persistence Layer for conversation data management

Provides comprehensive data persistence with:
- SQLite for local storage
- PostgreSQL for production
- Migration system
- Session management
- Result caching
- Structured storage
- Batch progress tracking
"""

import asyncio
import hashlib
import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import uuid

try:
    import asyncpg
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False
    logging.warning("asyncpg not installed, PostgreSQL support disabled")

try:
    import aiosqlite
    HAS_AIOSQLITE = True
except ImportError:
    HAS_AIOSQLITE = False
    logging.warning("aiosqlite not installed, SQLite support disabled")

logger = logging.getLogger(__name__)

# ============================================================================
# Domain Models
# ============================================================================

class SessionStatus(Enum):
    PENDING = "pending"
    RUNNING = "running" 
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"

class BatchStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class TemplateCategory(Enum):
    GENERAL = "general"
    SUPPORT = "support"
    SALES = "sales"
    MEDICAL = "medical"
    EDUCATION = "education"
    CUSTOM = "custom"

class ProviderType(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    GEMINI = "gemini"
    HUGGINGFACE = "huggingface"
    VLLM = "vllm"
    ALTERNATIVE = "alternative"

class CacheType(Enum):
    EMBEDDING = "embedding"
    LLM = "llm"
    FACET = "facet"

class ResultType(Enum):
    FACET = "facet"
    CLUSTER = "cluster"
    INSIGHT = "insight"
    EMBEDDING = "embedding"

@dataclass
class AnalysisSession:
    id: str
    created_at: int
    updated_at: int
    status: SessionStatus
    config: Dict[str, Any]
    current_batch: int = 0
    total_batches: Optional[int] = None
    total_conversations: Optional[int] = None
    processed_conversations: int = 0
    error_message: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class BatchProgress:
    id: Optional[int]
    session_id: str
    batch_number: int
    status: BatchStatus
    started_at: Optional[int] = None
    completed_at: Optional[int] = None
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    processing_time_ms: Optional[int] = None

@dataclass
class AnalysisTemplate:
    id: str
    name: str
    description: Optional[str]
    category: TemplateCategory
    is_public: bool
    config: Dict[str, Any]
    custom_prompts: Optional[Dict[str, Any]] = None
    facet_definitions: Optional[Dict[str, Any]] = None
    created_at: int = field(default_factory=lambda: int(time.time()))
    updated_at: int = field(default_factory=lambda: int(time.time()))
    created_by: Optional[str] = None

@dataclass
class Project:
    id: str
    name: str
    description: Optional[str]
    created_at: int = field(default_factory=lambda: int(time.time()))
    updated_at: int = field(default_factory=lambda: int(time.time()))
    settings: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    is_archived: bool = False

@dataclass
class ProviderConfig:
    id: str
    name: str
    provider_type: ProviderType
    config: Dict[str, Any]
    priority: int = 100
    is_active: bool = True
    is_fallback: bool = False
    rate_limit: Optional[Dict[str, Any]] = None
    cost_per_token: Optional[Dict[str, Any]] = None
    created_at: int = field(default_factory=lambda: int(time.time()))
    updated_at: int = field(default_factory=lambda: int(time.time()))

@dataclass
class CacheEntry:
    id: str
    cache_key: str
    cache_type: CacheType
    provider: Optional[str]
    input_hash: str
    output: Dict[str, Any]
    created_at: int = field(default_factory=lambda: int(time.time()))
    accessed_at: int = field(default_factory=lambda: int(time.time()))
    access_count: int = 1
    ttl_seconds: int = 86400

# ============================================================================
# Migration System
# ============================================================================

@dataclass
class Migration:
    version: int
    name: str
    sql: str

class MigrationManager:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or "briefx.db"
        self.migrations = self._get_migrations()
    
    def _get_migrations(self) -> List[Migration]:
        """Get all database migrations"""
        return [
            Migration(
                version=1,
                name="initial_schema",
                sql=self._get_initial_schema()
            )
        ]
    
    def _get_initial_schema(self) -> str:
        """Initial database schema"""
        return """
        -- Analysis sessions table
        CREATE TABLE IF NOT EXISTS analysis_sessions (
            id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT CHECK(status IN ('pending', 'running', 'paused', 'completed', 'failed')) DEFAULT 'pending',
            config JSON NOT NULL,
            current_batch INTEGER DEFAULT 0,
            total_batches INTEGER,
            total_conversations INTEGER,
            processed_conversations INTEGER DEFAULT 0,
            error_message TEXT,
            results JSON,
            metadata JSON
        );

        -- Batch progress tracking
        CREATE TABLE IF NOT EXISTS batch_progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL REFERENCES analysis_sessions(id) ON DELETE CASCADE,
            batch_number INTEGER NOT NULL,
            status TEXT CHECK(status IN ('pending', 'processing', 'completed', 'failed')) DEFAULT 'pending',
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            input_data JSON,
            output_data JSON,
            error_message TEXT,
            retry_count INTEGER DEFAULT 0,
            processing_time_ms INTEGER,
            UNIQUE(session_id, batch_number)
        );

        -- Analysis templates for reusable configurations
        CREATE TABLE IF NOT EXISTS analysis_templates (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            description TEXT,
            category TEXT CHECK(category IN ('general', 'support', 'sales', 'medical', 'education', 'custom')),
            is_public BOOLEAN DEFAULT false,
            config JSON NOT NULL,
            custom_prompts JSON,
            facet_definitions JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_by TEXT
        );

        -- Projects for organizing analyses
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            settings JSON,
            tags JSON,
            is_archived BOOLEAN DEFAULT false
        );

        -- Link analyses to projects
        CREATE TABLE IF NOT EXISTS project_analyses (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            session_id TEXT NOT NULL REFERENCES analysis_sessions(id) ON DELETE CASCADE,
            name TEXT,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            tags JSON,
            UNIQUE(project_id, session_id)
        );

        -- Provider configurations for multi-provider support
        CREATE TABLE IF NOT EXISTS provider_configs (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            provider_type TEXT CHECK(provider_type IN ('openai', 'anthropic', 'ollama', 'gemini', 'huggingface', 'vllm', 'alternative')),
            config JSON NOT NULL,
            priority INTEGER DEFAULT 100,
            is_active BOOLEAN DEFAULT true,
            is_fallback BOOLEAN DEFAULT false,
            rate_limit JSON,
            cost_per_token JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Track provider usage for cost optimization
        CREATE TABLE IF NOT EXISTS provider_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            provider_id TEXT NOT NULL REFERENCES provider_configs(id),
            session_id TEXT REFERENCES analysis_sessions(id),
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            tokens_used INTEGER,
            cost REAL,
            success BOOLEAN,
            error_message TEXT,
            response_time_ms INTEGER
        );

        -- Partial results for progressive loading
        CREATE TABLE IF NOT EXISTS partial_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL REFERENCES analysis_sessions(id) ON DELETE CASCADE,
            batch_number INTEGER,
            result_type TEXT CHECK(result_type IN ('facet', 'cluster', 'insight', 'embedding')),
            data JSON NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Cache for embeddings and LLM responses
        CREATE TABLE IF NOT EXISTS response_cache (
            id TEXT PRIMARY KEY,
            cache_key TEXT NOT NULL UNIQUE,
            cache_type TEXT CHECK(cache_type IN ('embedding', 'llm', 'facet')),
            provider TEXT,
            input_hash TEXT NOT NULL,
            output JSON NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            access_count INTEGER DEFAULT 1,
            ttl_seconds INTEGER DEFAULT 86400
        );

        -- User preferences and settings
        CREATE TABLE IF NOT EXISTS user_settings (
            id TEXT PRIMARY KEY DEFAULT 'default',
            preferences JSON,
            default_template_id TEXT REFERENCES analysis_templates(id),
            default_providers JSON,
            ui_theme TEXT DEFAULT 'light',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Indexes for performance
        CREATE INDEX IF NOT EXISTS idx_sessions_status ON analysis_sessions(status);
        CREATE INDEX IF NOT EXISTS idx_sessions_created ON analysis_sessions(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_batch_session ON batch_progress(session_id, batch_number);
        CREATE INDEX IF NOT EXISTS idx_batch_status ON batch_progress(status);
        CREATE INDEX IF NOT EXISTS idx_projects_archived ON projects(is_archived);
        CREATE INDEX IF NOT EXISTS idx_provider_active ON provider_configs(is_active, priority);
        CREATE INDEX IF NOT EXISTS idx_cache_key ON response_cache(cache_key, cache_type);
        CREATE INDEX IF NOT EXISTS idx_cache_accessed ON response_cache(accessed_at);
        CREATE INDEX IF NOT EXISTS idx_partial_results_session ON partial_results(session_id, batch_number);

        -- Triggers for updated_at timestamps
        CREATE TRIGGER IF NOT EXISTS update_sessions_timestamp 
        AFTER UPDATE ON analysis_sessions
        BEGIN
            UPDATE analysis_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
        END;

        CREATE TRIGGER IF NOT EXISTS update_templates_timestamp
        AFTER UPDATE ON analysis_templates
        BEGIN
            UPDATE analysis_templates SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
        END;

        CREATE TRIGGER IF NOT EXISTS update_projects_timestamp
        AFTER UPDATE ON projects
        BEGIN
            UPDATE projects SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
        END;

        CREATE TRIGGER IF NOT EXISTS update_providers_timestamp
        AFTER UPDATE ON provider_configs
        BEGIN
            UPDATE provider_configs SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
        END;

        -- Migration tracking
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    
    async def apply_migrations(self, connection) -> None:
        """Apply all pending migrations"""
        # Check current schema version
        try:
            async with connection.execute("SELECT MAX(version) FROM schema_migrations") as cursor:
                result = await cursor.fetchone()
                current_version = result[0] if result and result[0] else 0
        except:
            current_version = 0
        
        # Apply pending migrations
        for migration in self.migrations:
            if migration.version > current_version:
                logger.info(f"Applying migration {migration.version}: {migration.name}")
                
                # Execute migration SQL
                await connection.executescript(migration.sql)
                
                # Record migration
                await connection.execute(
                    "INSERT INTO schema_migrations (version, name) VALUES (?, ?)",
                    (migration.version, migration.name)
                )
                
                await connection.commit()
                logger.info(f"Migration {migration.version} applied successfully")

# ============================================================================
# Enhanced Persistence Manager
# ============================================================================

class EnhancedPersistenceManager:
    def __init__(
        self,
        database_url: Optional[str] = None,
        use_postgresql: bool = False,
        connection_pool_size: int = 10
    ):
        self.database_url = database_url
        self.use_postgresql = use_postgresql
        self.connection_pool_size = connection_pool_size
        self.connection_pool = None
        self.current_session: Optional[AnalysisSession] = None
        self.migration_manager = MigrationManager()
        
        # Setup database path for SQLite
        if not use_postgresql and not database_url:
            self.database_url = "briefx.db"
    
    async def initialize(self) -> None:
        """Initialize database connection and apply migrations"""
        if self.use_postgresql:
            await self._initialize_postgresql()
        else:
            await self._initialize_sqlite()
        
        logger.info("Enhanced persistence manager initialized successfully")
    
    async def _initialize_postgresql(self) -> None:
        """Initialize PostgreSQL connection pool"""
        try:
            if not HAS_ASYNCPG:
                raise ImportError("asyncpg not installed")
            self.connection_pool = await asyncpg.create_pool(
                self.database_url,
                min_size=1,
                max_size=self.connection_pool_size
            )
            
            # Apply migrations
            async with self.connection_pool.acquire() as connection:
                await self.migration_manager.apply_migrations(connection)
                
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL: {e}")
            raise
    
    async def _initialize_sqlite(self) -> None:
        """Initialize SQLite database"""
        try:
            # Create database directory if needed
            db_path = Path(self.database_url)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Connect and apply migrations
            async with aiosqlite.connect(self.database_url) as connection:
                await self.migration_manager.apply_migrations(connection)
                
        except Exception as e:
            logger.error(f"Failed to initialize SQLite: {e}")
            raise
    
    async def close(self) -> None:
        """Close database connections"""
        if self.use_postgresql and self.connection_pool:
            await self.connection_pool.close()
    
    # ========================================================================
    # Session Management
    # ========================================================================
    
    async def create_session(
        self,
        config: Dict[str, Any],
        total_conversations: Optional[int] = None,
        total_batches: Optional[int] = None
    ) -> AnalysisSession:
        """Create a new analysis session"""
        session_id = str(uuid.uuid4())
        current_time = int(time.time())
        
        session = AnalysisSession(
            id=session_id,
            created_at=current_time,
            updated_at=current_time,
            status=SessionStatus.PENDING,
            config=config,
            total_conversations=total_conversations,
            total_batches=total_batches
        )
        
        if self.use_postgresql:
            async with self.connection_pool.acquire() as connection:
                await connection.execute(
                    """
                    INSERT INTO analysis_sessions 
                    (id, status, config, total_batches, total_conversations, processed_conversations)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    session.id,
                    session.status.value,
                    json.dumps(session.config),
                    session.total_batches,
                    session.total_conversations,
                    session.processed_conversations
                )
        else:
            async with aiosqlite.connect(self.database_url) as connection:
                await connection.execute(
                    """
                    INSERT INTO analysis_sessions 
                    (id, status, config, total_batches, total_conversations, processed_conversations)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session.id,
                        session.status.value,
                        json.dumps(session.config),
                        session.total_batches,
                        session.total_conversations,
                        session.processed_conversations
                    )
                )
                await connection.commit()
        
        self.current_session = session
        return session
    
    async def get_session(self, session_id: str) -> Optional[AnalysisSession]:
        """Get session by ID"""
        if self.use_postgresql:
            async with self.connection_pool.acquire() as connection:
                row = await connection.fetchrow(
                    "SELECT * FROM analysis_sessions WHERE id = $1",
                    session_id
                )
        else:
            async with aiosqlite.connect(self.database_url) as connection:
                connection.row_factory = aiosqlite.Row
                async with connection.execute(
                    "SELECT * FROM analysis_sessions WHERE id = ?",
                    (session_id,)
                ) as cursor:
                    row = await cursor.fetchone()
        
        if not row:
            return None
        
        return AnalysisSession(
            id=row['id'],
            created_at=int(time.mktime(datetime.fromisoformat(str(row['created_at'])).timetuple())),
            updated_at=int(time.mktime(datetime.fromisoformat(str(row['updated_at'])).timetuple())),
            status=SessionStatus(row['status']),
            config=json.loads(row['config']) if row['config'] else {},
            current_batch=row['current_batch'],
            total_batches=row['total_batches'],
            total_conversations=row['total_conversations'],
            processed_conversations=row['processed_conversations'],
            error_message=row['error_message'],
            results=json.loads(row['results']) if row['results'] else None,
            metadata=json.loads(row['metadata']) if row['metadata'] else None
        )
    
    async def update_session_status(
        self,
        session_id: str,
        status: SessionStatus,
        error_message: Optional[str] = None
    ) -> None:
        """Update session status"""
        if self.use_postgresql:
            async with self.connection_pool.acquire() as connection:
                await connection.execute(
                    """
                    UPDATE analysis_sessions 
                    SET status = $1, error_message = $2, updated_at = NOW()
                    WHERE id = $3
                    """,
                    status.value,
                    error_message,
                    session_id
                )
        else:
            async with aiosqlite.connect(self.database_url) as connection:
                await connection.execute(
                    """
                    UPDATE analysis_sessions 
                    SET status = ?, error_message = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (status.value, error_message, session_id)
                )
                await connection.commit()
        
        # Update current session if it matches
        if self.current_session and self.current_session.id == session_id:
            self.current_session.status = status
            self.current_session.error_message = error_message
            self.current_session.updated_at = int(time.time())
    
    async def save_session_results(
        self,
        session_id: str,
        results: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save analysis results to session"""
        if self.use_postgresql:
            async with self.connection_pool.acquire() as connection:
                await connection.execute(
                    """
                    UPDATE analysis_sessions 
                    SET results = $1, metadata = $2, updated_at = NOW()
                    WHERE id = $3
                    """,
                    json.dumps(results),
                    json.dumps(metadata) if metadata else None,
                    session_id
                )
        else:
            async with aiosqlite.connect(self.database_url) as connection:
                await connection.execute(
                    """
                    UPDATE analysis_sessions 
                    SET results = ?, metadata = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (
                        json.dumps(results),
                        json.dumps(metadata) if metadata else None,
                        session_id
                    )
                )
                await connection.commit()
        
        # Update current session
        if self.current_session and self.current_session.id == session_id:
            self.current_session.results = results
            self.current_session.metadata = metadata
            self.current_session.updated_at = int(time.time())
    
    # ========================================================================
    # Batch Progress Management
    # ========================================================================
    
    async def save_batch_progress(
        self,
        session_id: str,
        batch_number: int,
        status: BatchStatus,
        output_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        processing_time_ms: Optional[int] = None
    ) -> None:
        """Save batch progress"""
        if self.use_postgresql:
            async with self.connection_pool.acquire() as connection:
                await connection.execute(
                    """
                    INSERT INTO batch_progress 
                    (session_id, batch_number, status, output_data, error_message, processing_time_ms)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (session_id, batch_number) DO UPDATE SET
                    status = $3, output_data = $4, error_message = $5, processing_time_ms = $6,
                    completed_at = CASE WHEN $3 IN ('completed', 'failed') THEN NOW() ELSE completed_at END
                    """,
                    session_id,
                    batch_number,
                    status.value,
                    json.dumps(output_data) if output_data else None,
                    error_message,
                    processing_time_ms
                )
        else:
            async with aiosqlite.connect(self.database_url) as connection:
                await connection.execute(
                    """
                    INSERT OR REPLACE INTO batch_progress 
                    (session_id, batch_number, status, output_data, error_message, processing_time_ms)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        batch_number,
                        status.value,
                        json.dumps(output_data) if output_data else None,
                        error_message,
                        processing_time_ms
                    )
                )
                await connection.commit()
    
    async def get_batch_progress(
        self,
        session_id: str,
        batch_number: Optional[int] = None
    ) -> Union[List[BatchProgress], Optional[BatchProgress]]:
        """Get batch progress for session"""
        if batch_number is not None:
            # Get specific batch
            if self.use_postgresql:
                async with self.connection_pool.acquire() as connection:
                    row = await connection.fetchrow(
                        "SELECT * FROM batch_progress WHERE session_id = $1 AND batch_number = $2",
                        session_id, batch_number
                    )
            else:
                async with aiosqlite.connect(self.database_url) as connection:
                    connection.row_factory = aiosqlite.Row
                    async with connection.execute(
                        "SELECT * FROM batch_progress WHERE session_id = ? AND batch_number = ?",
                        (session_id, batch_number)
                    ) as cursor:
                        row = await cursor.fetchone()
            
            if not row:
                return None
            
            return self._row_to_batch_progress(row)
        else:
            # Get all batches for session
            if self.use_postgresql:
                async with self.connection_pool.acquire() as connection:
                    rows = await connection.fetch(
                        "SELECT * FROM batch_progress WHERE session_id = $1 ORDER BY batch_number",
                        session_id
                    )
            else:
                async with aiosqlite.connect(self.database_url) as connection:
                    connection.row_factory = aiosqlite.Row
                    async with connection.execute(
                        "SELECT * FROM batch_progress WHERE session_id = ? ORDER BY batch_number",
                        (session_id,)
                    ) as cursor:
                        rows = await cursor.fetchall()
            
            return [self._row_to_batch_progress(row) for row in rows]
    
    def _row_to_batch_progress(self, row) -> BatchProgress:
        """Convert database row to BatchProgress object"""
        return BatchProgress(
            id=row['id'],
            session_id=row['session_id'],
            batch_number=row['batch_number'],
            status=BatchStatus(row['status']),
            started_at=int(time.mktime(datetime.fromisoformat(str(row['started_at'])).timetuple())) if row['started_at'] else None,
            completed_at=int(time.mktime(datetime.fromisoformat(str(row['completed_at'])).timetuple())) if row['completed_at'] else None,
            input_data=json.loads(row['input_data']) if row['input_data'] else None,
            output_data=json.loads(row['output_data']) if row['output_data'] else None,
            error_message=row['error_message'],
            retry_count=row['retry_count'],
            processing_time_ms=row['processing_time_ms']
        )
    
    # ========================================================================
    # Response Caching
    # ========================================================================
    
    async def cache_response(
        self,
        cache_key: str,
        cache_type: CacheType,
        input_data: Any,
        output_data: Dict[str, Any],
        provider: Optional[str] = None,
        ttl_seconds: int = 86400
    ) -> None:
        """Cache a response"""
        input_hash = hashlib.sha256(
            json.dumps(input_data, sort_keys=True).encode()
        ).hexdigest()
        
        cache_id = str(uuid.uuid4())
        
        if self.use_postgresql:
            async with self.connection_pool.acquire() as connection:
                await connection.execute(
                    """
                    INSERT INTO response_cache 
                    (id, cache_key, cache_type, provider, input_hash, output, ttl_seconds)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (cache_key) DO UPDATE SET
                    output = $6, accessed_at = NOW(), access_count = response_cache.access_count + 1
                    """,
                    cache_id,
                    cache_key,
                    cache_type.value,
                    provider,
                    input_hash,
                    json.dumps(output_data),
                    ttl_seconds
                )
        else:
            async with aiosqlite.connect(self.database_url) as connection:
                await connection.execute(
                    """
                    INSERT OR REPLACE INTO response_cache 
                    (id, cache_key, cache_type, provider, input_hash, output, ttl_seconds)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cache_id,
                        cache_key,
                        cache_type.value,
                        provider,
                        input_hash,
                        json.dumps(output_data),
                        ttl_seconds
                    )
                )
                await connection.commit()
    
    async def get_cached_response(
        self,
        cache_key: str,
        cache_type: CacheType
    ) -> Optional[Dict[str, Any]]:
        """Get cached response"""
        if self.use_postgresql:
            async with self.connection_pool.acquire() as connection:
                row = await connection.fetchrow(
                    """
                    SELECT output FROM response_cache 
                    WHERE cache_key = $1 AND cache_type = $2 
                    AND (ttl_seconds = 0 OR created_at + INTERVAL '1 second' * ttl_seconds > NOW())
                    """,
                    cache_key, cache_type.value
                )
        else:
            async with aiosqlite.connect(self.database_url) as connection:
                async with connection.execute(
                    """
                    SELECT output FROM response_cache 
                    WHERE cache_key = ? AND cache_type = ? 
                    AND (ttl_seconds = 0 OR datetime(created_at, '+' || ttl_seconds || ' seconds') > datetime('now'))
                    """,
                    (cache_key, cache_type.value)
                ) as cursor:
                    row = await cursor.fetchone()
        
        if not row:
            return None
        
        # Update access statistics
        await self._update_cache_access(cache_key)
        
        return json.loads(row[0])
    
    async def _update_cache_access(self, cache_key: str) -> None:
        """Update cache access statistics"""
        if self.use_postgresql:
            async with self.connection_pool.acquire() as connection:
                await connection.execute(
                    """
                    UPDATE response_cache 
                    SET accessed_at = NOW(), access_count = access_count + 1
                    WHERE cache_key = $1
                    """,
                    cache_key
                )
        else:
            async with aiosqlite.connect(self.database_url) as connection:
                await connection.execute(
                    """
                    UPDATE response_cache 
                    SET accessed_at = CURRENT_TIMESTAMP, access_count = access_count + 1
                    WHERE cache_key = ?
                    """,
                    (cache_key,)
                )
                await connection.commit()
    
    async def cleanup_cache(self, older_than_hours: int = 24) -> int:
        """Clean up expired cache entries"""
        if self.use_postgresql:
            async with self.connection_pool.acquire() as connection:
                result = await connection.execute(
                    """
                    DELETE FROM response_cache 
                    WHERE ttl_seconds > 0 AND created_at + INTERVAL '1 second' * ttl_seconds < NOW() - INTERVAL '$1 hours'
                    """,
                    older_than_hours
                )
                return result
        else:
            async with aiosqlite.connect(self.database_url) as connection:
                cursor = await connection.execute(
                    """
                    DELETE FROM response_cache 
                    WHERE ttl_seconds > 0 AND datetime(created_at, '+' || ttl_seconds || ' seconds') < datetime('now', '-{} hours')
                    """.format(older_than_hours)
                )
                await connection.commit()
                return cursor.rowcount
    
    # ========================================================================
    # Template Management
    # ========================================================================
    
    async def save_template(self, template: AnalysisTemplate) -> None:
        """Save analysis template"""
        if self.use_postgresql:
            async with self.connection_pool.acquire() as connection:
                await connection.execute(
                    """
                    INSERT INTO analysis_templates 
                    (id, name, description, category, is_public, config, custom_prompts, facet_definitions, created_by)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (id) DO UPDATE SET
                    name = $2, description = $3, category = $4, is_public = $5, 
                    config = $6, custom_prompts = $7, facet_definitions = $8, updated_at = NOW()
                    """,
                    template.id, template.name, template.description, template.category.value,
                    template.is_public, json.dumps(template.config),
                    json.dumps(template.custom_prompts) if template.custom_prompts else None,
                    json.dumps(template.facet_definitions) if template.facet_definitions else None,
                    template.created_by
                )
        else:
            async with aiosqlite.connect(self.database_url) as connection:
                await connection.execute(
                    """
                    INSERT OR REPLACE INTO analysis_templates 
                    (id, name, description, category, is_public, config, custom_prompts, facet_definitions, created_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        template.id, template.name, template.description, template.category.value,
                        template.is_public, json.dumps(template.config),
                        json.dumps(template.custom_prompts) if template.custom_prompts else None,
                        json.dumps(template.facet_definitions) if template.facet_definitions else None,
                        template.created_by
                    )
                )
                await connection.commit()
    
    async def get_templates(
        self, 
        category: Optional[TemplateCategory] = None,
        is_public: Optional[bool] = None
    ) -> List[AnalysisTemplate]:
        """Get analysis templates with optional filtering"""
        conditions = []
        params = []
        
        if category:
            conditions.append("category = ?")
            params.append(category.value)
        
        if is_public is not None:
            conditions.append("is_public = ?")
            params.append(is_public)
        
        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        query = f"SELECT * FROM analysis_templates{where_clause} ORDER BY updated_at DESC"
        
        if self.use_postgresql:
            # Convert to PostgreSQL parameter format
            pg_query = query.replace("?", lambda m, counter=[0]: f"${counter[0] + 1}" or counter.__setitem__(0, counter[0] + 1))
            async with self.connection_pool.acquire() as connection:
                rows = await connection.fetch(pg_query, *params)
        else:
            async with aiosqlite.connect(self.database_url) as connection:
                connection.row_factory = aiosqlite.Row
                async with connection.execute(query, params) as cursor:
                    rows = await cursor.fetchall()
        
        templates = []
        for row in rows:
            templates.append(AnalysisTemplate(
                id=row['id'],
                name=row['name'],
                description=row['description'],
                category=TemplateCategory(row['category']),
                is_public=bool(row['is_public']),
                config=json.loads(row['config']),
                custom_prompts=json.loads(row['custom_prompts']) if row['custom_prompts'] else None,
                facet_definitions=json.loads(row['facet_definitions']) if row['facet_definitions'] else None,
                created_at=int(time.mktime(datetime.fromisoformat(str(row['created_at'])).timetuple())),
                updated_at=int(time.mktime(datetime.fromisoformat(str(row['updated_at'])).timetuple())),
                created_by=row['created_by']
            ))
        
        return templates
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    async def get_recent_sessions(self, limit: int = 10) -> List[AnalysisSession]:
        """Get recent analysis sessions"""
        if self.use_postgresql:
            async with self.connection_pool.acquire() as connection:
                rows = await connection.fetch(
                    "SELECT * FROM analysis_sessions ORDER BY created_at DESC LIMIT $1",
                    limit
                )
        else:
            async with aiosqlite.connect(self.database_url) as connection:
                connection.row_factory = aiosqlite.Row
                async with connection.execute(
                    "SELECT * FROM analysis_sessions ORDER BY created_at DESC LIMIT ?",
                    (limit,)
                ) as cursor:
                    rows = await cursor.fetchall()
        
        sessions = []
        for row in rows:
            session = await self._row_to_session(row)
            sessions.append(session)
        
        return sessions
    
    async def _row_to_session(self, row) -> AnalysisSession:
        """Convert database row to AnalysisSession"""
        return AnalysisSession(
            id=row['id'],
            created_at=int(time.mktime(datetime.fromisoformat(str(row['created_at'])).timetuple())),
            updated_at=int(time.mktime(datetime.fromisoformat(str(row['updated_at'])).timetuple())),
            status=SessionStatus(row['status']),
            config=json.loads(row['config']) if row['config'] else {},
            current_batch=row['current_batch'],
            total_batches=row['total_batches'],
            total_conversations=row['total_conversations'],
            processed_conversations=row['processed_conversations'],
            error_message=row['error_message'],
            results=json.loads(row['results']) if row['results'] else None,
            metadata=json.loads(row['metadata']) if row['metadata'] else None
        )
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        stats = {}
        
        tables = [
            'analysis_sessions', 'batch_progress', 'analysis_templates', 
            'projects', 'provider_configs', 'response_cache', 'partial_results'
        ]
        
        if self.use_postgresql:
            async with self.connection_pool.acquire() as connection:
                for table in tables:
                    result = await connection.fetchval(f"SELECT COUNT(*) FROM {table}")
                    stats[f"{table}_count"] = result
        else:
            async with aiosqlite.connect(self.database_url) as connection:
                for table in tables:
                    async with connection.execute(f"SELECT COUNT(*) FROM {table}") as cursor:
                        result = await cursor.fetchone()
                        stats[f"{table}_count"] = result[0] if result else 0
        
        return stats

# ============================================================================
# Global persistence manager instance
# ============================================================================

_persistence_manager: Optional[EnhancedPersistenceManager] = None

def get_persistence_manager() -> EnhancedPersistenceManager:
    """Get global persistence manager instance"""
    global _persistence_manager
    if _persistence_manager is None:
        _persistence_manager = EnhancedPersistenceManager()
    return _persistence_manager

async def initialize_persistence(
    database_url: Optional[str] = None,
    use_postgresql: bool = False
) -> EnhancedPersistenceManager:
    """Initialize global persistence manager"""
    global _persistence_manager
    _persistence_manager = EnhancedPersistenceManager(
        database_url=database_url,
        use_postgresql=use_postgresql
    )
    await _persistence_manager.initialize()
    return _persistence_manager