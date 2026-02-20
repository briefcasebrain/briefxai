"""Enhanced session management system for analysis operations"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List, Any, Callable
from enum import Enum
import threading

logger = logging.getLogger(__name__)

class SessionStatus(Enum):
    """Session status enumeration"""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class SessionEvent:
    """Represents an event in a session"""
    timestamp: float
    event_type: str
    message: str
    data: Optional[Dict[str, Any]] = None
    progress: Optional[float] = None

@dataclass
class SessionInfo:
    """Complete session information"""
    session_id: str
    status: SessionStatus
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    current_step: str = "Initializing"
    total_steps: int = 100
    metadata: Dict[str, Any] = None
    events: List[SessionEvent] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.events is None:
            self.events = []

    @property
    def current_message(self) -> str:
        """Alias for current_step, used by app.py endpoints."""
        return self.current_step

class SessionManager:
    """Advanced session management with persistence and event streaming"""
    
    def __init__(self, persistence_dir: Optional[Path] = None):
        self.sessions: Dict[str, SessionInfo] = {}
        self.persistence_dir = persistence_dir or Path("./sessions")
        self.persistence_dir.mkdir(exist_ok=True)
        self.event_subscribers: Dict[str, List[Callable]] = {}
        self.cleanup_interval = 3600  # 1 hour in seconds
        self.max_session_age = 24 * 3600  # 24 hours
        self._cleanup_thread = None
        self._start_cleanup_task()
        
        # Load existing sessions
        self._load_sessions()
    
    def create_session(
        self, 
        session_id: str, 
        total_steps: int = 100,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SessionInfo:
        """Create a new analysis session"""
        
        if session_id in self.sessions:
            logger.warning(f"Session {session_id} already exists, returning existing session")
            return self.sessions[session_id]
        
        session = SessionInfo(
            session_id=session_id,
            status=SessionStatus.CREATED,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            total_steps=total_steps,
            metadata=metadata or {}
        )
        
        self.sessions[session_id] = session
        self._persist_session(session)
        self._emit_event(session_id, "session_created", f"Session {session_id} created")
        
        logger.info(f"Created session {session_id}")
        return session
    
    def start_session(self, session_id: str) -> bool:
        """Start a session"""
        session = self.sessions.get(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            return False
        
        session.status = SessionStatus.RUNNING
        session.started_at = datetime.now()
        session.updated_at = datetime.now()
        
        self._persist_session(session)
        self._emit_event(session_id, "session_started", "Session started")
        
        logger.info(f"Started session {session_id}")
        return True
    
    def update_progress(
        self, 
        session_id: str, 
        progress: float, 
        step_message: str,
        step_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update session progress"""
        session = self.sessions.get(session_id)
        if not session:
            logger.warning(f"Session {session_id} not found for progress update")
            return False
        
        session.progress = max(0.0, min(100.0, progress))
        session.current_step = step_message
        session.updated_at = datetime.now()
        
        # Add event
        event = SessionEvent(
            timestamp=time.time(),
            event_type="progress_update",
            message=step_message,
            data=step_data,
            progress=progress
        )
        session.events.append(event)
        
        # Keep only last 100 events to prevent memory issues
        if len(session.events) > 100:
            session.events = session.events[-100:]
        
        self._persist_session(session)
        self._emit_event(session_id, "progress_update", step_message, {"progress": progress})
        
        logger.debug(f"Session {session_id}: {progress:.1f}% - {step_message}")
        return True
    
    def complete_session(
        self, 
        session_id: str, 
        result_message: str = "Completed successfully",
        result_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Mark session as completed"""
        session = self.sessions.get(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            return False
        
        session.status = SessionStatus.COMPLETED
        session.progress = 100.0
        session.current_step = result_message
        session.completed_at = datetime.now()
        session.updated_at = datetime.now()
        
        # Add completion event
        event = SessionEvent(
            timestamp=time.time(),
            event_type="session_completed",
            message=result_message,
            data=result_data,
            progress=100.0
        )
        session.events.append(event)
        
        self._persist_session(session)
        self._emit_event(session_id, "session_completed", result_message, result_data)
        
        logger.info(f"Completed session {session_id}: {result_message}")
        return True
    
    def fail_session(
        self, 
        session_id: str, 
        error_message: str,
        error_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Mark session as failed"""
        session = self.sessions.get(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            return False
        
        session.status = SessionStatus.FAILED
        session.current_step = f"Failed: {error_message}"
        session.updated_at = datetime.now()
        
        # Add failure event
        event = SessionEvent(
            timestamp=time.time(),
            event_type="session_failed",
            message=error_message,
            data=error_data
        )
        session.events.append(event)
        
        self._persist_session(session)
        self._emit_event(session_id, "session_failed", error_message, error_data)
        
        logger.error(f"Session {session_id} failed: {error_message}")
        return True
    
    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """Get session information"""
        return self.sessions.get(session_id)

    def get_session_info(self, session_id: str) -> Optional[SessionInfo]:
        """Alias for get_session(), used by app.py endpoints."""
        return self.get_session(session_id)
    
    def get_session_progress(self, session_id: str) -> Dict[str, Any]:
        """Get session progress information"""
        session = self.sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}
        
        return {
            "session_id": session_id,
            "status": session.status.value,
            "progress": session.progress,
            "current_step": session.current_step,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "started_at": session.started_at.isoformat() if session.started_at else None,
            "completed_at": session.completed_at.isoformat() if session.completed_at else None,
            "metadata": session.metadata
        }
    
    def get_session_events(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent session events"""
        session = self.sessions.get(session_id)
        if not session:
            return []
        
        events = session.events[-limit:] if session.events else []
        return [
            {
                "timestamp": event.timestamp,
                "event_type": event.event_type,
                "message": event.message,
                "data": event.data,
                "progress": event.progress
            }
            for event in events
        ]
    
    def list_sessions(self, status_filter: Optional[SessionStatus] = None) -> List[Dict[str, Any]]:
        """List all sessions with optional status filter"""
        sessions = []
        for session in self.sessions.values():
            if status_filter and session.status != status_filter:
                continue
            
            sessions.append({
                "session_id": session.session_id,
                "status": session.status.value,
                "progress": session.progress,
                "current_step": session.current_step,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat()
            })
        
        return sorted(sessions, key=lambda x: x["updated_at"], reverse=True)
    
    def subscribe_to_events(self, session_id: str, callback: Callable) -> bool:
        """Subscribe to session events"""
        if session_id not in self.event_subscribers:
            self.event_subscribers[session_id] = []
        
        self.event_subscribers[session_id].append(callback)
        logger.debug(f"Added event subscriber for session {session_id}")
        return True
    
    def unsubscribe_from_events(self, session_id: str, callback: Callable) -> bool:
        """Unsubscribe from session events"""
        if session_id in self.event_subscribers:
            try:
                self.event_subscribers[session_id].remove(callback)
                logger.debug(f"Removed event subscriber for session {session_id}")
                return True
            except ValueError:
                pass
        return False
    
    def cleanup_old_sessions(self) -> int:
        """Clean up old completed/failed sessions"""
        current_time = datetime.now()
        to_remove = []
        
        for session_id, session in self.sessions.items():
            if session.status in [SessionStatus.COMPLETED, SessionStatus.FAILED]:
                age = (current_time - session.updated_at).total_seconds()
                if age > self.max_session_age:
                    to_remove.append(session_id)
        
        for session_id in to_remove:
            self._remove_session(session_id)
        
        logger.info(f"Cleaned up {len(to_remove)} old sessions")
        return len(to_remove)
    
    def _emit_event(
        self, 
        session_id: str, 
        event_type: str, 
        message: str, 
        data: Optional[Dict[str, Any]] = None
    ):
        """Emit event to subscribers"""
        if session_id in self.event_subscribers:
            event_data = {
                "session_id": session_id,
                "event_type": event_type,
                "message": message,
                "data": data,
                "timestamp": time.time()
            }
            
            for callback in self.event_subscribers[session_id]:
                try:
                    callback(event_data)
                except Exception as e:
                    logger.warning(f"Event callback failed: {e}")
    
    def _persist_session(self, session: SessionInfo):
        """Persist session to disk"""
        try:
            session_file = self.persistence_dir / f"{session.session_id}.json"
            session_data = {
                "session_id": session.session_id,
                "status": session.status.value,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "started_at": session.started_at.isoformat() if session.started_at else None,
                "completed_at": session.completed_at.isoformat() if session.completed_at else None,
                "progress": session.progress,
                "current_step": session.current_step,
                "total_steps": session.total_steps,
                "metadata": session.metadata,
                "events": [
                    {
                        "timestamp": event.timestamp,
                        "event_type": event.event_type,
                        "message": event.message,
                        "data": event.data,
                        "progress": event.progress
                    }
                    for event in session.events
                ]
            }
            
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to persist session {session.session_id}: {e}")
    
    def _load_sessions(self):
        """Load existing sessions from disk"""
        try:
            for session_file in self.persistence_dir.glob("*.json"):
                try:
                    with open(session_file, 'r') as f:
                        data = json.load(f)
                    
                    # Convert back to SessionInfo
                    events = [
                        SessionEvent(
                            timestamp=event["timestamp"],
                            event_type=event["event_type"],
                            message=event["message"],
                            data=event.get("data"),
                            progress=event.get("progress")
                        )
                        for event in data.get("events", [])
                    ]
                    
                    session = SessionInfo(
                        session_id=data["session_id"],
                        status=SessionStatus(data["status"]),
                        created_at=datetime.fromisoformat(data["created_at"]),
                        updated_at=datetime.fromisoformat(data["updated_at"]),
                        started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
                        completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
                        progress=data.get("progress", 0.0),
                        current_step=data.get("current_step", "Unknown"),
                        total_steps=data.get("total_steps", 100),
                        metadata=data.get("metadata", {}),
                        events=events
                    )
                    
                    self.sessions[session.session_id] = session
                    
                except Exception as e:
                    logger.warning(f"Failed to load session from {session_file}: {e}")
                    
            logger.info(f"Loaded {len(self.sessions)} existing sessions")
            
        except Exception as e:
            logger.warning(f"Failed to load sessions: {e}")
    
    def _remove_session(self, session_id: str):
        """Remove session and its persistence file"""
        if session_id in self.sessions:
            del self.sessions[session_id]
        
        # Remove persistence file
        session_file = self.persistence_dir / f"{session_id}.json"
        if session_file.exists():
            session_file.unlink()
        
        # Remove event subscribers
        if session_id in self.event_subscribers:
            del self.event_subscribers[session_id]
    
    def _start_cleanup_task(self):
        """Start background cleanup task"""
        def cleanup_loop():
            while True:
                try:
                    time.sleep(self.cleanup_interval)
                    self.cleanup_old_sessions()
                except Exception as e:
                    logger.warning(f"Cleanup task failed: {e}")
        
        self._cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self._cleanup_thread.start()

# Global session manager instance
session_manager = SessionManager()