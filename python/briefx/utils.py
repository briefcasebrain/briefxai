"""Utility functions and logging setup"""

import logging
import sys
from typing import List, Set
from datetime import datetime
import hashlib
from .data.models import ConversationData

# Set up logging
def setup_logging(verbose: bool = False, debug: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if debug else (logging.INFO if verbose else logging.WARNING)
    
    handlers = [logging.StreamHandler(sys.stdout)]

    # Only add file handler when not running under pytest
    if "pytest" not in sys.modules:
        handlers.append(logging.FileHandler('briefx.log'))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
    )
    
    # Reduce noise from external libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)

def dedup_conversations(conversations: List[ConversationData]) -> List[ConversationData]:
    """Remove duplicate conversations based on content hash"""
    seen_hashes: Set[str] = set()
    deduped = []
    
    for conv in conversations:
        # Create content hash
        content = conv.get_text().strip().lower()
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            deduped.append(conv)
    
    return deduped

def calculate_simple_sentiment(text: str) -> float:
    """Calculate basic sentiment score using keyword matching"""
    text_lower = text.lower()
    
    positive_words = [
        'good', 'great', 'excellent', 'amazing', 'wonderful', 
        'fantastic', 'love', 'perfect', 'awesome', 'helpful', 
        'thank', 'appreciate', 'happy', 'pleased', 'satisfied',
        'brilliant', 'outstanding', 'superb', 'magnificent'
    ]
    
    negative_words = [
        'bad', 'terrible', 'awful', 'horrible', 'hate', 'angry',
        'frustrated', 'annoyed', 'disappointed', 'broken', 
        'error', 'crash', 'bug', 'fail', 'wrong', 'useless',
        'worst', 'disgusted', 'furious', 'outraged'
    ]
    
    score = 0
    word_count = 0
    
    # Count positive words
    for word in positive_words:
        count = text_lower.count(word)
        score += count
        word_count += count
    
    # Count negative words
    for word in negative_words:
        count = text_lower.count(word)
        score -= count
        word_count += count
    
    # Normalize to -1.0 to 1.0 range
    if word_count > 0:
        return max(-1.0, min(1.0, score / word_count))
    return 0.0

def determine_category(text: str) -> str:
    """Determine conversation category based on keywords"""
    text_lower = text.lower()
    
    # Define category patterns
    categories = {
        'Bug Report': ['bug', 'error', 'crash', 'broken', 'fix', 'issue with', 'not working', 'problem with'],
        'Feature Request': ['feature', 'request', 'add', 'implement', 'would be nice', 'suggestion', 'enhancement', 'improve'],
        'Support': ['help', 'support', 'how do', 'how to', 'problem', 'question', 'assistance', 'guide'],
        'Feedback': ['thank', 'great', 'love', 'awesome', 'excellent', 'feedback', 'review', 'opinion'],
        'Sales': ['price', 'cost', 'subscription', 'pay', 'billing', 'purchase', 'buy', 'plan'],
        'Documentation': ['document', 'api', 'guide', 'tutorial', 'reference', 'manual', 'instruction']
    }
    
    # Score each category
    category_scores = {}
    for category, keywords in categories.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        if score > 0:
            category_scores[category] = score
    
    # Return category with highest score
    if category_scores:
        return max(category_scores, key=category_scores.get)
    
    return 'General'

def format_duration(seconds: float) -> str:
    """Format duration in human readable format"""
    if seconds < 1:
        return f"{seconds:.2f}s"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def generate_session_id() -> str:
    """Generate a unique session ID"""
    import uuid
    return str(uuid.uuid4())

class ProgressTracker:
    """Simple progress tracking for analysis operations"""
    
    def __init__(self):
        self.sessions = {}
        self.logger = logging.getLogger(__name__)
    
    def start_session(self, session_id: str, total_steps: int = 100):
        """Start tracking progress for a session"""
        self.sessions[session_id] = {
            'progress': 0.0,
            'status': 'started',
            'message': 'Initializing...',
            'start_time': datetime.now(),
            'total_steps': total_steps
        }
        self.logger.info(f"Started session {session_id}")
    
    def update_progress(self, session_id: str, progress: float, message: str = ""):
        """Update progress for a session"""
        if session_id in self.sessions:
            self.sessions[session_id].update({
                'progress': progress,
                'message': message,
                'last_update': datetime.now()
            })
            self.logger.debug(f"Session {session_id}: {progress:.1f}% - {message}")
    
    def complete_session(self, session_id: str, result: str = "Completed"):
        """Mark session as completed"""
        if session_id in self.sessions:
            self.sessions[session_id].update({
                'progress': 100.0,
                'status': 'completed',
                'message': result,
                'end_time': datetime.now()
            })
            self.logger.info(f"Completed session {session_id}")
    
    def error_session(self, session_id: str, error: str):
        """Mark session as error"""
        if session_id in self.sessions:
            self.sessions[session_id].update({
                'status': 'error',
                'message': error,
                'end_time': datetime.now()
            })
            self.logger.error(f"Session {session_id} failed: {error}")
    
    def get_progress(self, session_id: str) -> dict:
        """Get progress for a session"""
        return self.sessions.get(session_id, {})

# Global progress tracker instance
progress_tracker = ProgressTracker()