"""File parsers for different conversation formats"""

import json
import csv
from io import StringIO
from typing import List, Dict, Any, Union
from .models import ConversationData, Message

def parse_json_conversations(content: Union[str, bytes]) -> List[ConversationData]:
    """Parse JSON conversation data"""
    if isinstance(content, bytes):
        content = content.decode('utf-8')
    
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    
    conversations = []
    
    if isinstance(data, list):
        for item in data:
            conv = _parse_single_conversation(item)
            if conv:
                conversations.append(conv)
    elif isinstance(data, dict):
        conv = _parse_single_conversation(data)
        if conv:
            conversations.append(conv)
    
    return conversations

def parse_csv_conversations(content: Union[str, bytes]) -> List[ConversationData]:
    """Parse CSV conversation data"""
    if isinstance(content, bytes):
        content = content.decode('utf-8')

    conversations = []
    current_conversation = []
    # Use a mutable dict so _process_csv_row can update current_id
    state = {'current_id': None}

    # Try to detect CSV format
    reader = csv.reader(StringIO(content))

    try:
        # Skip header if it exists
        first_row = next(reader)
        if not _is_data_row(first_row):
            # This is likely a header, continue with next row
            pass
        else:
            # First row is data, process it
            _process_csv_row(first_row, current_conversation, state, conversations)

        for row in reader:
            _process_csv_row(row, current_conversation, state, conversations)

    except Exception as e:
        raise ValueError(f"Error parsing CSV: {e}")

    # Add the last conversation
    if current_conversation:
        conversations.append(ConversationData(
            messages=current_conversation,
            metadata={}
        ))

    return conversations

def parse_text_conversations(content: Union[str, bytes]) -> List[ConversationData]:
    """Parse plain text conversation data"""
    if isinstance(content, bytes):
        content = content.decode('utf-8')
    
    messages = []
    current_role = "user"
    
    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Detect role markers
        if line.lower().startswith(('user:', 'human:', 'question:')):
            current_role = "user" 
            content_text = line.split(':', 1)[1].strip() if ':' in line else line
        elif line.lower().startswith(('assistant:', 'ai:', 'answer:', 'bot:')):
            current_role = "assistant"
            content_text = line.split(':', 1)[1].strip() if ':' in line else line
        else:
            content_text = line
        
        if content_text:
            messages.append(Message(
                role=current_role,
                content=content_text
            ))
    
    if messages:
        return [ConversationData(messages=messages, metadata={})]
    return []

def _parse_single_conversation(data: Dict[str, Any]) -> ConversationData:
    """Parse a single conversation from dict"""
    if 'messages' in data:
        # Standard format with messages array
        messages = []
        for msg_data in data['messages']:
            if isinstance(msg_data, dict) and 'role' in msg_data and 'content' in msg_data:
                messages.append(Message(
                    role=msg_data['role'],
                    content=msg_data['content']
                ))
        
        return ConversationData(
            messages=messages,
            metadata=data.get('metadata', {})
        )
    
    elif 'text' in data or 'content' in data:
        # Simple text format
        text = data.get('text', data.get('content', ''))
        return ConversationData(
            messages=[Message(role="user", content=str(text))],
            metadata=data
        )
    
    return None

def _is_data_row(row: List[str]) -> bool:
    """Check if a CSV row contains data (not header)"""
    if len(row) < 2:
        return False
    
    # Common header patterns
    headers = ['id', 'role', 'content', 'message', 'text', 'conversation_id']
    return not any(cell.lower().strip() in headers for cell in row[:3])

def _process_csv_row(row: List[str], current_conversation: List[Message],
                    state: Dict[str, Any], conversations: List[ConversationData]):
    """Process a single CSV row.

    Args:
        state: mutable dict with 'current_id' key to track conversation boundaries.
    """
    if len(row) < 3:
        return

    conv_id = row[0].strip()
    role = row[1].strip().lower()
    content = ','.join(row[2:]).strip().strip('"')

    # Check if starting a new conversation
    if state['current_id'] is not None and conv_id != state['current_id']:
        if current_conversation:
            conversations.append(ConversationData(
                messages=current_conversation.copy(),
                metadata={}
            ))
            current_conversation.clear()

    state['current_id'] = conv_id
    if content:
        current_conversation.append(Message(
            role="user" if role in ["user", "human", "question"] else "assistant",
            content=content
        ))

def detect_file_format(content: Union[str, bytes], filename: str = "") -> str:
    """Detect the format of conversation data"""
    if isinstance(content, bytes):
        try:
            content = content.decode('utf-8')
        except UnicodeDecodeError:
            return "binary"
    
    # Check filename extension first
    filename_lower = filename.lower()
    if filename_lower.endswith('.json'):
        return "json"
    elif filename_lower.endswith('.csv'):
        return "csv"
    elif filename_lower.endswith('.txt'):
        return "text"
    
    # Try to detect from content
    content_stripped = content.strip()
    
    # JSON detection
    if content_stripped.startswith(('{', '[')):
        try:
            json.loads(content_stripped)
            return "json"
        except json.JSONDecodeError:
            pass
    
    # CSV detection (look for comma-separated values)
    lines = content_stripped.split('\n')[:5]  # Check first 5 lines
    if len(lines) > 1:
        comma_count = sum(line.count(',') for line in lines)
        if comma_count > len(lines):  # More commas than lines suggests CSV
            return "csv"
    
    # Default to text
    return "text"