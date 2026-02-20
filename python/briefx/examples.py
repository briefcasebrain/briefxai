"""
Conversation data generation utilities for testing and benchmarking
"""

import random
from datetime import datetime
from typing import List, Optional

from .data.models import ConversationData, Message


# Topic templates for diverse conversations
TOPICS = {
    "technical": [
        "debugging a Python application",
        "optimizing database queries",
        "implementing a REST API",
        "setting up CI/CD pipeline",
        "troubleshooting Docker containers",
        "explaining machine learning concepts",
        "reviewing code architecture",
        "discussing microservices patterns"
    ],
    "creative": [
        "writing a short story",
        "brainstorming marketing ideas",
        "designing a user interface",
        "creating content strategy",
        "developing a brand identity",
        "composing music lyrics",
        "planning an event",
        "crafting social media campaigns"
    ],
    "analytical": [
        "analyzing market trends",
        "reviewing financial data",
        "evaluating business strategies",
        "assessing risk factors",
        "interpreting research findings",
        "comparing product features",
        "examining user behavior",
        "studying competitive landscape"
    ],
    "educational": [
        "explaining scientific concepts",
        "teaching programming basics",
        "discussing historical events",
        "learning a new language",
        "understanding mathematics",
        "exploring philosophy",
        "studying literature",
        "reviewing academic papers"
    ],
    "problem_solving": [
        "fixing technical issues",
        "resolving conflicts",
        "improving processes",
        "finding solutions",
        "overcoming challenges",
        "optimizing workflows",
        "addressing concerns",
        "tackling obstacles"
    ]
}

# Templates for generating messages
USER_TEMPLATES = [
    "Can you help me with {topic}?",
    "I need assistance with {topic}.",
    "What's the best approach for {topic}?",
    "How would you handle {topic}?",
    "I'm working on {topic} and need guidance.",
    "Could you explain {topic} to me?",
    "What are your thoughts on {topic}?",
    "I'm stuck with {topic}. Any suggestions?",
    "Can we discuss {topic}?",
    "I'd like to learn more about {topic}."
]

ASSISTANT_TEMPLATES = [
    "I'd be happy to help with {topic}. Here's what I suggest:",
    "Regarding {topic}, here are some key points to consider:",
    "For {topic}, I recommend the following approach:",
    "Let me explain {topic} in detail:",
    "Here's a comprehensive overview of {topic}:",
    "Based on my analysis of {topic}:",
    "The best practices for {topic} include:",
    "When dealing with {topic}, it's important to:",
    "Let me break down {topic} for you:",
    "Here's my perspective on {topic}:"
]

FOLLOW_UP_USER = [
    "That makes sense. What about {aspect}?",
    "Could you elaborate on {aspect}?",
    "How does {aspect} factor into this?",
    "What if we consider {aspect}?",
    "I'm curious about {aspect}.",
    "Can you provide an example of {aspect}?",
    "What are the implications of {aspect}?",
    "How would {aspect} affect the outcome?",
    "That's interesting. Tell me more about {aspect}.",
    "What's your experience with {aspect}?"
]

FOLLOW_UP_ASSISTANT = [
    "Excellent question about {aspect}. Let me explain:",
    "Regarding {aspect}, here's what you need to know:",
    "{aspect} is indeed important. Consider this:",
    "Let me address {aspect} specifically:",
    "When it comes to {aspect}:",
    "The key thing about {aspect} is:",
    "To answer your question about {aspect}:",
    "Here's how {aspect} fits into the bigger picture:",
    "Let me provide more context on {aspect}:",
    "{aspect} plays a crucial role because:"
]

ASPECTS = [
    "performance implications",
    "best practices",
    "common pitfalls",
    "alternative approaches",
    "real-world applications",
    "implementation details",
    "potential challenges",
    "optimization strategies",
    "security considerations",
    "scalability concerns",
    "cost factors",
    "timeline estimates",
    "resource requirements",
    "success metrics",
    "risk mitigation"
]


def generate_conversation_content(
    topic: str,
    min_messages: int = 4,
    max_messages: int = 10,
    seed: Optional[int] = None
) -> List[Message]:
    """Generate conversation content for a given topic"""
    
    if seed is not None:
        random.seed(seed)
    
    messages = []
    num_messages = random.randint(min_messages, max_messages)
    
    # Ensure even number for proper conversation flow
    if num_messages % 2 != 0:
        num_messages += 1
    
    for i in range(num_messages):
        if i == 0:
            # Initial user message
            template = random.choice(USER_TEMPLATES)
            content = template.format(topic=topic)
            messages.append(Message(role="user", content=content))
        
        elif i == 1:
            # Initial assistant response
            template = random.choice(ASSISTANT_TEMPLATES)
            content = template.format(topic=topic)
            # Add some detail
            details = [
                f"First, consider the fundamental aspects of {topic}.",
                f"The approach depends on your specific requirements.",
                f"There are several factors to keep in mind.",
                f"Let me outline the key considerations.",
                f"Here's a step-by-step breakdown."
            ]
            content += " " + random.choice(details)
            messages.append(Message(role="assistant", content=content))
        
        elif i % 2 == 0:
            # User follow-up
            aspect = random.choice(ASPECTS)
            template = random.choice(FOLLOW_UP_USER)
            content = template.format(aspect=aspect)
            messages.append(Message(role="user", content=content))
        
        else:
            # Assistant follow-up
            aspect = random.choice(ASPECTS)
            template = random.choice(FOLLOW_UP_ASSISTANT)
            content = template.format(aspect=aspect)
            # Add substance
            substance = [
                f"This involves careful consideration of various factors.",
                f"The implementation requires attention to detail.",
                f"Best practices suggest a methodical approach.",
                f"Industry standards recommend specific guidelines.",
                f"Experience shows that this approach works well.",
                f"Research indicates several effective strategies.",
                f"Practical applications demonstrate clear benefits."
            ]
            content += " " + random.choice(substance)
            messages.append(Message(role="assistant", content=content))
    
    return messages


def generate_example_conversations(
    count: int = 10,
    seed: Optional[int] = None,
    category: Optional[str] = None
) -> List[ConversationData]:
    """Generate example conversations for testing"""
    
    if seed is not None:
        random.seed(seed)
    
    conversations = []
    
    for i in range(count):
        # Select category
        if category and category in TOPICS:
            selected_category = category
        else:
            selected_category = random.choice(list(TOPICS.keys()))
        
        # Select topic from category
        topic = random.choice(TOPICS[selected_category])
        
        # Generate messages
        messages = generate_conversation_content(
            topic,
            min_messages=4,
            max_messages=12,
            seed=seed + i if seed else None
        )
        
        # Create conversation with metadata
        conversation = ConversationData(
            messages=messages,
            metadata={
                "id": f"example_{i+1}",
                "category": selected_category,
                "topic": topic,
                "generated_at": datetime.utcnow().isoformat(),
                "message_count": len(messages),
                "is_example": True
            }
        )
        
        conversations.append(conversation)
    
    return conversations


def generate_diverse_dataset(
    total_conversations: int = 100,
    seed: Optional[int] = None
) -> List[ConversationData]:
    """Generate a diverse dataset with balanced categories"""
    
    if seed is not None:
        random.seed(seed)
    
    conversations = []
    categories = list(TOPICS.keys())
    
    # Distribute conversations across categories
    per_category = total_conversations // len(categories)
    remainder = total_conversations % len(categories)
    
    for i, category in enumerate(categories):
        # Add extra conversation to some categories to match total
        count = per_category + (1 if i < remainder else 0)
        
        category_conversations = generate_example_conversations(
            count=count,
            category=category,
            seed=seed + i * 1000 if seed else None
        )
        
        conversations.extend(category_conversations)
    
    # Shuffle to mix categories
    random.shuffle(conversations)
    
    # Re-number IDs
    for i, conv in enumerate(conversations):
        conv.metadata["id"] = f"example_{i+1}"
    
    return conversations


def save_examples_to_file(
    conversations: List[ConversationData],
    filename: str = "example_conversations.json"
):
    """Save example conversations to a JSON file"""
    
    import json
    
    data = {
        "conversations": [conv.to_dict() for conv in conversations],
        "metadata": {
            "total": len(conversations),
            "generated_at": datetime.utcnow().isoformat(),
            "source": "synthetic_data",
            "version": "1.0.0"
        }
    }
    
    with open(filename, "w") as f:
        json.dump(data, f, indent=2, default=str)
    
    return filename


# Utility function for quick testing
def create_test_dataset():
    """Create a small test dataset for quick testing"""
    
    return generate_example_conversations(
        count=5,
        seed=42
    )