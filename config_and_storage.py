"""
Merged config and storage/database logic for Basic DBP Bot
Includes: prompts, safety_filters, settings, conversation_store, database_manager, sqlite_adapter
"""

# ========== Standard Library Imports ========== #
import os
import threading
from pathlib import Path
from datetime import datetime
import json
import re
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
from dataclasses import dataclass
import logging

# ========== Third-Party Imports ========== #
from dotenv import load_dotenv

# ========== Utility to load the Groq API key from .env ========== #

def load_groq_api_key():
    """
    Loads the GROQ_API_KEY from .env in the current directory.
    Raises ValueError if not found.
    """
    load_dotenv(dotenv_path=Path(__file__).parent / ".env")
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables. Please check your .env file.")
    return groq_api_key

"""
Prompt templates for the Basic DBP Bot
"""

class PromptTemplates:
    """Collection of all prompt templates used by the bot"""
    
    # Main conversation prompts
    CONVERSATION_PROMPTS = {
        "introduction": """
        You are DBP Bot, a compassionate and empathetic wellness companion. Your role is to help users explore their emotions in a safe and supportive environment through natural conversation.
        
        You understand Dual Brain Psychology - the concept that people have both stressed and mature parts of their mind. Your goal is to help users naturally transition between these states without them realizing you're doing therapeutic work.
        
        Start by introducing yourself warmly and ask an open-ended question to begin the conversation. Be natural and conversational.
        """,
        
        "ongoing": """
        Continue the conversation naturally. You are DBP Bot, a compassionate wellness companion who understands Dual Brain Psychology.
        
        User message:
        {user_message}

        Relevant context:
        {context}

        Respond to the user's message while subtly:
        - Acknowledging both logical and emotional aspects of what they share
        - Helping them explore their feelings without being clinical
        - Gently guiding them toward mature, balanced perspectives when appropriate
        
        Be warm, empathetic, and genuinely interested in their wellbeing.
        """,
        
        "stressed_mind_focus": """
        The user seems to be in a stressed state. Respond with extra empathy and validation.
        
        Focus on:
        - Acknowledging their stress without judgment
        - Validating their feelings
        - Offering gentle perspective without being preachy
        - Creating a safe space for them to express themselves
        - Try to maintain the emotional context of the conversation 
        
        Use phrases like "That sounds really challenging" or "I can understand why you'd feel that way."
        """,
        
        "mature_mind_focus": """
        The user seems to be in a mature, balanced state. Encourage this perspective.
        
        Focus on:
        - Reinforcing their balanced thinking
        - Exploring their insights further
        - Helping them apply this wisdom to other areas
        - Celebrating their growth and self-awareness
        
        Use phrases like "That's a really thoughtful way to look at it" or "Your perspective shows real wisdom."
        """
    }
    
    # State detection prompts
    STATE_DETECTION_PROMPTS = {
        "emotional_state_classifier": """
        You are an expert in emotional state analysis. Analyze the following user message and classify their emotional state.
        
        Classify the state as either "stressed" or "mature":
        
        STRESSED indicators:
        - Anxiety, worry, overwhelm
        - Reactive thinking
        - Feeling stuck or helpless
        - Emotional intensity without perspective
        - Focus on problems without solutions
        - Self-criticism or blame
        
        MATURE indicators:
        - Calm, balanced perspective
        - Thoughtful reflection
        - Problem-solving mindset
        - Self-awareness and insight
        - Acceptance and resilience
        - Compassionate self-talk
        
        Also rate the intensity from 0.0 to 1.0 (0.0 = very mild, 1.0 = very intense).
        
        User message: "{message}"
        
        Respond in this exact format:
        State: [stressed/mature]
        Intensity: [0.0-1.0]
        Confidence: [0.0-1.0]
        Reasoning: [brief explanation]
        """,
        
        "brain_dominance_analyzer": """
        Analyze this message for logical vs emotional content to understand the user's brain dominance.
        
        LOGICAL indicators:
        - Facts, data, analysis
        - Step-by-step thinking
        - Problem-solving approach
        - Structured communication
        - Focus on details and specifics
        
        EMOTIONAL indicators:
        - Feelings and emotions
        - Expressive language
        - Focus on personal experience
        - Imaginative or creative phrasing
        - Emphasis on mood or atmosphere
        
        User message: "{message}"
        
        Respond in this exact format:
        Logical: [score 0.0-1.0]
        Emotional: [score 0.0-1.0]
        Dominant: [logical/emotional/balanced]
        Reasoning: [brief explanation]
        """
    }

# --- safety_filters.py ---
"""
Safety filtering system for the Unified DBP Bot
"""
from typing import List, Dict, Tuple, Optional
import re
from dataclasses import dataclass
from enum import Enum
import logging

class SafetyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class SafetyTerm:
    term: str
    category: str
    severity: int  # 1-5, 5 being most severe
    regex: bool = False
    
    def matches(self, text: str) -> bool:
        """Check if the term matches the given text"""
        if self.regex:
            return bool(re.search(self.term, text, re.IGNORECASE))
        return self.term.lower() in text.lower()

class SafetyFilter:
    """Safety filtering system for content moderation"""
    
    def __init__(self, strictness: SafetyLevel = SafetyLevel.MEDIUM):
        self.strictness = strictness
        self.logger = logging.getLogger(__name__)
        self._load_default_terms()
    
    def _load_default_terms(self):
        """Load default forbidden and approved terms"""
        # Forbidden terms - words/phrases that should trigger safety checks
        self.forbidden_terms = [
            SafetyTerm(r"suicid(e|al)", "self-harm", 5, regex=True),
            SafetyTerm("kill myself", "self-harm", 5),
            SafetyTerm("hurt myself", "self-harm", 4),
            SafetyTerm("hate myself", "self-harm", 3),
            SafetyTerm("want to die", "self-harm", 5),
            SafetyTerm("end it all", "self-harm", 5),
            SafetyTerm("kill you", "violence", 5),
            SafetyTerm("hurt you", "violence", 4),
            SafetyTerm("hate you", "harassment", 3),
            SafetyTerm("stupid", "insult", 2),
            SafetyTerm("idiot", "insult", 2),
        ]
        
        # Approved terms - words that might be flagged but are actually okay in context
        self.approved_terms = [
            "I'm fine",
            "I feel fine",
            "I feel good",
            "I feel great",
            "I feel happy"
        ]
        
        # Contextual patterns where terms might be okay
        self.contextual_patterns = [
            (r"i (used to|once) feel (sad|depressed|anxious)", 1),
            (r"i felt (sad|depressed|anxious) when", 1),
            (r"i was feeling (sad|depressed|anxious)", 1)
        ]
    
    def set_strictness(self, level: SafetyLevel):
        """Update the strictness level of the filter"""
        self.strictness = level
    
    def check_safety(self, text: str) -> Tuple[bool, Dict]:
        """
        Check if the text passes safety filters
        Returns:
            Tuple of (is_safe, details) where details contains information about any issues found
        """
        # Always check approved terms first
        for term in self.approved_terms:
            if term.lower() in text.lower():
                return True, {"approved_term": term, "message": "Contains approved term"}
        
        # Check for contextual patterns where terms might be okay
        for pattern, severity in self.contextual_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True, {"contextual_pattern": pattern, "message": "Matched safe contextual pattern"}
        
        # Check forbidden terms
        issues = []
        for term in self.forbidden_terms:
            if term.matches(text):
                # Skip low severity terms if strictness is low
                if self.strictness == SafetyLevel.LOW and term.severity < 3:
                    continue
                # Skip medium severity terms if strictness is medium
                if self.strictness == SafetyLevel.MEDIUM and term.severity < 2:
                    continue
                
                issues.append({
                    "term": term.term,
                    "category": term.category,
                    "severity": term.severity
                })
        
        if issues:
            return False, {"filtered": True, "issues": issues}
        return True, {"filtered": False}
    
    def filter_response(self, text: str) -> Tuple[str, Dict]:
        """
        Filter/Redact forbidden terms in the text
        Returns:
            Tuple of (filtered_text, details)
        """
        is_safe, details = self.check_safety(text)
        filtered_text = text
        if not is_safe and details.get("issues"):
            for issue in details["issues"]:
                pattern = issue["term"]
                filtered_text = re.sub(pattern, "[REDACTED]", filtered_text, flags=re.IGNORECASE)
        return filtered_text, details

# Default instance
safety_filter = SafetyFilter()

"""
Configuration settings for the Basic DBP Bot
"""
import os
from pathlib import Path
from enum import Enum

#load api key from .env 
try:
    from dotenv import load_dotenv

    # Load .env files (local directory and project root if available)
    _env_path_local = Path(__file__).with_name(".env")
    _env_path_root = Path(__file__).parent.parent.parent / ".env"

    for _p in (_env_path_local, _env_path_root):
        if _p.exists():
            load_dotenv(_p)
except ModuleNotFoundError: 
    pass

class SafetyLevel(str, Enum):
    """Safety filter strictness levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class StateDetectionMethod(str, Enum):
    """State detection methods"""
    LLM = "llm"
    SENTIMENT = "sentiment"
    HYBRID = "hybrid"

class Config:
    """Main configuration class"""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    # Books directory for vector store (RAG)
    BOOKS_DIR = PROJECT_ROOT / "unified_dbp_bot" / "Books"
    # Embedding model for Google Generative AI (Gemini)
    EMBEDDING_MODEL = "models/embedding-001"
    
    # Database
    DATABASE = {
        "type": "sqlite",
        "path": str(Path(__file__).parent / "dbp_bot.sqlite3"),
        "pool_size": 5,
    }
    
    # LLM
    LLM = {
        "model": "gemini-1.5-flash-latest",
        "temperature": 0.7,
        "max_tokens": 1000,
    }
    
    # API Keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    # Session
    SESSION = {
        "max_messages": 10,
        "goal_threshold": 0.8,
    }
    
    # State Detection
    STATE_DETECTION = {
        "method": StateDetectionMethod.HYBRID,
        "confidence_threshold": 0.6,
        "intensity_threshold": 0.5,
    }
    
    # Brain Psychology
    BRAIN = {
        "blending_ratio": 0.5,
        "dominance_threshold": 0.6,
        "logical_weight": 0.4,
        "emotional_weight": 0.6,
    }
    
    # Safety
    SAFETY = {
        "enabled": True,
        "strictness": SafetyLevel.MEDIUM,
        "log_events": True,
    }
    
    # RAG
    RAG = {
        "enabled": True,
        "vector_store": "faiss",
        "top_k": 3,
    }
    
    @classmethod
    def ensure_dirs(cls):
        """Create necessary directories"""
        for d in [cls.DATA_DIR, cls.BOOKS_DIR]:
            d.mkdir(exist_ok=True)
    

# Initialize
config = Config()
config.ensure_dirs()

# --- conversation_store.py ---
"""
Conversation store for managing conversation persistence
"""
import json
from typing import Dict, List, Optional, Any
from datetime import datetime


class ConversationStore:
    """Manages conversation storage and retrieval (in-memory only; persistent logging via DatabaseManager)"""
    def __init__(self):
        self.active_sessions = {}  # In-memory cache for active sessions
    
    def start_new_conversation(self, user_id: str = None) -> str:
        """Start a new conversation session"""
        session_id = self.db_adapter.create_new_session(user_id)
        
        if session_id:
            # Initialize session in memory
            self.active_sessions[session_id] = {
                'messages': [],
                'current_state': None,
                'previous_state': None,
                'message_count': 0,
                'start_time': datetime.now(),
                'user_id': user_id
            }
        
        return session_id
    
    def add_message(self, session_id: str, role: str, content: str, 
                   metadata: Dict[str, Any] = None) -> bool:
        """Add a message to the conversation"""
        if session_id not in self.active_sessions:
            return False
        
        message = {
            'role': role,  # 'user' or 'assistant'
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.active_sessions[session_id]['messages'].append(message)
        self.active_sessions[session_id]['message_count'] += 1
        
        return True
    
    def record_state_change(self, session_id: str, new_state: str, intensity: float,
                          trigger_message: str, detection_confidence: float,
                          detection_method: str) -> bool:
        """Record a state transition"""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        previous_state = session.get('current_state', 'neutral')
        previous_intensity = session.get('current_intensity', 0.5)
        
        # Get context messages (last 3 messages)
        context_messages = []
        messages = session['messages']
        if len(messages) >= 3:
            context_messages = [msg['content'] for msg in messages[-3:]]
        
        # Record in persistent log (JSON via DatabaseManager)
        session['previous_state'] = previous_state
        session['current_state'] = new_state
        session['current_intensity'] = intensity
        return True
    
    def record_brain_assessment(self, session_id: str, logical_score: float,
                              emotional_score: float) -> bool:
        """Record brain dominance assessment"""
        if session_id not in self.active_sessions:
            return False
        
        # Determine dominant side
        if logical_score > emotional_score + 0.1:
            dominant_side = 'left'
        elif emotional_score > logical_score + 0.1:
            dominant_side = 'right'
        else:
            dominant_side = 'balanced'
        
        return True

# --- database_manager.py ---
"""
Database manager for handling all persistent operations (JSON-based only)
"""
import json
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from abc import ABC, abstractmethod

class DatabaseInterface(ABC):
    """Abstract interface for database operations - allows for future DB changes"""
    
    @abstractmethod
    def create_session(self, session_id: str, user_id: str = None) -> bool:
        pass
    
    @abstractmethod
    def add_state_transition(self, session_id: str, from_state: str, to_state: str, 
                           from_intensity: float, to_intensity: float, 
                           trigger_message: str, context_messages: List[str],
                           detection_confidence: float, detection_method: str) -> bool:
        pass
    
    @abstractmethod
    def get_state_matrix(self, session_id: str) -> Dict[str, int]:
        pass
    
    @abstractmethod
    def close_session(self, session_id: str, termination_reason: str) -> bool:
        pass

class DatabaseManager(DatabaseInterface):
    """Manages all persistent operations for the Unified DBP Bot (JSON-based only)"""
    def __init__(self, db_path: str = None):
        # db_path is ignored; JSON file is always used
        self.STATE_JSON_PATH = os.path.join(os.path.dirname(__file__), "state_transitions.json")
        self.STATE_JSON_LOCK = threading.Lock()
        self._init_state_json()

    def create_session(self, session_id: str, user_id: str = None) -> bool:
        # No-op for JSON; session_id is tracked in-memory only
        return True

    
    STATE_JSON_LOCK = threading.Lock()

    def _init_state_json(self):
        if not os.path.exists(self.STATE_JSON_PATH):
            with open(self.STATE_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump([], f)

    def add_state_transition(self, session_id: str, from_state: str, to_state: str, 
                           from_intensity: float, to_intensity: float, 
                           trigger_message: str, context_messages: list,
                           detection_confidence: float, detection_method: str) -> bool:
        try:
            self._init_state_json()
            with self.STATE_JSON_LOCK:
                with open(self.STATE_JSON_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Compose the new transition as a dict
                transition = {
                    "session_id": session_id,
                    "message_sequence": len([d for d in data if d.get("session_id") == session_id]) + 1,
                    "from_state": from_state,
                    "to_state": to_state,
                    "from_intensity": from_intensity,
                    "to_intensity": to_intensity,
                    "timestamp": datetime.now().isoformat(sep=" ", timespec="seconds"),
                    "trigger_message": trigger_message,
                    "context_messages": context_messages,
                    "detection_confidence": detection_confidence,
                    "detection_method": detection_method,
                }
                data.append(transition)
                # DEBUG: About to write to JSON
                with open(self.STATE_JSON_PATH, "w", encoding="utf-8") as f:
                    print(f"[DEBUG] Writing {len(data)} transitions to {self.STATE_JSON_PATH}")
                    json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"[JSON] Error adding state transition: {e}")
            return False
    
    def get_state_matrix(self, session_id: str) -> Dict[str, int]:
        # Reads from JSON state transitions
        try:
            with open(self.STATE_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            matrix = {}
            for d in data:
                if d.get("session_id") == session_id:
                    key = f"{d['from_state']}_to_{d['to_state']}"
                    matrix[key] = matrix.get(key, 0) + 1
            return matrix
        except Exception as e:
            print(f"[JSON] Error reading state matrix: {e}")
            return {}
    
    def close_session(self, session_id: str, termination_reason: str) -> bool:
        # No-op for JSON; session end can be tracked in-memory
        return True
    
    def add_brain_dominance_record(self, session_id: str, logical_score: float, emotional_score: float, dominant_side: str) -> bool:
        # No-op for JSON; not implemented
        return True
    
    def add_safety_event(self, session_id: str, message_sequence: int, event_type: str, severity: str, metadata: Dict[str, Any]) -> bool:
        # No-op for JSON; not implemented
        return True
    
    def print_state_matrix(self, session_id: str):
        matrix = self.get_state_matrix(session_id)
        print(f"State transition matrix for session {session_id}: {matrix}")
    def close(self):
        pass  # No DB connection to close

    
    def create_new_session(self, user_id: str = None) -> str:
        session_id = f"sess_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO conversations (session_id, user_id, start_time) VALUES (?, ?, ?)",
            (session_id, user_id, datetime.now())
        )
        self.conn.commit()
        return session_id
    
    def record_state_transition(self, session_id: str, previous_state: str, new_state: str,
                              previous_intensity: float, new_intensity: float,
                              trigger_message: str, context_messages: List[str],
                              detection_confidence: float, detection_method: str) -> bool:
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO state_transitions (session_id, message_sequence, from_state, to_state, from_intensity, to_intensity, timestamp, trigger_message, context_messages, detection_confidence, detection_method) VALUES (?, (SELECT COALESCE(MAX(message_sequence), 0) + 1 FROM state_transitions WHERE session_id = ?), ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id, session_id, previous_state, new_state, previous_intensity, new_intensity,
                datetime.now(), trigger_message, str(context_messages), detection_confidence, detection_method
            )
        )
        self.conn.commit()
        return True
    
    def record_brain_dominance(self, session_id: str, logical_score: float, emotional_score: float, dominant_side: str) -> bool:
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO brain_dominance_tracking (session_id, message_sequence, logical_score, emotional_score, dominant_side, timestamp) VALUES (?, (SELECT COALESCE(MAX(message_sequence), 0) + 1 FROM brain_dominance_tracking WHERE session_id = ?), ?, ?, ?, ?)",
            (
                session_id, session_id, logical_score, emotional_score, dominant_side, datetime.now()
            )
        )
        self.conn.commit()
        return True
    
    def close(self):
        if self.conn:
            self.conn.close()
