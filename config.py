"""
Data Copilot MVP Configuration
Production-ready settings for the agentic SQL generation system
"""
import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Central configuration class for the Data Copilot system"""
    
    # Base paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    VECTOR_STORE_DIR = BASE_DIR / "vector_store"
    
    # API Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    # Database Configuration
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data_copilot.db")
    DATABASE_PATH = BASE_DIR / "data_copilot.db"
    
    # Server Configuration
    API_HOST = os.getenv("API_HOST", "localhost")
    API_PORT = int(os.getenv("API_PORT", 8000))
    STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", 8501))
    
    # Security & Performance Limits
    MAX_QUERY_TIME = int(os.getenv("MAX_QUERY_TIME", 30))
    MAX_RESULT_ROWS = int(os.getenv("MAX_RESULT_ROWS", 1000))
    ENABLE_SQL_INJECTION_PROTECTION = os.getenv("ENABLE_SQL_INJECTION_PROTECTION", "true").lower() == "true"
    
    # Embeddings Configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 384))
    
    # Business Rules
    MIN_CONTEXT_SIMILARITY = float(os.getenv("MIN_CONTEXT_SIMILARITY", 0.7))
    MAX_CONTEXT_RESULTS = int(os.getenv("MAX_CONTEXT_RESULTS", 10))
    DEFAULT_QUERY_TIMEOUT = int(os.getenv("DEFAULT_QUERY_TIMEOUT", 15))
    
    # Metrics & Monitoring
    ENABLE_METRICS = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    METRICS_RETENTION_DAYS = int(os.getenv("METRICS_RETENTION_DAYS", 30))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Agent Configuration
    AGENT_CONFIG = {
        "context_retrieval": {
            "max_results": 5,
            "similarity_threshold": 0.75
        },
        "sql_generation": {
            "temperature": 0.1,
            "max_tokens": 1000
        },
        "validation": {
            "enable_syntax_check": True,
            "enable_injection_check": True,
            "enable_performance_check": True
        },
        "explanation": {
            "include_business_context": True,
            "max_explanation_length": 500
        }
    }
    
    # Superstore Schema Configuration
    SUPERSTORE_SCHEMA = {
        "table_name": "superstore",
        "columns": {
            "row_id": "INTEGER PRIMARY KEY",
            "order_id": "TEXT",
            "order_date": "DATE",
            "ship_date": "DATE", 
            "ship_mode": "TEXT",
            "customer_id": "TEXT",
            "customer_name": "TEXT",
            "segment": "TEXT",
            "country": "TEXT",
            "city": "TEXT",
            "state": "TEXT",
            "postal_code": "TEXT",
            "region": "TEXT",
            "product_id": "TEXT",
            "category": "TEXT",
            "sub_category": "TEXT",
            "product_name": "TEXT",
            "sales": "REAL",
            "quantity": "INTEGER",
            "discount": "REAL",
            "profit": "REAL"
        }
    }
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist"""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.VECTOR_STORE_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """Validate configuration and return status"""
        issues = []
        
        if not cls.OPENAI_API_KEY:
            issues.append("OPENAI_API_KEY not set")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "config_summary": {
                "api_configured": bool(cls.OPENAI_API_KEY),
                "database_path": str(cls.DATABASE_PATH),
                "vector_store_path": str(cls.VECTOR_STORE_DIR),
                "api_endpoint": f"{cls.API_HOST}:{cls.API_PORT}",
                "streamlit_endpoint": f"{cls.API_HOST}:{cls.STREAMLIT_PORT}"
            }
        }

# Global config instance
config = Config()
config.ensure_directories()