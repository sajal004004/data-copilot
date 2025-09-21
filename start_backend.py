"""
Startup Script for Data Copilot MVP Backend
Production-ready FastAPI server with comprehensive error handling
"""
import uvicorn
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Start the FastAPI backend server"""
    
    logger.info("🚀 Starting Data Copilot MVP Backend...")
    logger.info(f"📍 Server will run on http://{config.API_HOST}:{config.API_PORT}")
    logger.info(f"📚 API docs available at http://{config.API_HOST}:{config.API_PORT}/docs")
    
    try:
        uvicorn.run(
            "backend.main:app",
            host=config.API_HOST,
            port=config.API_PORT,
            reload=True,
            log_level=config.LOG_LEVEL.lower(),
            access_log=True,
            reload_dirs=["backend", "config.py"]
        )
    except KeyboardInterrupt:
        logger.info("👋 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()