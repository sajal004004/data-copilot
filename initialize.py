"""
System Initialization Script for Data Copilot MVP
Sets up database, loads data, initializes vector store, and prepares demo environment
"""
import asyncio
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from backend.database.manager import db_manager
from backend.database.vector_store import vector_store
from backend.services.demo import demo_service
from backend.services.metrics import metrics_service
from config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def initialize_system():
    """Complete system initialization"""
    
    logger.info("🚀 Starting Data Copilot MVP initialization...")
    
    try:
        # 1. Validate configuration
        logger.info("📋 Validating configuration...")
        config_validation = config.validate_config()
        
        if not config_validation["valid"]:
            logger.error("❌ Configuration validation failed:")
            for issue in config_validation["issues"]:
                logger.error(f"  - {issue}")
            return False
        
        logger.info("✅ Configuration validated")
        
        # 2. Initialize database
        logger.info("🗄️ Initializing database...")
        db_init_result = db_manager.initialize_database()
        
        if not db_init_result["success"]:
            logger.error(f"❌ Database initialization failed: {db_init_result.get('error')}")
            return False
        
        logger.info(f"✅ Database initialized in {db_init_result['execution_time']:.2f}s")
        
        # 3. Load Superstore data
        logger.info("📊 Loading Superstore dataset...")
        csv_path = Path.cwd() / "Superstore.csv"
        
        if not csv_path.exists():
            logger.error(f"❌ Superstore.csv not found at {csv_path}")
            logger.info("Please ensure Superstore.csv is in the project root directory")
            return False
        
        ingest_result = db_manager.ingest_superstore_data(str(csv_path))
        
        if not ingest_result["success"]:
            logger.error(f"❌ Data ingestion failed: {ingest_result.get('error')}")
            return False
        
        logger.info(f"✅ Loaded {ingest_result['rows_inserted']} rows in {ingest_result['execution_time']:.2f}s")
        
        # 4. Initialize vector store with schema context
        logger.info("🔍 Setting up vector store...")
        schema_info = db_manager.get_schema_info()
        
        if schema_info.get("error"):
            logger.error(f"❌ Failed to get schema info: {schema_info['error']}")
            return False
        
        vector_store.add_schema_context(schema_info)
        logger.info("✅ Vector store initialized with schema context")
        
        # 5. Initialize demo scenarios and curated examples
        logger.info("🎭 Setting up demo scenarios and SQL examples...")
        await demo_service.initialize_demo_scenarios()
        logger.info("✅ Demo scenarios and curated examples loaded")
        
        # 6. Verify system readiness
        logger.info("🔧 Verifying system readiness...")
        
        # Test database connectivity
        test_query_result = db_manager.execute_safe_query("SELECT COUNT(*) as total_rows FROM superstore")
        if not test_query_result["success"]:
            logger.error("❌ Database connectivity test failed")
            return False
        
        total_rows = test_query_result["data"][0]["total_rows"]
        logger.info(f"✅ Database connectivity verified - {total_rows} rows available")
        
        # Test vector store
        vector_metrics = vector_store.get_metrics()
        total_embeddings = sum(vector_metrics["collection_counts"].values())
        logger.info(f"✅ Vector store verified - {total_embeddings} embeddings available")
        
        # 7. Display system summary
        logger.info("📊 System initialization complete!")
        logger.info("=" * 60)
        logger.info("🎯 DATA COPILOT MVP - READY FOR PRODUCTION")
        logger.info("=" * 60)
        logger.info(f"📁 Database: {config.DATABASE_PATH}")
        logger.info(f"🗂️ Vector Store: {config.VECTOR_STORE_DIR}")
        logger.info(f"📊 Data Rows: {total_rows:,}")
        logger.info(f"🧠 Embeddings: {total_embeddings:,}")
        logger.info(f"🎭 Demo Scenarios: {len(demo_service.demo_scenarios)}")
        logger.info(f"🌐 API Endpoint: http://{config.API_HOST}:{config.API_PORT}")
        logger.info(f"💻 Frontend: http://{config.API_HOST}:{config.STREAMLIT_PORT}")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ System initialization failed: {e}")
        return False

def display_startup_instructions():
    """Display instructions for starting the system"""
    
    print("\n" + "="*80)
    print("🚀 DATA COPILOT MVP - STARTUP INSTRUCTIONS")
    print("="*80)
    
    print("\n📋 PREREQUISITES:")
    print("1. Install Python 3.11+")
    print("2. Install required packages: pip install -r requirements.txt")
    print("3. Set up environment variables: cp .env.example .env")
    print("4. Add your OpenAI API key to the .env file")
    print("5. Ensure Superstore.csv is in the project root")
    
    print("\n🚀 STARTUP SEQUENCE:")
    print("1. Initialize system: python initialize.py")
    print("2. Start backend API: python -m uvicorn backend.main:app --host localhost --port 8000 --reload")
    print("3. Start frontend: streamlit run frontend/app.py --server.port 8501")
    
    print("\n🎯 ACCESS POINTS:")
    print(f"• Frontend (Streamlit): http://localhost:8501")
    print(f"• Backend API: http://localhost:8000")
    print(f"• API Documentation: http://localhost:8000/docs")
    print(f"• Health Check: http://localhost:8000/health")
    
    print("\n📊 DEMO SCENARIOS:")
    print("• Navigate to the 'Demo Scenarios' tab in the frontend")
    print("• Try sample queries like 'Show me sales trends by category'")
    print("• Monitor real-time agent execution in the workflow status")
    
    print("\n⚡ PERFORMANCE TARGETS:")
    print("• Query execution: <3 seconds")
    print("• Success rate: >95%")
    print("• Context utilization: >85%")
    
    print("\n🔧 TROUBLESHOOTING:")
    print("• Check logs for detailed error messages")
    print("• Verify all services are running on correct ports")
    print("• Ensure OpenAI API key is valid and has sufficient credits")
    
    print("="*80 + "\n")

async def main():
    """Main initialization function"""
    
    # Display instructions first
    display_startup_instructions()
    
    # Ask user if they want to proceed with initialization
    print("🔄 Ready to initialize the Data Copilot MVP system?")
    response = input("Press Enter to continue or 'q' to quit: ").strip().lower()
    
    if response in ['q', 'quit', 'exit']:
        print("👋 Initialization cancelled")
        return
    
    # Run initialization
    success = await initialize_system()
    
    if success:
        print("\n🎉 SUCCESS! Data Copilot MVP is ready to use.")
        print("\nNext steps:")
        print("1. Start the backend: python -m uvicorn backend.main:app --host localhost --port 8000 --reload")
        print("2. Start the frontend: streamlit run frontend/app.py --server.port 8501")
        print("3. Open http://localhost:8501 in your browser")
    else:
        print("\n❌ Initialization failed. Please check the logs above for details.")
        return False

if __name__ == "__main__":
    asyncio.run(main())