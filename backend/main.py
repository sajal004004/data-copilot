"""
FastAPI Backend for Data Copilot MVP
Production-ready REST API with comprehensive error handling and agent orchestration
"""
import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
from pydantic import BaseModel

from backend.models.schemas import (
    QueryRequest, QueryResponse, WorkflowState, SystemMetrics, 
    DemoScenario, AgentStatus, WorkflowStatus
)
from backend.agents.workflow import agentic_workflow
from backend.database.manager import db_manager
from backend.database.vector_store import vector_store
from backend.services.metrics import metrics_service
from backend.services.demo import demo_service
from config import config

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# Background tasks storage
active_workflows: Dict[str, WorkflowState] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    # Startup
    logger.info("Starting Data Copilot MVP...")
    
    # Initialize database
    db_init_result = db_manager.initialize_database()
    if not db_init_result["success"]:
        logger.error(f"Database initialization failed: {db_init_result['error']}")
        raise RuntimeError("Database initialization failed")
    
    # Load Superstore data
    try:
        import os
        csv_path = os.path.join(os.getcwd(), "Superstore.csv")
        if os.path.exists(csv_path):
            ingest_result = db_manager.ingest_superstore_data(csv_path)
            if ingest_result["success"]:
                logger.info(f"Superstore data loaded: {ingest_result['rows_inserted']} rows")
            else:
                logger.warning(f"Data ingestion failed: {ingest_result['error']}")
        else:
            logger.warning("Superstore.csv not found in current directory")
    except Exception as e:
        logger.warning(f"Could not load Superstore data: {e}")
    
    # Initialize vector store with schema context
    try:
        schema_info = db_manager.get_schema_info()
        vector_store.add_schema_context(schema_info)
        logger.info("Vector store initialized with schema context")
    except Exception as e:
        logger.warning(f"Vector store initialization warning: {e}")
    
    # Initialize demo scenarios
    await demo_service.initialize_demo_scenarios()
    
    logger.info("Data Copilot MVP started successfully!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Data Copilot MVP...")

# Create FastAPI app
app = FastAPI(
    title="Data Copilot MVP",
    description="Agentic SQL generation system with business intelligence",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],  # Streamlit frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "database_status": "connected",
        "vector_store_status": "ready"
    }

# Configuration validation endpoint
@app.get("/config/validate")
async def validate_configuration():
    """Validate system configuration"""
    validation_result = config.validate_config()
    return {
        "valid": validation_result["valid"],
        "issues": validation_result["issues"],
        "summary": validation_result["config_summary"]
    }

# Main query processing endpoint
@app.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks
) -> QueryResponse:
    """Process natural language query through agentic workflow"""
    start_time = time.time()
    
    try:
        logger.info(f"Processing query: {request.natural_language_query[:100]}...")
        
        # Execute agentic workflow
        workflow_state = await agentic_workflow.process_query(
            request.natural_language_query
        )
        
        # Store workflow for tracking
        active_workflows[workflow_state.session_id] = workflow_state
        
        # Build response
        response = QueryResponse(
            session_id=workflow_state.session_id,
            success=workflow_state.status == WorkflowStatus.COMPLETED,
            sql_query=workflow_state.generated_sql.query if workflow_state.generated_sql else None,
            results=workflow_state.query_results,
            business_explanation=workflow_state.business_explanation,
            insights=workflow_state.final_insights,
            agent_statuses={
                "context_retrieval": AgentStatus.COMPLETED if workflow_state.context_retrieval else AgentStatus.FAILED,
                "sql_generation": AgentStatus.COMPLETED if workflow_state.sql_generation else AgentStatus.FAILED,
                "validation": AgentStatus.COMPLETED if workflow_state.validation else AgentStatus.FAILED,
                "explanation": AgentStatus.COMPLETED if workflow_state.explanation else AgentStatus.FAILED,
                "execution": AgentStatus.COMPLETED if workflow_state.execution else AgentStatus.FAILED,
                "synthesis": AgentStatus.COMPLETED if workflow_state.synthesis else AgentStatus.FAILED,
            },
            workflow_status=workflow_state.status,
            execution_metrics={
                "total_execution_time": workflow_state.total_execution_time,
                "context_utilization_score": workflow_state.context_utilization_score,
                "business_relevance_score": workflow_state.business_relevance_score,
            },
            context_used=[
                {
                    "content": item.content,
                    "source": item.source,
                    "similarity_score": item.similarity_score,
                    "metadata": item.metadata
                }
                for item in workflow_state.retrieved_context
            ],
            confidence_score=workflow_state.business_relevance_score,
            execution_time=time.time() - start_time
        )
        
        # Record metrics in background
        background_tasks.add_task(
            metrics_service.record_query_metrics,
            request.natural_language_query,
            response,
            workflow_state
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        
        return QueryResponse(
            session_id="error",
            success=False,
            workflow_status=WorkflowStatus.FAILED,
            error_message=str(e),
            execution_time=time.time() - start_time
        )

# Workflow status endpoint
@app.get("/workflow/{session_id}")
async def get_workflow_status(session_id: str):
    """Get status of a specific workflow"""
    if session_id not in active_workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    workflow = active_workflows[session_id]
    return {
        "session_id": session_id,
        "status": workflow.status,
        "current_step": workflow.current_step,
        "execution_time": workflow.total_execution_time,
        "agent_statuses": {
            "context_retrieval": workflow.context_retrieval.get("status") if workflow.context_retrieval else "pending",
            "sql_generation": workflow.sql_generation.get("status") if workflow.sql_generation else "pending",
            "validation": workflow.validation.get("status") if workflow.validation else "pending",
            "explanation": workflow.explanation.get("status") if workflow.explanation else "pending",
            "execution": workflow.execution.get("status") if workflow.execution else "pending",
            "synthesis": workflow.synthesis.get("status") if workflow.synthesis else "pending",
        }
    }

# Database schema endpoint
@app.get("/schema")
async def get_database_schema():
    """Get database schema information"""
    try:
        schema_info = db_manager.get_schema_info()
        return schema_info
    except Exception as e:
        logger.error(f"Failed to get schema: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Data summary endpoint
@app.get("/data/summary")
async def get_data_summary():
    """Get comprehensive data summary for business context"""
    try:
        summary = db_manager.get_data_summary()
        return summary
    except Exception as e:
        logger.error(f"Failed to get data summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# System metrics endpoint
@app.get("/metrics", response_model=SystemMetrics)
async def get_system_metrics():
    """Get comprehensive system performance metrics"""
    try:
        # Combine metrics from different services
        db_metrics = db_manager.get_performance_metrics()
        workflow_metrics = agentic_workflow.get_workflow_metrics()
        vector_metrics = vector_store.get_metrics()
        system_metrics = metrics_service.get_system_metrics()
        
        return SystemMetrics(
            total_queries_processed=system_metrics.get("total_queries", 0),
            average_execution_time=db_metrics.get("average_execution_time", 0),
            success_rate=workflow_metrics.get("success_rate", 0),
            context_hit_rate=workflow_metrics.get("average_context_utilization", 0),
            p95_latency=db_metrics.get("p95_execution_time", 0),
            error_rate=db_metrics.get("error_rate", 0),
            agent_performance={
                "database": {
                    "avg_execution_time": db_metrics.get("average_execution_time", 0),
                    "cache_hit_rate": db_metrics.get("cache_hit_rate", 0),
                    "error_rate": db_metrics.get("error_rate", 0)
                },
                "vector_store": {
                    "total_searches": vector_metrics.get("total_searches", 0),
                    "context_retrievals": vector_metrics.get("context_retrievals", 0)
                }
            },
            time_saved_minutes=system_metrics.get("time_saved_minutes", 0),
            cost_per_query=system_metrics.get("cost_per_query", 0.05),
            business_value_score=system_metrics.get("business_value_score", 0.8),
            business_impact={
                "total_time_saved_minutes": system_metrics.get("time_saved_minutes", 0),
                "total_cost_savings": system_metrics.get("cost_savings", 0),
                "avg_value_per_query": system_metrics.get("business_value_score", 0.8)
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# SQL validation endpoint
@app.post("/validate/sql")
async def validate_sql(sql_query: str):
    """Validate SQL query for safety and syntax"""
    try:
        validation_result = db_manager._validate_query_safety(sql_query)
        return {
            "valid": validation_result["safe"],
            "reason": validation_result["reason"],
            "safe": validation_result["safe"]
        }
    except Exception as e:
        logger.error(f"SQL validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Context search endpoint
@app.get("/context/search")
async def search_context(
    query: str = Query(..., description="Search query"),
    source: Optional[str] = Query(None, description="Context source filter"),
    limit: int = Query(5, description="Maximum results")
):
    """Search for relevant context"""
    try:
        if source == "sql_examples":
            results = vector_store.search_sql_examples(query, limit)
        elif source == "business_context":
            results = vector_store.search_business_context(query, limit)
        elif source == "schema":
            results = vector_store.search_schema_context(query, limit)
        else:
            results = vector_store.get_comprehensive_context(query)
        
        return {
            "query": query,
            "source": source,
            "results": results,
            "count": len(results) if isinstance(results, list) else len(results.get("sql_examples", []))
        }
        
    except Exception as e:
        logger.error(f"Context search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Demo scenarios endpoint
@app.get("/demo/scenarios")
async def get_demo_scenarios():
    """Get available demo scenarios"""
    try:
        scenarios = await demo_service.get_demo_scenarios()
        return {"scenarios": scenarios}
    except Exception as e:
        logger.error(f"Failed to get demo scenarios: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/demo/scenario/{scenario_id}")
async def execute_demo_scenario(scenario_id: str):
    """Execute a specific demo scenario"""
    try:
        result = await demo_service.execute_demo_scenario(scenario_id)
        return result
    except Exception as e:
        logger.error(f"Demo scenario execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Performance testing endpoint
@app.post("/test/performance")
async def run_performance_test(
    num_queries: int = Query(10, description="Number of test queries"),
    concurrent: bool = Query(False, description="Run queries concurrently")
):
    """Run performance testing with multiple queries"""
    try:
        test_queries = [
            "Show me total sales by category",
            "Which customers have the highest profit margins?",
            "What are the sales trends over time?",
            "Show me the top performing products",
            "Which regions have declining sales?"
        ]
        
        results = []
        start_time = time.time()
        
        if concurrent:
            # Run queries concurrently
            tasks = []
            for i in range(num_queries):
                query = test_queries[i % len(test_queries)]
                task = agentic_workflow.process_query(f"{query} (test {i+1})")
                tasks.append(task)
            
            workflow_results = await asyncio.gather(*tasks)
            results = [{"success": ws.status == WorkflowStatus.COMPLETED, "time": ws.total_execution_time} for ws in workflow_results]
        else:
            # Run queries sequentially
            for i in range(num_queries):
                query = test_queries[i % len(test_queries)]
                workflow_state = await agentic_workflow.process_query(f"{query} (test {i+1})")
                results.append({
                    "success": workflow_state.status == WorkflowStatus.COMPLETED,
                    "time": workflow_state.total_execution_time
                })
        
        total_time = time.time() - start_time
        successful_queries = sum(1 for r in results if r["success"])
        avg_time = sum(r["time"] for r in results) / len(results)
        
        return {
            "test_summary": {
                "total_queries": num_queries,
                "successful_queries": successful_queries,
                "success_rate": successful_queries / num_queries,
                "total_time": total_time,
                "average_query_time": avg_time,
                "concurrent": concurrent
            },
            "detailed_results": results
        }
        
    except Exception as e:
        logger.error(f"Performance test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# System reset endpoint (for development)
@app.post("/system/reset")
async def reset_system():
    """Reset system state (development only)"""
    try:
        global active_workflows
        active_workflows.clear()
        
        # Reset metrics
        await metrics_service.reset_metrics()
        
        return {"message": "System reset successfully"}
        
    except Exception as e:
        logger.error(f"System reset failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Export data endpoint
@app.get("/export/workflows")
async def export_workflows():
    """Export workflow data for analysis"""
    try:
        return {
            "active_workflows": len(active_workflows),
            "workflows": [
                {
                    "session_id": session_id,
                    "status": workflow.status,
                    "execution_time": workflow.total_execution_time,
                    "created_at": workflow.created_at.isoformat()
                }
                for session_id, workflow in active_workflows.items()
            ]
        }
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
        log_level=config.LOG_LEVEL.lower()
    )