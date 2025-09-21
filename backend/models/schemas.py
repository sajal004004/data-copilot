"""
Pydantic Models for the Data Copilot MVP
Type-safe models for requests, responses, and agent state management
"""
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum

class QueryType(str, Enum):
    """Types of SQL queries supported"""
    SELECT = "select"
    ANALYSIS = "analysis"
    AGGREGATION = "aggregation"
    TREND = "trend"
    COMPARISON = "comparison"

class AgentStatus(str, Enum):
    """Status of individual agents in the workflow"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class WorkflowStatus(str, Enum):
    """Overall workflow status"""
    INITIALIZED = "initialized"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class SQLQuery(BaseModel):
    """SQL Query model with metadata"""
    query: str = Field(..., description="The SQL query text")
    query_type: QueryType = Field(default=QueryType.SELECT, description="Type of query")
    estimated_rows: Optional[int] = Field(None, description="Estimated result rows")
    complexity_score: Optional[float] = Field(None, description="Query complexity (0-1)")
    execution_plan: Optional[str] = Field(None, description="Query execution plan")

class QueryRequest(BaseModel):
    """Request model for natural language to SQL conversion"""
    natural_language_query: str = Field(..., min_length=5, description="Natural language question")
    context_preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences for context")
    include_explanation: bool = Field(default=True, description="Include business explanation")
    max_results: int = Field(default=100, description="Maximum number of results")
    
    @validator('natural_language_query')
    def validate_query(cls, v):
        if len(v.strip()) < 5:
            raise ValueError('Query must be at least 5 characters long')
        return v.strip()

class ContextItem(BaseModel):
    """Individual context item from vector search"""
    content: str = Field(..., description="Context content")
    source: str = Field(..., description="Source of context (sql_examples, business_rules, schema)")
    similarity_score: float = Field(..., ge=0, le=1, description="Similarity score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class AgentResult(BaseModel):
    """Result from an individual agent"""
    agent_name: str = Field(..., description="Name of the agent")
    status: AgentStatus = Field(..., description="Agent execution status")
    result: Dict[str, Any] = Field(default_factory=dict, description="Agent result data")
    execution_time: float = Field(..., ge=0, description="Execution time in seconds")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    confidence_score: Optional[float] = Field(None, ge=0, le=1, description="Confidence in result")

class WorkflowState(BaseModel):
    """Complete state of the agentic workflow"""
    session_id: str = Field(..., description="Unique session identifier")
    original_query: str = Field(..., description="Original natural language query")
    current_step: str = Field(default="initialized", description="Current workflow step")
    status: WorkflowStatus = Field(default=WorkflowStatus.INITIALIZED, description="Overall workflow status")
    
    # Agent results
    context_retrieval: Optional[AgentResult] = None
    sql_generation: Optional[AgentResult] = None
    validation: Optional[AgentResult] = None
    explanation: Optional[AgentResult] = None
    execution: Optional[AgentResult] = None
    synthesis: Optional[AgentResult] = None
    
    # Workflow data
    retrieved_context: List[ContextItem] = Field(default_factory=list)
    generated_sql: Optional[SQLQuery] = None
    validation_results: Dict[str, Any] = Field(default_factory=dict)
    query_results: Optional[Dict[str, Any]] = None
    business_explanation: Optional[str] = None
    final_insights: Optional[Dict[str, Any]] = None
    
    # Metrics
    total_execution_time: float = Field(default=0.0, description="Total workflow execution time")
    context_utilization_score: float = Field(default=0.0, description="How well context was utilized")
    business_relevance_score: float = Field(default=0.0, description="Business relevance of results")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

class QueryResponse(BaseModel):
    """Complete response for a natural language query"""
    session_id: str = Field(..., description="Unique session identifier")
    success: bool = Field(..., description="Whether the query was successful")
    
    # Core results
    sql_query: Optional[str] = Field(None, description="Generated SQL query")
    results: Optional[Dict[str, Any]] = Field(None, description="Query execution results")
    business_explanation: Optional[str] = Field(None, description="Business-focused explanation")
    insights: Optional[Dict[str, Any]] = Field(None, description="AI-generated insights")
    
    # Agent status
    agent_statuses: Dict[str, AgentStatus] = Field(default_factory=dict, description="Status of each agent")
    workflow_status: WorkflowStatus = Field(..., description="Overall workflow status")
    
    # Metrics and metadata
    execution_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    context_used: List[ContextItem] = Field(default_factory=list, description="Context items utilized")
    confidence_score: float = Field(default=0.0, ge=0, le=1, description="Overall confidence")
    
    # Error handling
    error_message: Optional[str] = Field(None, description="Error message if failed")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    
    # Timestamps
    execution_time: float = Field(default=0.0, description="Total execution time")
    timestamp: datetime = Field(default_factory=datetime.now)

class ValidationResult(BaseModel):
    """SQL query validation results"""
    is_valid: bool = Field(..., description="Whether the query is valid")
    is_safe: bool = Field(..., description="Whether the query is safe to execute")
    syntax_errors: List[str] = Field(default_factory=list, description="Syntax error messages")
    security_issues: List[str] = Field(default_factory=list, description="Security issues found")
    performance_warnings: List[str] = Field(default_factory=list, description="Performance warnings")
    estimated_cost: Optional[float] = Field(None, description="Estimated query cost")
    recommended_optimizations: List[str] = Field(default_factory=list, description="Optimization suggestions")

class BusinessInsight(BaseModel):
    """Business insight generated from query results"""
    insight_type: str = Field(..., description="Type of insight (trend, anomaly, recommendation)")
    title: str = Field(..., description="Brief insight title")
    description: str = Field(..., description="Detailed insight description")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in the insight")
    actionable_recommendations: List[str] = Field(default_factory=list, description="Actionable recommendations")
    supporting_data: Dict[str, Any] = Field(default_factory=dict, description="Data supporting the insight")

class SystemMetrics(BaseModel):
    """System-wide performance metrics"""
    total_queries_processed: int = Field(default=0, description="Total queries processed")
    average_execution_time: float = Field(default=0.0, description="Average execution time")
    success_rate: float = Field(default=0.0, ge=0, le=1, description="Query success rate")
    context_hit_rate: float = Field(default=0.0, ge=0, le=1, description="Context utilization rate")
    p95_latency: float = Field(default=0.0, description="95th percentile latency")
    error_rate: float = Field(default=0.0, ge=0, le=1, description="Error rate")
    
    # Agent-specific metrics
    agent_performance: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Per-agent performance")
    
    # Business impact metrics
    time_saved_minutes: float = Field(default=0.0, description="Estimated time saved in minutes")
    cost_per_query: float = Field(default=0.0, description="Average cost per query")
    business_value_score: float = Field(default=0.0, ge=0, le=1, description="Business value delivered")
    business_impact: Dict[str, float] = Field(default_factory=dict, description="Grouped business impact metrics")

class AgentConfig(BaseModel):
    """Configuration for individual agents"""
    agent_name: str = Field(..., description="Name of the agent")
    enabled: bool = Field(default=True, description="Whether the agent is enabled")
    timeout_seconds: float = Field(default=30.0, description="Agent timeout in seconds")
    max_retries: int = Field(default=2, description="Maximum retry attempts")
    configuration: Dict[str, Any] = Field(default_factory=dict, description="Agent-specific configuration")

class WorkflowConfig(BaseModel):
    """Configuration for the entire workflow"""
    agents: List[AgentConfig] = Field(..., description="Configuration for each agent")
    parallel_execution: bool = Field(default=False, description="Enable parallel agent execution where possible")
    fail_fast: bool = Field(default=False, description="Stop workflow on first agent failure")
    enable_caching: bool = Field(default=True, description="Enable result caching")
    max_workflow_time: float = Field(default=120.0, description="Maximum workflow execution time")

# Example data models for testing and demos
class DemoScenario(BaseModel):
    """Demo scenario for showcasing system capabilities"""
    scenario_id: str = Field(..., description="Unique scenario identifier")
    title: str = Field(..., description="Scenario title")
    description: str = Field(..., description="Scenario description")
    natural_language_query: str = Field(..., description="Example query")
    expected_sql_pattern: Optional[str] = Field(None, description="Expected SQL pattern")
    expected_insights: List[str] = Field(default_factory=list, description="Expected business insights")
    complexity_level: str = Field(default="medium", description="Complexity level (low, medium, high)")
    business_value: str = Field(..., description="Business value demonstration")