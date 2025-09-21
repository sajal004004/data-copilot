"""
LangGraph Agentic Workflow for Data Copilot MVP
6-agent orchestrated system for intelligent SQL generation with business context
"""
import asyncio
import logging
import time
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from backend.models.schemas import (
    WorkflowState, AgentResult, AgentStatus, WorkflowStatus, 
    ContextItem, SQLQuery, ValidationResult, BusinessInsight
)
from backend.database.manager import db_manager
from backend.database.vector_store import vector_store
from config import config

logger = logging.getLogger(__name__)

class AgenticWorkflow:
    """LangGraph-based agentic workflow for intelligent SQL generation"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            api_key=config.OPENAI_API_KEY,
            temperature=0.1
        )
        self.graph = self._build_workflow_graph()
        self.metrics = {
            "workflows_executed": 0,
            "successful_workflows": 0,
            "agent_execution_times": {},
            "context_utilization": []
        }
    
    def _build_workflow_graph(self) -> StateGraph:
        """Build the LangGraph workflow with 6 agents"""
        workflow = StateGraph(WorkflowState)
        
        # Add agent nodes
        workflow.add_node("context_retrieval", self._context_retrieval_agent)
        workflow.add_node("sql_generation", self._sql_generation_agent)
        workflow.add_node("validation", self._validation_agent)
        workflow.add_node("explanation", self._explanation_agent)
        workflow.add_node("execution", self._execution_agent)
        workflow.add_node("synthesis", self._synthesis_agent)
        
        # Define workflow edges (sequential with conditional branching)
        workflow.add_edge("context_retrieval", "sql_generation")
        workflow.add_edge("sql_generation", "validation")
        
        # Conditional edge after validation
        workflow.add_conditional_edges(
            "validation",
            self._should_proceed_after_validation,
            {
                "proceed": "explanation",
                "retry_sql": "sql_generation",
                "fail": END
            }
        )
        
        workflow.add_edge("explanation", "execution")
        
        # Conditional edge after execution
        workflow.add_conditional_edges(
            "execution",
            self._should_proceed_after_execution,
            {
                "proceed": "synthesis",
                "fail": END
            }
        )
        
        workflow.add_edge("synthesis", END)
        
        # Set entry point
        workflow.set_entry_point("context_retrieval")
        
        return workflow.compile()
    
    async def process_query(self, natural_query: str, session_id: Optional[str] = None) -> WorkflowState:
        """Process a natural language query through the complete agentic workflow"""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Initialize workflow state
        state = WorkflowState(
            session_id=session_id,
            original_query=natural_query,
            status=WorkflowStatus.RUNNING
        )
        
        start_time = time.time()
        
        try:
            # Execute the workflow
            logger.info(f"Starting agentic workflow for session {session_id}")
            
            # Convert state to dict for LangGraph compatibility
            state_dict = state.dict()
            result_state = await self._execute_workflow(state_dict)
            
            # Convert back to WorkflowState
            final_state = WorkflowState(**result_state)
            final_state.total_execution_time = time.time() - start_time
            final_state.completed_at = datetime.now()
            final_state.status = WorkflowStatus.COMPLETED
            
            # Update metrics
            self.metrics["workflows_executed"] += 1
            self.metrics["successful_workflows"] += 1
            
            logger.info(f"Workflow completed successfully in {final_state.total_execution_time:.2f}s")
            
            return final_state
            
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            state.status = WorkflowStatus.FAILED
            state.total_execution_time = time.time() - start_time
            return state
    
    async def _execute_workflow(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the workflow using LangGraph"""
        # This is a simplified version - in production, you'd use the actual LangGraph execution
        # For now, we'll execute agents sequentially
        
        # Context Retrieval Agent
        state = await self._context_retrieval_agent(state)
        if state.get("status") == WorkflowStatus.FAILED:
            return state
        
        # SQL Generation Agent
        state = await self._sql_generation_agent(state)
        if state.get("status") == WorkflowStatus.FAILED:
            return state
        
        # Validation Agent
        state = await self._validation_agent(state)
        if state.get("status") == WorkflowStatus.FAILED:
            return state
        
        # Explanation Agent
        state = await self._explanation_agent(state)
        if state.get("status") == WorkflowStatus.FAILED:
            return state
        
        # Execution Agent
        state = await self._execution_agent(state)
        if state.get("status") == WorkflowStatus.FAILED:
            return state
        
        # Synthesis Agent
        state = await self._synthesis_agent(state)
        
        return state
    
    async def _context_retrieval_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Agent 1: Retrieve relevant context for the query"""
        start_time = time.time()
        
        try:
            logger.info("Context Retrieval Agent: Starting")
            
            query = state["original_query"]
            
            # Get comprehensive context from vector store
            context = vector_store.get_comprehensive_context(query)
            
            # Convert context to ContextItem objects
            context_items = []
            
            # Add SQL examples
            for example in context["sql_examples"]:
                context_items.append(ContextItem(
                    content=f"Q: {example['business_question']}\nSQL: {example['sql_query']}",
                    source="sql_examples",
                    similarity_score=example["similarity_score"],
                    metadata=example
                ))
            
            # Add business context
            for rule in context["business_context"]:
                context_items.append(ContextItem(
                    content=f"Rule: {rule['rule']}\nDescription: {rule['description']}",
                    source="business_rules",
                    similarity_score=rule["similarity_score"],
                    metadata=rule
                ))
            
            # Add schema context
            for schema_item in context["schema_context"]:
                context_items.append(ContextItem(
                    content=f"Schema: {schema_item['content']}",
                    source="schema",
                    similarity_score=schema_item["similarity_score"],
                    metadata=schema_item
                ))
            
            # Calculate context utilization score
            avg_similarity = sum(item.similarity_score for item in context_items) / max(len(context_items), 1)
            
            execution_time = time.time() - start_time
            
            # Update state
            state["retrieved_context"] = [item.dict() for item in context_items]
            state["context_utilization_score"] = avg_similarity
            state["context_retrieval"] = AgentResult(
                agent_name="context_retrieval",
                status=AgentStatus.COMPLETED,
                result={
                    "context_items_count": len(context_items),
                    "avg_similarity": avg_similarity,
                    "context_quality": context["context_quality_score"]
                },
                execution_time=execution_time,
                confidence_score=avg_similarity
            ).dict()
            
            logger.info(f"Context Retrieval Agent: Retrieved {len(context_items)} context items")
            
            return state
            
        except Exception as e:
            logger.error(f"Context Retrieval Agent failed: {e}")
            state["context_retrieval"] = AgentResult(
                agent_name="context_retrieval",
                status=AgentStatus.FAILED,
                result={},
                execution_time=time.time() - start_time,
                error_message=str(e)
            ).dict()
            state["status"] = WorkflowStatus.FAILED
            return state
    
    async def _sql_generation_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Agent 2: Generate SQL query using context and business intelligence"""
        start_time = time.time()
        
        try:
            logger.info("SQL Generation Agent: Starting")
            
            query = state["original_query"]
            context_items = state.get("retrieved_context", [])
            
            # Get database schema for reference
            schema_info = db_manager.get_schema_info()
            
            # Build context prompt
            context_prompt = self._build_sql_generation_prompt(query, context_items, schema_info)
            
            # Generate SQL using LLM
            messages = [
                SystemMessage(content=context_prompt),
                HumanMessage(content=f"Generate SQL for: {query}")
            ]
            
            response = await self.llm.ainvoke(messages)
            generated_sql = self._extract_sql_from_response(response.content)
            
            # Estimate query complexity
            complexity_score = self._calculate_query_complexity(generated_sql)
            
            # Create SQL query object
            sql_query = SQLQuery(
                query=generated_sql,
                complexity_score=complexity_score,
                estimated_rows=None  # Will be estimated during validation
            )
            
            execution_time = time.time() - start_time
            
            # Update state
            state["generated_sql"] = sql_query.dict()
            state["sql_generation"] = AgentResult(
                agent_name="sql_generation",
                status=AgentStatus.COMPLETED,
                result={
                    "sql_query": generated_sql,
                    "complexity_score": complexity_score,
                    "context_items_used": len(context_items)
                },
                execution_time=execution_time,
                confidence_score=min(complexity_score, state.get("context_utilization_score", 0.5))
            ).dict()
            
            logger.info(f"SQL Generation Agent: Generated SQL with complexity {complexity_score:.2f}")
            
            return state
            
        except Exception as e:
            logger.error(f"SQL Generation Agent failed: {e}")
            state["sql_generation"] = AgentResult(
                agent_name="sql_generation",
                status=AgentStatus.FAILED,
                result={},
                execution_time=time.time() - start_time,
                error_message=str(e)
            ).dict()
            state["status"] = WorkflowStatus.FAILED
            return state
    
    async def _validation_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Agent 3: Validate SQL query for safety, syntax, and performance"""
        start_time = time.time()
        
        try:
            logger.info("Validation Agent: Starting")
            
            sql_query = state["generated_sql"]["query"]
            
            # Perform comprehensive validation
            validation_result = self._comprehensive_sql_validation(sql_query)
            
            execution_time = time.time() - start_time
            
            # Update state
            state["validation_results"] = validation_result
            state["validation"] = AgentResult(
                agent_name="validation",
                status=AgentStatus.COMPLETED if validation_result["is_valid"] else AgentStatus.FAILED,
                result=validation_result,
                execution_time=execution_time,
                confidence_score=1.0 if validation_result["is_safe"] else 0.0
            ).dict()
            
            if not validation_result["is_valid"] or not validation_result["is_safe"]:
                state["status"] = WorkflowStatus.FAILED
            
            logger.info(f"Validation Agent: Query validation {'passed' if validation_result['is_valid'] else 'failed'}")
            
            return state
            
        except Exception as e:
            logger.error(f"Validation Agent failed: {e}")
            state["validation"] = AgentResult(
                agent_name="validation",
                status=AgentStatus.FAILED,
                result={},
                execution_time=time.time() - start_time,
                error_message=str(e)
            ).dict()
            state["status"] = WorkflowStatus.FAILED
            return state
    
    async def _explanation_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Agent 4: Generate business-focused explanation of the query and expected results"""
        start_time = time.time()
        
        try:
            logger.info("Explanation Agent: Starting")
            
            original_query = state["original_query"]
            sql_query = state["generated_sql"]["query"]
            context_items = state.get("retrieved_context", [])
            
            # Generate business explanation
            explanation_prompt = self._build_explanation_prompt(original_query, sql_query, context_items)
            
            messages = [
                SystemMessage(content=explanation_prompt),
                HumanMessage(content=f"Explain this query in business terms: {original_query}")
            ]
            
            response = await self.llm.ainvoke(messages)
            business_explanation = response.content
            
            execution_time = time.time() - start_time
            
            # Update state
            state["business_explanation"] = business_explanation
            state["explanation"] = AgentResult(
                agent_name="explanation",
                status=AgentStatus.COMPLETED,
                result={
                    "explanation": business_explanation,
                    "explanation_length": len(business_explanation)
                },
                execution_time=execution_time,
                confidence_score=0.9
            ).dict()
            
            logger.info("Explanation Agent: Generated business explanation")
            
            return state
            
        except Exception as e:
            logger.error(f"Explanation Agent failed: {e}")
            state["explanation"] = AgentResult(
                agent_name="explanation",
                status=AgentStatus.FAILED,
                result={},
                execution_time=time.time() - start_time,
                error_message=str(e)
            ).dict()
            return state
    
    async def _execution_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Agent 5: Execute the validated SQL query safely"""
        start_time = time.time()
        
        try:
            logger.info("Execution Agent: Starting")
            
            sql_query = state["generated_sql"]["query"]
            context = state.get("retrieved_context", [])
            
            # Execute query using database manager
            result = db_manager.execute_safe_query(sql_query, {"context": context})
            
            execution_time = time.time() - start_time
            
            # Update state
            state["query_results"] = result
            state["execution"] = AgentResult(
                agent_name="execution",
                status=AgentStatus.COMPLETED if result["success"] else AgentStatus.FAILED,
                result={
                    "success": result["success"],
                    "row_count": result.get("row_count", 0),
                    "execution_time": result.get("execution_time", 0)
                },
                execution_time=execution_time,
                confidence_score=1.0 if result["success"] else 0.0,
                error_message=result.get("error")
            ).dict()
            
            if not result["success"]:
                state["status"] = WorkflowStatus.FAILED
            
            logger.info(f"Execution Agent: Query {'executed successfully' if result['success'] else 'failed'}")
            
            return state
            
        except Exception as e:
            logger.error(f"Execution Agent failed: {e}")
            state["execution"] = AgentResult(
                agent_name="execution",
                status=AgentStatus.FAILED,
                result={},
                execution_time=time.time() - start_time,
                error_message=str(e)
            ).dict()
            state["status"] = WorkflowStatus.FAILED
            return state
    
    async def _synthesis_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Agent 6: Synthesize results with business insights and recommendations"""
        start_time = time.time()
        
        try:
            logger.info("Synthesis Agent: Starting")
            
            original_query = state["original_query"]
            results = state.get("query_results", {})
            explanation = state.get("business_explanation", "")
            
            # Generate insights and recommendations
            insights = await self._generate_business_insights(original_query, results, explanation)
            
            execution_time = time.time() - start_time
            
            # Calculate business relevance score
            business_relevance_score = self._calculate_business_relevance(insights, results)
            
            # Update state
            state["final_insights"] = insights
            state["business_relevance_score"] = business_relevance_score
            state["synthesis"] = AgentResult(
                agent_name="synthesis",
                status=AgentStatus.COMPLETED,
                result={
                    "insights_generated": len(insights.get("insights", [])),
                    "business_relevance_score": business_relevance_score,
                    "recommendations_count": len(insights.get("recommendations", []))
                },
                execution_time=execution_time,
                confidence_score=business_relevance_score
            ).dict()
            
            logger.info(f"Synthesis Agent: Generated {len(insights.get('insights', []))} business insights")
            
            return state
            
        except Exception as e:
            logger.error(f"Synthesis Agent failed: {e}")
            state["synthesis"] = AgentResult(
                agent_name="synthesis",
                status=AgentStatus.FAILED,
                result={},
                execution_time=time.time() - start_time,
                error_message=str(e)
            ).dict()
            return state
    
    def _should_proceed_after_validation(self, state: Dict[str, Any]) -> str:
        """Determine next step after validation"""
        validation_results = state.get("validation_results", {})
        
        if not validation_results.get("is_valid", False):
            return "fail"
        elif not validation_results.get("is_safe", False):
            return "fail"
        else:
            return "proceed"
    
    def _should_proceed_after_execution(self, state: Dict[str, Any]) -> str:
        """Determine next step after execution"""
        execution_result = state.get("query_results", {})
        
        if execution_result.get("success", False):
            return "proceed"
        else:
            return "fail"
    
    def _build_sql_generation_prompt(self, query: str, context_items: List[Dict], schema_info: Dict) -> str:
        """Build comprehensive prompt for SQL generation"""
        prompt = f"""
You are an expert SQL analyst for a retail superstore business. Generate accurate, safe SQL queries based on natural language questions.

DATABASE SCHEMA:
{schema_info}

BUSINESS CONTEXT AND EXAMPLES:
"""
        
        # Add relevant context
        for item in context_items[:5]:  # Limit to top 5 most relevant
            prompt += f"\n- {item['content'][:200]}..."
        
        prompt += """

REQUIREMENTS:
1. Generate only SELECT statements (no INSERT, UPDATE, DELETE)
2. Use proper SQL syntax for SQLite
3. Include relevant business logic and calculations
4. Add meaningful column aliases
5. Use appropriate aggregations and grouping
6. Consider performance implications

SAFETY RULES:
- No dynamic SQL or string concatenation
- No system tables or functions
- Limit results to reasonable numbers
- Use parameterized approaches where possible
"""
        
        return prompt
    
    def _build_explanation_prompt(self, original_query: str, sql_query: str, context_items: List[Dict]) -> str:
        """Build prompt for business explanation generation"""
        prompt = f"""
You are a business analyst explaining SQL queries in terms that business stakeholders can understand.

ORIGINAL QUESTION: {original_query}
GENERATED SQL: {sql_query}

BUSINESS CONTEXT:
"""
        
        for item in context_items[:3]:
            if item.get('source') == 'business_rules':
                prompt += f"\n- {item['content'][:150]}..."
        
        prompt += """

EXPLANATION REQUIREMENTS:
1. Explain WHAT the query does in business terms
2. Explain WHY this analysis is valuable
3. Describe the KEY METRICS and dimensions being analyzed
4. Highlight any BUSINESS INSIGHTS that can be derived
5. Suggest FOLLOW-UP QUESTIONS or actions
6. Keep language clear and non-technical
7. Focus on business value and actionable insights
"""
        
        return prompt
    
    def _extract_sql_from_response(self, response: str) -> str:
        """Extract SQL query from LLM response"""
        # Find SQL query in the response (look for SELECT statements)
        lines = response.strip().split('\n')
        sql_lines = []
        in_sql = False
        
        for line in lines:
            line = line.strip()
            if line.upper().startswith('SELECT') or line.upper().startswith('WITH'):
                in_sql = True
                sql_lines.append(line)
            elif in_sql and line and not line.startswith('#') and not line.startswith('--'):
                sql_lines.append(line)
            elif in_sql and (line.endswith(';') or not line):
                if line.endswith(';'):
                    sql_lines.append(line.rstrip(';'))
                break
        
        return ' '.join(sql_lines).strip()
    
    def _calculate_query_complexity(self, sql_query: str) -> float:
        """Calculate query complexity score (0-1)"""
        complexity_factors = {
            'JOIN': 0.2,
            'SUBQUERY': 0.3,
            'GROUP BY': 0.1,
            'ORDER BY': 0.05,
            'HAVING': 0.15,
            'WINDOW': 0.3,
            'CTE': 0.25
        }
        
        sql_upper = sql_query.upper()
        score = 0.1  # Base complexity
        
        for factor, weight in complexity_factors.items():
            if factor in sql_upper:
                score += weight
        
        return min(score, 1.0)
    
    def _comprehensive_sql_validation(self, sql_query: str) -> Dict[str, Any]:
        """Perform comprehensive SQL validation"""
        # Use the database manager's validation
        safety_check = db_manager._validate_query_safety(sql_query)
        
        return {
            "is_valid": safety_check["safe"],
            "is_safe": safety_check["safe"],
            "syntax_errors": [] if safety_check["safe"] else [safety_check["reason"]],
            "security_issues": [] if safety_check["safe"] else [safety_check["reason"]],
            "performance_warnings": [],
            "estimated_cost": 0.1,  # Placeholder
            "recommended_optimizations": []
        }
    
    async def _generate_business_insights(self, query: str, results: Dict, explanation: str) -> Dict[str, Any]:
        """Generate business insights from query results"""
        if not results.get("success") or not results.get("data"):
            return {"insights": [], "recommendations": []}
        
        # Analyze results for patterns and insights
        data = results["data"]
        
        insights = {
            "insights": [],
            "recommendations": [],
            "key_findings": [],
            "next_steps": []
        }
        
        # Add basic insights based on data
        if len(data) > 0:
            insights["key_findings"].append(f"Analysis returned {len(data)} records")
            
            # Look for numerical trends if applicable
            numeric_columns = [col for col in results.get("columns", []) if any(isinstance(row.get(col), (int, float)) for row in data)]
            
            for col in numeric_columns[:3]:  # Limit to first 3 numeric columns
                values = [row.get(col, 0) for row in data if isinstance(row.get(col), (int, float))]
                if values:
                    avg_val = sum(values) / len(values)
                    max_val = max(values)
                    min_val = min(values)
                    
                    insights["insights"].append({
                        "type": "metric_summary",
                        "title": f"{col.title()} Analysis",
                        "description": f"Average: {avg_val:.2f}, Range: {min_val:.2f} to {max_val:.2f}",
                        "confidence": 0.9
                    })
        
        # Add recommendations
        insights["recommendations"].extend([
            "Consider drilling down into specific categories for deeper insights",
            "Monitor trends over time to identify patterns",
            "Compare results across different dimensions"
        ])
        
        return insights
    
    def _calculate_business_relevance(self, insights: Dict, results: Dict) -> float:
        """Calculate business relevance score"""
        score = 0.5  # Base score
        
        if results.get("success"):
            score += 0.2
        
        if insights.get("insights"):
            score += 0.2
        
        if insights.get("recommendations"):
            score += 0.1
        
        return min(score, 1.0)
    
    def get_workflow_metrics(self) -> Dict[str, Any]:
        """Get workflow performance metrics"""
        return {
            "total_workflows": self.metrics["workflows_executed"],
            "success_rate": self.metrics["successful_workflows"] / max(self.metrics["workflows_executed"], 1),
            "agent_metrics": self.metrics["agent_execution_times"],
            "average_context_utilization": sum(self.metrics["context_utilization"]) / max(len(self.metrics["context_utilization"]), 1)
        }

# Global workflow instance
agentic_workflow = AgenticWorkflow()