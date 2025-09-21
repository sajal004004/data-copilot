"""
Metrics Service for Data Copilot MVP
Comprehensive tracking and analysis of system performance and business impact
"""
import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque

from backend.models.schemas import QueryResponse, WorkflowState, SystemMetrics
from config import config

logger = logging.getLogger(__name__)

class MetricsService:
    """Production-ready metrics service for performance tracking and business impact measurement"""
    
    def __init__(self):
        self.metrics_storage = {
            "queries": deque(maxlen=10000),  # Store last 10k queries
            "performance": deque(maxlen=1000),  # Store last 1k performance measurements
            "agent_metrics": defaultdict(list),
            "business_impact": deque(maxlen=1000),
            "errors": deque(maxlen=1000)
        }
        
        self.real_time_metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_execution_time": 0.0,
            "context_hits": 0,
            "business_value_generated": 0.0,
            "cost_savings": 0.0,
            "time_saved_minutes": 0.0
        }
        
        self.performance_targets = {
            "max_execution_time": 3.0,  # 3 seconds target
            "min_success_rate": 0.95,   # 95% success rate
            "min_context_hit_rate": 0.85,  # 85% context utilization
            "max_p95_latency": 5.0      # P95 latency under 5 seconds
        }
        
        # Start background metrics aggregation
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background tasks for metrics aggregation"""
        # This would normally be handled by a task scheduler in production
        pass
    
    async def record_query_metrics(
        self, 
        original_query: str, 
        response: QueryResponse, 
        workflow_state: WorkflowState
    ):
        """Record comprehensive metrics for a query execution"""
        timestamp = datetime.now()
        
        try:
            # Query-level metrics
            query_metric = {
                "timestamp": timestamp.isoformat(),
                "session_id": response.session_id,
                "original_query": original_query,
                "success": response.success,
                "execution_time": response.execution_time,
                "sql_generated": bool(response.sql_query),
                "results_returned": bool(response.results and response.results.get("data")),
                "context_items_used": len(response.context_used),
                "confidence_score": response.confidence_score,
                "agent_statuses": dict(response.agent_statuses),
                "error_message": response.error_message
            }
            
            self.metrics_storage["queries"].append(query_metric)
            
            # Update real-time counters
            self.real_time_metrics["total_queries"] += 1
            self.real_time_metrics["total_execution_time"] += response.execution_time
            
            if response.success:
                self.real_time_metrics["successful_queries"] += 1
                
                # Calculate business impact
                business_impact = self._calculate_business_impact(
                    original_query, response, workflow_state
                )
                self.real_time_metrics["business_value_generated"] += business_impact["value_score"]
                self.real_time_metrics["time_saved_minutes"] += business_impact["time_saved"]
                self.real_time_metrics["cost_savings"] += business_impact["cost_savings"]
                
                # Record business impact
                self.metrics_storage["business_impact"].append({
                    "timestamp": timestamp.isoformat(),
                    "session_id": response.session_id,
                    **business_impact
                })
            else:
                self.real_time_metrics["failed_queries"] += 1
                
                # Record error
                self.metrics_storage["errors"].append({
                    "timestamp": timestamp.isoformat(),
                    "session_id": response.session_id,
                    "error_type": "query_execution",
                    "error_message": response.error_message,
                    "original_query": original_query
                })
            
            # Context utilization tracking
            if response.context_used:
                self.real_time_metrics["context_hits"] += 1
            
            # Agent-specific metrics
            for agent_name, status in response.agent_statuses.items():
                agent_data = getattr(workflow_state, agent_name, None)
                if agent_data:
                    self.metrics_storage["agent_metrics"][agent_name].append({
                        "timestamp": timestamp.isoformat(),
                        "status": status,
                        "execution_time": agent_data.execution_time if hasattr(agent_data, 'execution_time') else 0,
                        "confidence_score": agent_data.confidence_score if hasattr(agent_data, 'confidence_score') else None,
                        "session_id": response.session_id
                    })
            
            # Performance metrics
            self.metrics_storage["performance"].append({
                "timestamp": timestamp.isoformat(),
                "execution_time": response.execution_time,
                "success": response.success,
                "context_utilization": len(response.context_used),
                "session_id": response.session_id
            })
            
            logger.debug(f"Recorded metrics for session {response.session_id}")
            
        except Exception as e:
            logger.error(f"Failed to record metrics: {e}")
    
    def _calculate_business_impact(
        self, 
        query: str, 
        response: QueryResponse, 
        workflow_state: WorkflowState
    ) -> Dict[str, float]:
        """Calculate business impact metrics"""
        
        # Estimate time saved compared to manual SQL writing
        query_complexity = len(query.split()) / 20.0  # Rough complexity estimate
        estimated_manual_time = max(5, query_complexity * 15)  # 5-45 minutes
        actual_time = response.execution_time / 60.0  # Convert to minutes
        time_saved = max(0, estimated_manual_time - actual_time)
        
        # Calculate value score based on multiple factors
        value_factors = {
            "success": 0.3 if response.success else 0.0,
            "context_utilization": min(len(response.context_used) / 10.0, 0.2),
            "confidence": response.confidence_score * 0.2,
            "insights_generated": 0.3 if response.insights else 0.0
        }
        
        value_score = sum(value_factors.values())
        
        # Estimate cost savings (developer time saved)
        hourly_rate = 75.0  # $75/hour developer rate
        cost_savings = (time_saved / 60.0) * hourly_rate
        
        return {
            "time_saved": time_saved,
            "value_score": value_score,
            "cost_savings": cost_savings,
            "manual_time_estimate": estimated_manual_time,
            "actual_time": actual_time,
            "value_factors": value_factors
        }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        now = datetime.now()
        
        # Calculate derived metrics
        total_queries = self.real_time_metrics["total_queries"]
        success_rate = (
            self.real_time_metrics["successful_queries"] / max(total_queries, 1)
        )
        context_hit_rate = (
            self.real_time_metrics["context_hits"] / max(total_queries, 1)
        )
        avg_execution_time = (
            self.real_time_metrics["total_execution_time"] / max(total_queries, 1)
        )
        
        # Calculate P95 latency
        recent_times = [
            m["execution_time"] for m in list(self.metrics_storage["performance"])[-100:]
        ]
        p95_latency = (
            sorted(recent_times)[int(0.95 * len(recent_times))] 
            if recent_times else 0.0
        )
        
        # Agent performance summary
        agent_performance = {}
        for agent_name, metrics in self.metrics_storage["agent_metrics"].items():
            recent_metrics = metrics[-50:]  # Last 50 executions
            if recent_metrics:
                avg_time = sum(m["execution_time"] for m in recent_metrics) / len(recent_metrics)
                success_count = sum(1 for m in recent_metrics if m["status"] == "completed")
                agent_performance[agent_name] = {
                    "avg_execution_time": avg_time,
                    "success_rate": success_count / len(recent_metrics),
                    "total_executions": len(metrics)
                }
        
        # Performance vs targets
        performance_status = {
            "execution_time": {
                "current": avg_execution_time,
                "target": self.performance_targets["max_execution_time"],
                "status": "good" if avg_execution_time <= self.performance_targets["max_execution_time"] else "warning"
            },
            "success_rate": {
                "current": success_rate,
                "target": self.performance_targets["min_success_rate"],
                "status": "good" if success_rate >= self.performance_targets["min_success_rate"] else "warning"
            },
            "context_hit_rate": {
                "current": context_hit_rate,
                "target": self.performance_targets["min_context_hit_rate"],
                "status": "good" if context_hit_rate >= self.performance_targets["min_context_hit_rate"] else "warning"
            },
            "p95_latency": {
                "current": p95_latency,
                "target": self.performance_targets["max_p95_latency"],
                "status": "good" if p95_latency <= self.performance_targets["max_p95_latency"] else "warning"
            }
        }
        
        return {
            "timestamp": now.isoformat(),
            "total_queries": total_queries,
            "successful_queries": self.real_time_metrics["successful_queries"],
            "failed_queries": self.real_time_metrics["failed_queries"],
            "success_rate": success_rate,
            "context_hit_rate": context_hit_rate,
            "average_execution_time": avg_execution_time,
            "p95_latency": p95_latency,
            "error_rate": self.real_time_metrics["failed_queries"] / max(total_queries, 1),
            "business_impact": {
                "total_value_generated": self.real_time_metrics["business_value_generated"],
                "total_time_saved_minutes": self.real_time_metrics["time_saved_minutes"],
                "total_cost_savings": self.real_time_metrics["cost_savings"],
                "avg_value_per_query": self.real_time_metrics["business_value_generated"] / max(total_queries, 1)
            },
            "agent_performance": agent_performance,
            "performance_vs_targets": performance_status,
            "recent_trends": self._calculate_trends()
        }
    
    def _calculate_trends(self) -> Dict[str, Any]:
        """Calculate performance trends over time"""
        # Get metrics from last hour vs previous hour
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        two_hours_ago = now - timedelta(hours=2)
        
        recent_queries = [
            q for q in self.metrics_storage["queries"]
            if datetime.fromisoformat(q["timestamp"]) >= hour_ago
        ]
        
        previous_queries = [
            q for q in self.metrics_storage["queries"]
            if two_hours_ago <= datetime.fromisoformat(q["timestamp"]) < hour_ago
        ]
        
        def calculate_metrics_for_period(queries):
            if not queries:
                return {"success_rate": 0, "avg_time": 0, "count": 0}
            
            successful = sum(1 for q in queries if q["success"])
            total_time = sum(q["execution_time"] for q in queries)
            
            return {
                "success_rate": successful / len(queries),
                "avg_time": total_time / len(queries),
                "count": len(queries)
            }
        
        recent_metrics = calculate_metrics_for_period(recent_queries)
        previous_metrics = calculate_metrics_for_period(previous_queries)
        
        # Calculate trends
        trends = {}
        for metric in ["success_rate", "avg_time"]:
            if previous_metrics[metric] > 0:
                change = (recent_metrics[metric] - previous_metrics[metric]) / previous_metrics[metric]
                trends[metric] = {
                    "current": recent_metrics[metric],
                    "previous": previous_metrics[metric],
                    "change_percent": change * 100,
                    "trend": "improving" if (change > 0 and metric == "success_rate") or (change < 0 and metric == "avg_time") else "declining"
                }
            else:
                trends[metric] = {
                    "current": recent_metrics[metric],
                    "previous": 0,
                    "change_percent": 0,
                    "trend": "stable"
                }
        
        return trends
    
    def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter metrics by time period
        period_queries = [
            q for q in self.metrics_storage["queries"]
            if datetime.fromisoformat(q["timestamp"]) >= cutoff_time
        ]
        
        period_performance = [
            p for p in self.metrics_storage["performance"]
            if datetime.fromisoformat(p["timestamp"]) >= cutoff_time
        ]
        
        period_errors = [
            e for e in self.metrics_storage["errors"]
            if datetime.fromisoformat(e["timestamp"]) >= cutoff_time
        ]
        
        # Generate report
        if not period_queries:
            return {"error": "No queries in specified time period"}
        
        successful_queries = [q for q in period_queries if q["success"]]
        
        report = {
            "period_hours": hours,
            "summary": {
                "total_queries": len(period_queries),
                "successful_queries": len(successful_queries),
                "success_rate": len(successful_queries) / len(period_queries),
                "total_errors": len(period_errors),
                "error_rate": len(period_errors) / len(period_queries)
            },
            "performance": {
                "avg_execution_time": sum(q["execution_time"] for q in period_queries) / len(period_queries),
                "min_execution_time": min(q["execution_time"] for q in period_queries),
                "max_execution_time": max(q["execution_time"] for q in period_queries),
                "p95_execution_time": sorted([q["execution_time"] for q in period_queries])[int(0.95 * len(period_queries))]
            },
            "context_utilization": {
                "avg_context_items": sum(q["context_items_used"] for q in period_queries) / len(period_queries),
                "queries_with_context": sum(1 for q in period_queries if q["context_items_used"] > 0),
                "context_hit_rate": sum(1 for q in period_queries if q["context_items_used"] > 0) / len(period_queries)
            },
            "business_impact": {
                "total_time_saved": self.real_time_metrics["time_saved_minutes"],
                "total_cost_savings": self.real_time_metrics["cost_savings"],
                "avg_confidence_score": sum(q["confidence_score"] for q in successful_queries) / max(len(successful_queries), 1)
            },
            "top_errors": self._get_top_errors(period_errors),
            "recommendations": self._generate_performance_recommendations(period_queries, period_performance)
        }
        
        return report
    
    def _get_top_errors(self, errors: List[Dict]) -> List[Dict]:
        """Get top error patterns"""
        error_counts = defaultdict(int)
        for error in errors:
            error_type = error.get("error_message", "Unknown error")[:100]
            error_counts[error_type] += 1
        
        return [
            {"error": error, "count": count}
            for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ]
    
    def _generate_performance_recommendations(
        self, 
        queries: List[Dict], 
        performance: List[Dict]
    ) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        # Check success rate
        success_rate = sum(1 for q in queries if q["success"]) / len(queries)
        if success_rate < 0.9:
            recommendations.append(f"Success rate is {success_rate:.1%}. Consider improving SQL generation and validation.")
        
        # Check execution time
        avg_time = sum(q["execution_time"] for q in queries) / len(queries)
        if avg_time > 3.0:
            recommendations.append(f"Average execution time is {avg_time:.2f}s. Consider optimizing workflow agents.")
        
        # Check context utilization
        context_usage = sum(q["context_items_used"] for q in queries) / len(queries)
        if context_usage < 2:
            recommendations.append("Low context utilization. Consider expanding the example database.")
        
        # Check for patterns in failed queries
        failed_queries = [q for q in queries if not q["success"]]
        if len(failed_queries) > len(queries) * 0.1:
            recommendations.append("High failure rate detected. Review error patterns and improve validation.")
        
        return recommendations
    
    async def reset_metrics(self):
        """Reset all metrics (for development/testing)"""
        self.metrics_storage = {
            "queries": deque(maxlen=10000),
            "performance": deque(maxlen=1000),
            "agent_metrics": defaultdict(list),
            "business_impact": deque(maxlen=1000),
            "errors": deque(maxlen=1000)
        }
        
        self.real_time_metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_execution_time": 0.0,
            "context_hits": 0,
            "business_value_generated": 0.0,
            "cost_savings": 0.0,
            "time_saved_minutes": 0.0
        }
        
        logger.info("Metrics reset successfully")
    
    def export_metrics(self, format: str = "json") -> Dict[str, Any]:
        """Export metrics for external analysis"""
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "real_time_metrics": self.real_time_metrics,
            "recent_queries": list(self.metrics_storage["queries"])[-100:],
            "recent_performance": list(self.metrics_storage["performance"])[-100:],
            "agent_summaries": {
                agent: metrics[-20:] for agent, metrics in self.metrics_storage["agent_metrics"].items()
            },
            "recent_errors": list(self.metrics_storage["errors"])[-50:]
        }
        
        return export_data

# Global metrics service instance
metrics_service = MetricsService()