"""
Streamlit Frontend for Data Copilot MVP
Simple chat interface for natural language queries
"""
import streamlit as st
import requests
import json
import time
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Configure page
st.set_page_config(
    page_title="Data Copilot MVP",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for ChatGPT-like styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e88e5;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .chat-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #1e88e5;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
    }
    
    .assistant-message {
        background-color: #f1f8e9;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #4caf50;
    }
    
    .agent-status {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem;
        font-weight: bold;
    }
    
    .agent-completed {
        background-color: #c8e6c9;
        color: #2e7d32;
    }
    
    .agent-running {
        background-color: #fff3e0;
        color: #ef6c00;
    }
    
    .agent-failed {
        background-color: #ffcdd2;
        color: #c62828;
    }
    
    .metrics-card {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .demo-card {
        background-color: #fff8e1;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #ffcc02;
    }
</style>
""", unsafe_allow_html=True)

# Backend API configuration
API_BASE_URL = "http://localhost:8000"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "agent_status" not in st.session_state:
    st.session_state.agent_status = {}

def check_backend_health():
    """Check if backend API is available"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def send_query(query: str):
    """Send query to backend and get response"""
    try:
        payload = {
            "natural_language_query": query,
            "include_explanation": True,
            "max_results": 100
        }
        
        with st.spinner("🤖 Processing your query through the agentic workflow..."):
            response = requests.post(
                f"{API_BASE_URL}/query", 
                json=payload, 
                timeout=120
            )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None

def display_agent_status(agent_statuses: Dict[str, str]):
    """Display real-time agent execution status"""
    st.subheader("🔄 Agentic Workflow Status")
    
    agents = [
        ("Context Retrieval", "🔍"),
        ("SQL Generation", "⚡"),
        ("Validation", "✅"),
        ("Explanation", "📝"),
        ("Execution", "🚀"),
        ("Synthesis", "🧠")
    ]
    
    cols = st.columns(len(agents))
    
    for i, (agent_name, icon) in enumerate(agents):
        with cols[i]:
            agent_key = agent_name.lower().replace(" ", "_")
            status = agent_statuses.get(agent_key, "pending")
            
            if status == "completed":
                st.success(f"{icon} {agent_name}")
            elif status == "running":
                st.warning(f"{icon} {agent_name}")
            elif status == "failed":
                st.error(f"{icon} {agent_name}")
            else:
                st.info(f"{icon} {agent_name}")

def display_query_result(result: Dict[str, Any]):
    """Display comprehensive query result"""
    if not result:
        return
    
    # Display agent status
    if result.get("agent_statuses"):
        display_agent_status(result["agent_statuses"])
    
    st.divider()
    
    # Main result tabs
    tabs = st.tabs(["💬 Response", "📊 Results", "🔍 Insights", "⚙️ Technical Details"])
    
    with tabs[0]:  # Response tab
        if result.get("business_explanation"):
            st.markdown("### 🎯 Business Analysis")
            st.markdown(f'<div class="assistant-message">{result["business_explanation"]}</div>', 
                       unsafe_allow_html=True)
        
        if result.get("sql_query"):
            st.markdown("### 📝 Generated SQL")
            st.code(result["sql_query"], language="sql")
    
    with tabs[1]:  # Results tab
        if result.get("results") and result["results"].get("success"):
            query_results = result["results"]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows Returned", query_results.get("row_count", 0))
            with col2:
                st.metric("Execution Time", f"{query_results.get('execution_time', 0):.3f}s")
            with col3:
                st.metric("Confidence", f"{result.get('confidence_score', 0):.1%}")
            
            # Display data table
            if query_results.get("data"):
                df = pd.DataFrame(query_results["data"])
                st.dataframe(df, use_container_width=True)
                
                # Auto-generate simple visualizations
                if len(df) > 1:
                    st.markdown("### 📈 Quick Visualization")
                    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
                    
                    if len(numeric_columns) >= 1:
                        # Create appropriate chart based on data
                        if len(df.columns) >= 2:
                            x_col = df.columns[0]
                            y_col = numeric_columns[0]
                            
                            if len(df) <= 20:  # Bar chart for small datasets
                                fig = px.bar(df, x=x_col, y=y_col, 
                                           title=f"{y_col} by {x_col}")
                            else:  # Line chart for larger datasets
                                fig = px.line(df, x=x_col, y=y_col, 
                                            title=f"{y_col} trend")
                            
                            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Query execution failed")
            if result.get("error_message"):
                st.error(f"Error: {result['error_message']}")
    
    with tabs[2]:  # Insights tab
        if result.get("insights"):
            insights = result["insights"]
            
            if insights.get("insights"):
                st.markdown("### 🔍 Key Insights")
                for insight in insights["insights"]:
                    if isinstance(insight, dict):
                        st.markdown(f"**{insight.get('title', 'Insight')}**")
                        st.markdown(insight.get('description', ''))
                    else:
                        st.markdown(f"• {insight}")
            
            if insights.get("recommendations"):
                st.markdown("### 💡 Recommendations")
                for rec in insights["recommendations"]:
                    st.markdown(f"• {rec}")
            
            # Context utilization
            if result.get("context_used"):
                st.markdown("### 🧠 Context Utilized")
                context_df = pd.DataFrame(result["context_used"])
                if not context_df.empty:
                    st.dataframe(context_df[["source", "similarity_score"]], use_container_width=True)
    
    with tabs[3]:  # Technical details
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ⚡ Performance Metrics")
            metrics = result.get("execution_metrics", {})
            
            st.metric("Total Execution Time", f"{result.get('execution_time', 0):.3f}s")
            st.metric("Context Utilization", f"{metrics.get('context_utilization_score', 0):.1%}")
            st.metric("Business Relevance", f"{metrics.get('business_relevance_score', 0):.1%}")
        
        with col2:
            st.markdown("### 🔧 Workflow Details")
            st.json({
                "session_id": result.get("session_id", ""),
                "workflow_status": result.get("workflow_status", ""),
                "agent_statuses": result.get("agent_statuses", {}),
                "context_items": len(result.get("context_used", []))
            })

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Data Copilot MVP</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Natural Language to SQL with AI</p>', unsafe_allow_html=True)
    
    # Check backend connectivity
    if not check_backend_health():
        st.error("❌ Backend API is not available. Please ensure the FastAPI server is running on http://localhost:8000")
        st.stop()
    
    # Simple sidebar with sample queries
    with st.sidebar:
        st.header("💡 Sample Queries")
        st.markdown("Click any query below to try it out:")
        
        sample_queries = [
            "Show me sales trends by category",
            "Which customers have the highest profit margins?",
            "What products are declining in sales?",
            "Compare regional performance this year",
            "Show me seasonal sales patterns",
            "Top 10 profitable products",
            "Customer segments by revenue"
        ]
        
        for query in sample_queries:
            if st.button(query, key=f"sample_{hash(query)}", use_container_width=True):
                st.session_state.sample_query = query
                st.rerun()
    
    # Display the chat interface
    display_chat_interface()

def display_chat_interface():
    """Display the main chat interface"""
    
    # Auto-fill sample query if selected
    initial_query = ""
    if hasattr(st.session_state, 'sample_query'):
        initial_query = st.session_state.sample_query
        delattr(st.session_state, 'sample_query')
    
    # Chat input
    query = st.chat_input("Ask me anything about your data...", key="main_chat_input")
    
    # Handle sample query
    if initial_query:
        query = initial_query
    
    # Process new query
    if query:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Get response from backend
        with st.chat_message("user"):
            st.markdown(query)
        
        with st.chat_message("assistant"):
            result = send_query(query)
            
            if result:
                st.session_state.current_session_id = result.get("session_id")
                st.session_state.messages.append({"role": "assistant", "content": result})
                display_query_result(result)
            else:
                st.error("Failed to process query. Please try again.")

if __name__ == "__main__":
    main()