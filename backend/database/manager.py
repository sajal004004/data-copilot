"""
Database Manager for Data Copilot MVP
Handles SQLite database operations, data ingestion, and query execution with safety checks
"""
import sqlite3
import pandas as pd
import logging
import time
import sqlparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from contextlib import contextmanager
from config import config

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Production-ready database manager with safety and performance features"""
    
    def __init__(self):
        self.db_path = config.DATABASE_PATH
        self.connection_pool = []
        self.query_cache = {}
        self.metrics = {
            "queries_executed": 0,
            "cache_hits": 0,
            "execution_times": [],
            "errors": 0
        }
        
    @contextmanager
    def get_connection(self):
        """Context manager for database connections with proper cleanup"""
        conn = sqlite3.connect(self.db_path, timeout=config.DEFAULT_QUERY_TIMEOUT)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()
    
    def initialize_database(self) -> Dict[str, Any]:
        """Initialize database with Superstore schema and constraints"""
        start_time = time.time()
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create superstore table with optimized schema
                create_table_sql = """
                CREATE TABLE IF NOT EXISTS superstore (
                    row_id INTEGER PRIMARY KEY,
                    order_id TEXT NOT NULL,
                    order_date DATE NOT NULL,
                    ship_date DATE,
                    ship_mode TEXT,
                    customer_id TEXT NOT NULL,
                    customer_name TEXT NOT NULL,
                    segment TEXT,
                    country TEXT,
                    city TEXT,
                    state TEXT,
                    postal_code TEXT,
                    region TEXT,
                    product_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    sub_category TEXT,
                    product_name TEXT NOT NULL,
                    sales REAL NOT NULL DEFAULT 0,
                    quantity INTEGER NOT NULL DEFAULT 0,
                    discount REAL DEFAULT 0,
                    profit REAL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
                cursor.execute(create_table_sql)
                
                # Create indexes for performance
                indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_order_date ON superstore(order_date);",
                    "CREATE INDEX IF NOT EXISTS idx_category ON superstore(category);",
                    "CREATE INDEX IF NOT EXISTS idx_region ON superstore(region);",
                    "CREATE INDEX IF NOT EXISTS idx_customer_id ON superstore(customer_id);",
                    "CREATE INDEX IF NOT EXISTS idx_sales ON superstore(sales);",
                    "CREATE INDEX IF NOT EXISTS idx_profit ON superstore(profit);"
                ]
                
                for index_sql in indexes:
                    cursor.execute(index_sql)
                
                # Create metrics table
                metrics_table_sql = """
                CREATE TABLE IF NOT EXISTS query_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_text TEXT NOT NULL,
                    execution_time REAL NOT NULL,
                    result_count INTEGER,
                    success BOOLEAN NOT NULL,
                    error_message TEXT,
                    context_used TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
                cursor.execute(metrics_table_sql)
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            return {"success": False, "error": str(e)}
        
        execution_time = time.time() - start_time
        logger.info(f"Database initialized successfully in {execution_time:.2f}s")
        
        return {
            "success": True, 
            "execution_time": execution_time,
            "tables_created": ["superstore", "query_metrics"]
        }
    
    def ingest_superstore_data(self, csv_path: str) -> Dict[str, Any]:
        """Ingest Superstore CSV data with validation and error handling"""
        start_time = time.time()
        
        try:
            # Read and validate CSV with encoding detection
            try:
                df = pd.read_csv(csv_path, encoding='utf-8')
            except UnicodeDecodeError:
                # Try common alternative encodings
                for encoding in ['latin-1', 'iso-8859-1', 'cp1252', 'utf-8-sig']:
                    try:
                        df = pd.read_csv(csv_path, encoding=encoding)
                        logger.info(f"Successfully loaded CSV with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise ValueError("Could not decode CSV file with any common encoding")
            
            logger.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            
            # Clean and standardize column names
            df.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df.columns]
            
            # Data validation and cleaning
            df = self._clean_superstore_data(df)
            
            # Insert data into database
            with self.get_connection() as conn:
                # Clear existing data
                conn.execute("DELETE FROM superstore")
                
                # Insert new data
                df.to_sql('superstore', conn, if_exists='append', index=False)
                conn.commit()
                
                # Verify insertion
                cursor = conn.cursor()
                count_result = cursor.execute("SELECT COUNT(*) FROM superstore").fetchone()
                inserted_count = count_result[0]
                
        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
            return {"success": False, "error": str(e)}
        
        execution_time = time.time() - start_time
        logger.info(f"Data ingested successfully: {inserted_count} rows in {execution_time:.2f}s")
        
        return {
            "success": True,
            "rows_inserted": inserted_count,
            "execution_time": execution_time,
            "data_summary": self.get_data_summary()
        }
    
    def _clean_superstore_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate Superstore data"""
        # Convert date columns
        date_columns = ['order_date', 'ship_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Ensure numeric columns are properly typed
        numeric_columns = ['sales', 'quantity', 'discount', 'profit']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Remove rows with missing critical data
        critical_columns = ['order_id', 'customer_id', 'product_id', 'category']
        df = df.dropna(subset=[col for col in critical_columns if col in df.columns])
        
        # Add row_id if not present
        if 'row_id' not in df.columns:
            df.reset_index(drop=True, inplace=True)
            df.index.name = 'row_id'
            df.reset_index(inplace=True)
        
        return df
    
    def execute_safe_query(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute SQL query with comprehensive safety checks and metrics"""
        start_time = time.time()
        query_hash = hash(query)
        
        # Check cache first
        if query_hash in self.query_cache:
            self.metrics["cache_hits"] += 1
            return self.query_cache[query_hash]
        
        try:
            # Validate query safety
            validation_result = self._validate_query_safety(query)
            if not validation_result["safe"]:
                return {
                    "success": False,
                    "error": f"Query validation failed: {validation_result['reason']}",
                    "execution_time": time.time() - start_time
                }
            
            # Execute query with timeout
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Set query timeout (in milliseconds)
                timeout_ms = config.DEFAULT_QUERY_TIMEOUT * 1000
                cursor.execute(f"PRAGMA busy_timeout = {timeout_ms}")
                
                # Execute the query
                cursor.execute(query)
                
                # Fetch results with row limit
                if query.strip().lower().startswith('select'):
                    results = cursor.fetchmany(config.MAX_RESULT_ROWS)
                    columns = [description[0] for description in cursor.description]
                    
                    result_data = {
                        "success": True,
                        "data": [dict(zip(columns, row)) for row in results],
                        "columns": columns,
                        "row_count": len(results),
                        "execution_time": time.time() - start_time,
                        "cached": False
                    }
                else:
                    conn.commit()
                    result_data = {
                        "success": True,
                        "message": "Query executed successfully",
                        "affected_rows": cursor.rowcount,
                        "execution_time": time.time() - start_time
                    }
                
                # Cache successful SELECT queries
                if query.strip().lower().startswith('select'):
                    self.query_cache[query_hash] = result_data
                
                # Record metrics
                self._record_query_metrics(query, result_data["execution_time"], True, len(results) if 'data' in result_data else 0, context)
                
                return result_data
                
        except Exception as e:
            error_msg = str(e)
            execution_time = time.time() - start_time
            
            logger.error(f"Query execution failed: {error_msg}")
            self.metrics["errors"] += 1
            
            # Record error metrics
            self._record_query_metrics(query, execution_time, False, 0, context, error_msg)
            
            return {
                "success": False,
                "error": error_msg,
                "execution_time": execution_time
            }
        
        finally:
            self.metrics["queries_executed"] += 1
            self.metrics["execution_times"].append(time.time() - start_time)
    
    def _validate_query_safety(self, query: str) -> Dict[str, Any]:
        """Comprehensive query validation for safety and performance"""
        query_lower = query.lower().strip()
        
        # Check for dangerous operations
        dangerous_keywords = [
            'drop', 'delete', 'update', 'insert', 'alter', 'create', 
            'truncate', 'replace', 'attach', 'detach'
        ]
        
        for keyword in dangerous_keywords:
            if keyword in query_lower.split():
                return {"safe": False, "reason": f"Dangerous operation detected: {keyword}"}
        
        # Validate SQL syntax
        try:
            parsed = sqlparse.parse(query)
            if not parsed:
                return {"safe": False, "reason": "Invalid SQL syntax"}
        except Exception as e:
            return {"safe": False, "reason": f"SQL parsing error: {e}"}
        
        # Check for SQL injection patterns
        injection_patterns = ["'", '"', '--', '/*', '*/', 'union', 'exec', 'xp_']
        for pattern in injection_patterns:
            if pattern in query_lower:
                return {"safe": False, "reason": f"Potential SQL injection detected: {pattern}"}
        
        return {"safe": True, "reason": "Query passed all safety checks"}
    
    def _record_query_metrics(self, query: str, execution_time: float, success: bool, 
                            result_count: int, context: Optional[Dict] = None, 
                            error_message: Optional[str] = None):
        """Record query execution metrics for analytics"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO query_metrics 
                    (query_text, execution_time, result_count, success, error_message, context_used)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (query, execution_time, result_count, success, error_message, 
                     str(context) if context else None))
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to record metrics: {e}")
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Get comprehensive database schema information for context"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get table info
                cursor.execute("PRAGMA table_info(superstore)")
                columns = cursor.fetchall()
                
                # Get sample data
                cursor.execute("SELECT * FROM superstore LIMIT 5")
                sample_data = cursor.fetchall()
                
                # Get statistics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_rows,
                        COUNT(DISTINCT category) as unique_categories,
                        COUNT(DISTINCT region) as unique_regions,
                        COUNT(DISTINCT customer_id) as unique_customers,
                        MIN(order_date) as earliest_order,
                        MAX(order_date) as latest_order,
                        SUM(sales) as total_sales,
                        SUM(profit) as total_profit
                    FROM superstore
                """)
                stats = cursor.fetchone()
                
                return {
                    "table_name": "superstore",
                    "columns": [{"name": col[1], "type": col[2], "nullable": not col[3]} for col in columns],
                    "sample_data": [dict(row) for row in sample_data],
                    "statistics": dict(stats) if stats else {},
                    "business_context": {
                        "description": "Retail superstore sales data with orders, customers, products, and financial metrics",
                        "key_metrics": ["sales", "profit", "quantity", "discount"],
                        "dimensions": ["category", "region", "segment", "ship_mode"],
                        "date_range": f"{stats[4]} to {stats[5]}" if stats else "Unknown"
                    }
                }
                
        except Exception as e:
            logger.error(f"Failed to get schema info: {e}")
            return {"error": str(e)}
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get comprehensive data summary for business context"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Execute summary queries
                summaries = {}
                
                # Category performance
                cursor.execute("""
                    SELECT category, 
                           COUNT(*) as order_count,
                           SUM(sales) as total_sales,
                           SUM(profit) as total_profit,
                           AVG(discount) as avg_discount
                    FROM superstore 
                    GROUP BY category 
                    ORDER BY total_sales DESC
                """)
                summaries["category_performance"] = [dict(row) for row in cursor.fetchall()]
                
                # Regional analysis
                cursor.execute("""
                    SELECT region,
                           COUNT(*) as order_count,
                           SUM(sales) as total_sales,
                           SUM(profit) as total_profit
                    FROM superstore 
                    GROUP BY region 
                    ORDER BY total_sales DESC
                """)
                summaries["regional_analysis"] = [dict(row) for row in cursor.fetchall()]
                
                # Customer segments
                cursor.execute("""
                    SELECT segment,
                           COUNT(DISTINCT customer_id) as customer_count,
                           SUM(sales) as total_sales,
                           AVG(profit) as avg_profit
                    FROM superstore 
                    GROUP BY segment 
                    ORDER BY total_sales DESC
                """)
                summaries["segment_analysis"] = [dict(row) for row in cursor.fetchall()]
                
                return summaries
                
        except Exception as e:
            logger.error(f"Failed to get data summary: {e}")
            return {"error": str(e)}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get database performance and usage metrics"""
        return {
            "total_queries": self.metrics["queries_executed"],
            "cache_hit_rate": self.metrics["cache_hits"] / max(self.metrics["queries_executed"], 1),
            "average_execution_time": sum(self.metrics["execution_times"]) / max(len(self.metrics["execution_times"]), 1),
            "error_rate": self.metrics["errors"] / max(self.metrics["queries_executed"], 1),
            "p95_execution_time": sorted(self.metrics["execution_times"])[int(0.95 * len(self.metrics["execution_times"]))] if self.metrics["execution_times"] else 0
        }

# Global database manager instance
db_manager = DatabaseManager()