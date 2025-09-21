#!/usr/bin/env python3
"""
Data Copilot MVP - System Validation Script
Validates complete system functionality and performance metrics.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
import requests
from typing import Dict, List, Any
import subprocess

class SystemValidator:
    """Comprehensive system validation for Data Copilot MVP."""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.frontend_url = "http://localhost:8501"
        self.results = {
            "passed": 0,
            "failed": 0,
            "tests": []
        }
    
    def log_test(self, name: str, passed: bool, details: str = ""):
        """Log test result."""
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}")
        if details:
            print(f"    {details}")
        
        self.results["tests"].append({
            "name": name,
            "passed": passed,
            "details": details
        })
        
        if passed:
            self.results["passed"] += 1
        else:
            self.results["failed"] += 1
    
    def test_backend_health(self) -> bool:
        """Test backend API health."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.log_test("Backend Health Check", True, 
                            f"Status: {data.get('status', 'unknown')}")
                return True
            else:
                self.log_test("Backend Health Check", False, 
                            f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Backend Health Check", False, str(e))
            return False
    
    def test_database_connection(self) -> bool:
        """Test database connectivity."""
        try:
            response = requests.get(f"{self.base_url}/schema", timeout=10)
            if response.status_code == 200:
                schema = response.json()
                
                # Check if schema has table_name (single table format)
                if "table_name" in schema:
                    has_superstore = schema["table_name"] == "superstore"
                    tables_count = 1 if has_superstore else 0
                else:
                    # Legacy format with tables array
                    tables = schema.get("tables", [])
                    has_superstore = any(t.get("name") == "superstore" for t in tables)
                    tables_count = len(tables)
                
                self.log_test("Database Connection", has_superstore,
                            f"Found {tables_count} tables, superstore: {has_superstore}")
                return has_superstore
            else:
                self.log_test("Database Connection", False,
                            f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Database Connection", False, str(e))
            return False
    
    def test_simple_query(self) -> bool:
        """Test basic SQL generation."""
        try:
            query_data = {
                "natural_language_query": "Show me total sales by category",
                "include_explanation": True
            }
            
            start_time = time.time()
            response = requests.post(f"{self.base_url}/query", 
                                   json=query_data, timeout=30)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                success = result.get("success", False)
                sql_query = result.get("sql_query", "")
                results = result.get("results") or []  # Handle null results
                has_results = len(results) > 0
                
                passed = success and sql_query and "SELECT" in sql_query.upper() and has_results
                error_msg = result.get("error_message", "")
                status_msg = f"Response time: {response_time:.2f}s, Results: {len(results)}"
                if error_msg:
                    status_msg += f", Error: {error_msg[:100]}"
                    
                self.log_test("Simple Query Generation", passed, status_msg)
                return passed
            else:
                self.log_test("Simple Query Generation", False,
                            f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Simple Query Generation", False, str(e))
            return False
    
    def test_complex_query(self) -> bool:
        """Test complex business query."""
        try:
            query_data = {
                "natural_language_query": "Show me the top 5 customers by profit with their order counts and average order value",
                "include_explanation": True
            }
            
            start_time = time.time()
            response = requests.post(f"{self.base_url}/query", 
                                   json=query_data, timeout=30)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                success = result.get("success", False)
                sql_query = result.get("sql_query", "")
                results = result.get("results") or []  # Handle null results
                has_results = len(results) > 0
                
                # Check for complex query patterns
                complex_indicators = ["GROUP BY", "ORDER BY", "LIMIT", "SUM", "COUNT", "AVG"]
                has_complex_features = sum(1 for indicator in complex_indicators 
                                         if indicator in sql_query.upper()) >= 3
                
                passed = success and has_complex_features and has_results
                self.log_test("Complex Query Generation", passed,
                            f"Response time: {response_time:.2f}s, Complex features: {has_complex_features}")
                return passed
            else:
                self.log_test("Complex Query Generation", False,
                            f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Complex Query Generation", False, str(e))
            return False
    
    def test_security_validation(self) -> bool:
        """Test security validation blocks dangerous queries."""
        try:
            dangerous_queries = [
                "DROP TABLE superstore; SELECT * FROM superstore",
                "DELETE FROM superstore WHERE 1=1",
                "UPDATE superstore SET sales = 0",
                "'; DROP TABLE superstore; --"
            ]
            
            blocked_count = 0
            for dangerous_query in dangerous_queries:
                query_data = {
                    "natural_language_query": dangerous_query,
                    "include_explanation": False
                }
                
                response = requests.post(f"{self.base_url}/query", 
                                       json=query_data, timeout=10)
                
                if response.status_code == 200:
                    result = response.json()
                    if not result.get("success", True):  # Should fail validation
                        blocked_count += 1
                else:
                    blocked_count += 1  # HTTP error is also blocking
            
            passed = blocked_count == len(dangerous_queries)
            self.log_test("Security Validation", passed,
                        f"Blocked {blocked_count}/{len(dangerous_queries)} dangerous queries")
            return passed
        except Exception as e:
            self.log_test("Security Validation", False, str(e))
            return False
    
    def test_demo_scenarios(self) -> bool:
        """Test demo scenarios execution."""
        try:
            scenarios = [
                "declining_categories",
                "customer_segments", 
                "sales_anomalies",
                "regional_performance"
            ]
            
            successful_scenarios = 0
            for scenario in scenarios:
                response = requests.post(f"{self.base_url}/demo/scenario/{scenario}",
                                       timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("success", False):
                        successful_scenarios += 1
            
            passed = successful_scenarios >= len(scenarios) * 0.75  # 75% success rate
            self.log_test("Demo Scenarios", passed,
                        f"Successful scenarios: {successful_scenarios}/{len(scenarios)}")
            return passed
        except Exception as e:
            self.log_test("Demo Scenarios", False, str(e))
            return False
    
    def test_performance_metrics(self) -> bool:
        """Test performance metrics collection."""
        try:
            response = requests.get(f"{self.base_url}/metrics", timeout=10)
            if response.status_code == 200:
                metrics = response.json()
                
                # Check for key metrics
                has_metrics = all(key in metrics for key in [
                    "total_queries", "success_rate", "avg_response_time", "avg_business_value"
                ])
                
                self.log_test("Performance Metrics", has_metrics,
                            f"Metrics available: {list(metrics.keys())}")
                return has_metrics
            else:
                self.log_test("Performance Metrics", False,
                            f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Performance Metrics", False, str(e))
            return False
    
    def test_response_time_target(self) -> bool:
        """Test response time meets <3 second target."""
        try:
            query_data = {
                "natural_language_query": "Show me sales by region",
                "include_explanation": True
            }
            
            response_times = []
            for _ in range(3):  # Test 3 times
                start_time = time.time()
                response = requests.post(f"{self.base_url}/query", 
                                       json=query_data, timeout=30)
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    response_times.append(response_time)
            
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
                passed = avg_response_time < 3.0
                self.log_test("Response Time Target", passed,
                            f"Average response time: {avg_response_time:.2f}s (target: <3s)")
                return passed
            else:
                self.log_test("Response Time Target", False, "No successful responses")
                return False
        except Exception as e:
            self.log_test("Response Time Target", False, str(e))
            return False
    
    def test_frontend_accessibility(self) -> bool:
        """Test frontend accessibility."""
        try:
            response = requests.get(self.frontend_url, timeout=10)
            passed = response.status_code == 200
            self.log_test("Frontend Accessibility", passed,
                        f"HTTP {response.status_code}")
            return passed
        except Exception as e:
            self.log_test("Frontend Accessibility", False, str(e))
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete system validation."""
        print("🚀 Starting Data Copilot MVP System Validation")
        print("=" * 60)
        
        # Core functionality tests
        self.test_backend_health()
        self.test_database_connection()
        self.test_simple_query()
        self.test_complex_query()
        self.test_security_validation()
        self.test_demo_scenarios()
        self.test_performance_metrics()
        self.test_response_time_target()
        self.test_frontend_accessibility()
        
        print("=" * 60)
        print(f"✅ Tests Passed: {self.results['passed']}")
        print(f"❌ Tests Failed: {self.results['failed']}")
        
        total_tests = self.results['passed'] + self.results['failed']
        success_rate = (self.results['passed'] / total_tests * 100) if total_tests > 0 else 0
        print(f"📊 Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 80:
            print("\n🎉 SYSTEM VALIDATION SUCCESSFUL!")
            print("The Data Copilot MVP is ready for production use.")
        else:
            print("\n⚠️  SYSTEM VALIDATION INCOMPLETE")
            print("Some tests failed. Please review the results above.")
        
        return self.results

def main():
    """Main validation entry point."""
    print("Data Copilot MVP - System Validation")
    print("Validating complete system functionality...")
    print()
    
    # Check if services are running
    print("Checking if services are running...")
    
    try:
        backend_check = requests.get("http://localhost:8000/health", timeout=5)
        if backend_check.status_code != 200:
            print("❌ Backend service not running on port 8000")
            print("Please start the backend with: python start_backend.py")
            return False
    except:
        print("❌ Backend service not responding on port 8000")
        print("Please start the backend with: python start_backend.py")
        return False
    
    try:
        frontend_check = requests.get("http://localhost:8501", timeout=5)
        if frontend_check.status_code != 200:
            print("⚠️  Frontend service not responding on port 8501")
            print("Consider starting the frontend with: python start_frontend.py")
    except:
        print("⚠️  Frontend service not responding on port 8501")
        print("Consider starting the frontend with: python start_frontend.py")
    
    print("✅ Backend service is running")
    print()
    
    # Run validation
    validator = SystemValidator()
    results = validator.run_all_tests()
    
    # Save results
    with open("validation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📋 Detailed results saved to: validation_results.json")
    
    return results['passed'] >= results['failed']

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)