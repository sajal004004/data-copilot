"""
Trust & Safety Layer for Data Copilot MVP
Comprehensive security, validation, and business rule enforcement
"""
import re
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from enum import Enum

import sqlparse
from sqlparse.sql import Statement
from sqlparse.tokens import Keyword, DML

logger = logging.getLogger(__name__)

class SecurityLevel(str, Enum):
    """Security levels for different validation modes"""
    STRICT = "strict"
    NORMAL = "normal"
    PERMISSIVE = "permissive"

class ValidationResult:
    """Comprehensive validation result with detailed feedback"""
    
    def __init__(self):
        self.is_valid = True
        self.is_safe = True
        self.security_level = SecurityLevel.NORMAL
        self.issues = []
        self.warnings = []
        self.suggestions = []
        self.risk_score = 0.0
        self.estimated_cost = 0.0
        self.performance_impact = "low"

class TrustSafetyEngine:
    """Production-ready trust and safety engine for SQL validation"""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.NORMAL):
        self.security_level = security_level
        self.validation_rules = self._initialize_validation_rules()
        self.business_rules = self._initialize_business_rules()
        self.performance_limits = self._initialize_performance_limits()
        
        # Track validation metrics
        self.metrics = {
            "validations_performed": 0,
            "queries_blocked": 0,
            "security_violations": 0,
            "performance_warnings": 0
        }
    
    def _initialize_validation_rules(self) -> Dict[str, Any]:
        """Initialize comprehensive validation rules"""
        return {
            "sql_injection": {
                "enabled": True,
                "patterns": [
                    r"'.*OR.*'.*=.*'",  # Classic OR injection
                    r"'.*UNION.*SELECT",  # Union-based injection
                    r"'.*;\s*DROP",  # Drop table injection
                    r"'.*;\s*DELETE",  # Delete injection
                    r"'.*;\s*UPDATE",  # Update injection
                    r"'.*;\s*INSERT",  # Insert injection
                    r"'.*;\s*EXEC",  # Execute injection
                    r"'.*;\s*xp_",  # SQL Server extended procedures
                    r"--.*",  # SQL comments (potential injection)
                    r"/\*.*\*/",  # Multi-line comments
                    r"'.*WAITFOR.*DELAY",  # Time-based injection
                    r"'.*BENCHMARK\(",  # MySQL benchmark injection
                    r"'.*SLEEP\(",  # MySQL sleep injection
                ],
                "severity": "critical"
            },
            "dangerous_operations": {
                "enabled": True,
                "blocked_statements": [
                    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE",
                    "TRUNCATE", "REPLACE", "ATTACH", "DETACH", "PRAGMA",
                    "VACUUM", "REINDEX"
                ],
                "severity": "high"
            },
            "data_exfiltration": {
                "enabled": True,
                "patterns": [
                    r"UNION\s+ALL\s+SELECT",  # Excessive data extraction
                    r"SELECT\s+\*\s+FROM\s+.*\s+WHERE\s+1\s*=\s*1",  # Dump all data
                    r"SELECT.*INTO\s+OUTFILE",  # File system access
                    r"SELECT.*INTO\s+DUMPFILE",  # MySQL file dump
                    r"LOAD_FILE\(",  # File reading functions
                ],
                "severity": "high"
            },
            "system_access": {
                "enabled": True,
                "patterns": [
                    r"EXEC\s*\(",  # Execute system commands
                    r"xp_cmdshell",  # SQL Server command shell
                    r"sp_OACreate",  # SQL Server OLE automation
                    r"master\.",  # System database access
                    r"information_schema\.",  # Schema introspection
                    r"pg_",  # PostgreSQL system functions
                    r"mysql\.",  # MySQL system database
                ],
                "severity": "critical"
            }
        }
    
    def _initialize_business_rules(self) -> Dict[str, Any]:
        """Initialize business-specific validation rules"""
        return {
            "data_access": {
                "allowed_tables": ["superstore"],
                "restricted_columns": [],  # No restricted columns for demo
                "max_rows_returned": 10000,
                "require_where_clause": False  # Allow aggregations without WHERE
            },
            "query_complexity": {
                "max_joins": 5,
                "max_subqueries": 3,
                "max_union_operations": 2,
                "max_nested_levels": 4
            },
            "performance_rules": {
                "max_execution_time": 30,  # seconds
                "warn_on_full_table_scan": True,
                "require_limit_for_large_results": True,
                "max_sort_operations": 3
            },
            "business_logic": {
                "financial_data_access": "allowed",
                "customer_data_access": "allowed", 
                "anonymization_required": False,  # Demo environment
                "audit_sensitive_queries": True
            }
        }
    
    def _initialize_performance_limits(self) -> Dict[str, Any]:
        """Initialize performance monitoring limits"""
        return {
            "execution_time": {
                "max_seconds": 30,
                "warning_threshold": 10
            },
            "resource_usage": {
                "max_memory_mb": 100,
                "max_cpu_percent": 50
            },
            "result_size": {
                "max_rows": 10000,
                "max_columns": 50,
                "warning_rows": 1000
            }
        }
    
    def validate_query_comprehensive(self, sql_query: str, context: Optional[Dict] = None) -> ValidationResult:
        """Perform comprehensive query validation with detailed feedback"""
        start_time = time.time()
        result = ValidationResult()
        
        try:
            self.metrics["validations_performed"] += 1
            
            # 1. Basic SQL syntax validation
            syntax_result = self._validate_sql_syntax(sql_query)
            if not syntax_result["valid"]:
                result.is_valid = False
                result.issues.extend(syntax_result["errors"])
            
            # 2. Security validation
            security_result = self._validate_security(sql_query)
            if not security_result["safe"]:
                result.is_safe = False
                result.issues.extend(security_result["violations"])
                self.metrics["security_violations"] += 1
            
            # 3. Business rules validation
            business_result = self._validate_business_rules(sql_query)
            if not business_result["compliant"]:
                result.warnings.extend(business_result["warnings"])
                if business_result["blocking_issues"]:
                    result.is_valid = False
                    result.issues.extend(business_result["blocking_issues"])
            
            # 4. Performance analysis
            performance_result = self._analyze_performance_impact(sql_query)
            result.performance_impact = performance_result["impact_level"]
            result.estimated_cost = performance_result["estimated_cost"]
            result.warnings.extend(performance_result["warnings"])
            
            if performance_result["warnings"]:
                self.metrics["performance_warnings"] += 1
            
            # 5. Calculate risk score
            result.risk_score = self._calculate_risk_score(
                security_result, business_result, performance_result
            )
            
            # 6. Generate suggestions
            result.suggestions = self._generate_optimization_suggestions(
                sql_query, security_result, business_result, performance_result
            )
            
            # Block query if necessary
            if not result.is_valid or not result.is_safe or result.risk_score > 0.8:
                self.metrics["queries_blocked"] += 1
                logger.warning(f"Query blocked - Risk score: {result.risk_score:.2f}")
            
            logger.debug(f"Query validation completed in {time.time() - start_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            result.is_valid = False
            result.is_safe = False
            result.issues.append(f"Validation system error: {str(e)}")
            return result
    
    def _validate_sql_syntax(self, sql_query: str) -> Dict[str, Any]:
        """Validate SQL syntax using sqlparse"""
        try:
            parsed = sqlparse.parse(sql_query)
            
            if not parsed:
                return {
                    "valid": False,
                    "errors": ["Empty or invalid SQL query"]
                }
            
            statement = parsed[0]
            
            # Check if it's a valid SELECT statement
            first_token = None
            for token in statement.flatten():
                if token.ttype is not sqlparse.tokens.Whitespace:
                    first_token = token
                    break
            
            if not first_token or first_token.value.upper() not in ['SELECT', 'WITH']:
                return {
                    "valid": False,
                    "errors": ["Only SELECT and WITH statements are allowed"]
                }
            
            return {
                "valid": True,
                "errors": [],
                "statement_type": first_token.value.upper()
            }
            
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"SQL parsing error: {str(e)}"]
            }
    
    def _validate_security(self, sql_query: str) -> Dict[str, Any]:
        """Comprehensive security validation"""
        violations = []
        risk_factors = []
        
        query_upper = sql_query.upper()
        query_lower = sql_query.lower()
        
        # Check for SQL injection patterns
        for pattern in self.validation_rules["sql_injection"]["patterns"]:
            if re.search(pattern, sql_query, re.IGNORECASE):
                violations.append(f"Potential SQL injection detected: {pattern}")
                risk_factors.append("sql_injection")
        
        # Check for dangerous operations
        for operation in self.validation_rules["dangerous_operations"]["blocked_statements"]:
            if operation in query_upper:
                violations.append(f"Dangerous operation not allowed: {operation}")
                risk_factors.append("dangerous_operation")
        
        # Check for data exfiltration patterns
        for pattern in self.validation_rules["data_exfiltration"]["patterns"]:
            if re.search(pattern, sql_query, re.IGNORECASE):
                violations.append(f"Potential data exfiltration pattern: {pattern}")
                risk_factors.append("data_exfiltration")
        
        # Check for system access attempts
        for pattern in self.validation_rules["system_access"]["patterns"]:
            if re.search(pattern, sql_query, re.IGNORECASE):
                violations.append(f"System access attempt detected: {pattern}")
                risk_factors.append("system_access")
        
        # Additional security checks
        if "'" in sql_query and any(char in sql_query for char in [';', '--', '/*']):
            violations.append("Suspicious quote and comment combination")
            risk_factors.append("suspicious_combination")
        
        return {
            "safe": len(violations) == 0,
            "violations": violations,
            "risk_factors": risk_factors,
            "security_score": max(0, 1.0 - len(violations) * 0.2)
        }
    
    def _validate_business_rules(self, sql_query: str) -> Dict[str, Any]:
        """Validate against business rules and policies"""
        warnings = []
        blocking_issues = []
        
        try:
            parsed = sqlparse.parse(sql_query)[0]
            query_upper = sql_query.upper()
            
            # Check table access
            allowed_tables = self.business_rules["data_access"]["allowed_tables"]
            if allowed_tables:
                found_tables = self._extract_table_names(sql_query)
                for table in found_tables:
                    if table.lower() not in [t.lower() for t in allowed_tables]:
                        blocking_issues.append(f"Access to table '{table}' not allowed")
            
            # Check query complexity
            complexity_rules = self.business_rules["query_complexity"]
            
            join_count = len(re.findall(r'\bJOIN\b', query_upper))
            if join_count > complexity_rules["max_joins"]:
                warnings.append(f"Query has {join_count} JOINs (max: {complexity_rules['max_joins']})")
            
            subquery_count = sql_query.count('(') - sql_query.count(')')
            if abs(subquery_count) > complexity_rules["max_subqueries"]:
                warnings.append(f"Complex nested structure detected")
            
            union_count = len(re.findall(r'\bUNION\b', query_upper))
            if union_count > complexity_rules["max_union_operations"]:
                warnings.append(f"Query has {union_count} UNIONs (max: {complexity_rules['max_union_operations']})")
            
            # Check for performance patterns
            if "SELECT *" in query_upper:
                warnings.append("Consider selecting specific columns instead of *")
            
            if "ORDER BY" in query_upper and "LIMIT" not in query_upper:
                warnings.append("ORDER BY without LIMIT may impact performance")
            
            return {
                "compliant": len(blocking_issues) == 0,
                "warnings": warnings,
                "blocking_issues": blocking_issues,
                "business_score": max(0, 1.0 - len(warnings) * 0.1 - len(blocking_issues) * 0.5)
            }
            
        except Exception as e:
            return {
                "compliant": False,
                "warnings": [],
                "blocking_issues": [f"Business rule validation error: {str(e)}"],
                "business_score": 0.0
            }
    
    def _analyze_performance_impact(self, sql_query: str) -> Dict[str, Any]:
        """Analyze potential performance impact"""
        warnings = []
        impact_factors = []
        
        query_upper = sql_query.upper()
        
        # Analyze query patterns for performance impact
        if "SELECT *" in query_upper:
            impact_factors.append("full_column_scan")
            warnings.append("SELECT * may retrieve unnecessary data")
        
        if len(re.findall(r'\bJOIN\b', query_upper)) > 2:
            impact_factors.append("multiple_joins")
            warnings.append("Multiple JOINs may impact performance")
        
        if "GROUP BY" in query_upper and "HAVING" in query_upper:
            impact_factors.append("complex_aggregation")
            warnings.append("Complex aggregation may require significant memory")
        
        if query_upper.count("SELECT") > 3:
            impact_factors.append("complex_subqueries")
            warnings.append("Complex subqueries may impact execution time")
        
        if "ORDER BY" in query_upper and "LIMIT" not in query_upper:
            impact_factors.append("unlimited_sort")
            warnings.append("Sorting without LIMIT may be expensive")
        
        # Estimate impact level
        if len(impact_factors) == 0:
            impact_level = "low"
            estimated_cost = 0.1
        elif len(impact_factors) <= 2:
            impact_level = "medium"
            estimated_cost = 0.3
        else:
            impact_level = "high"
            estimated_cost = 0.7
        
        return {
            "impact_level": impact_level,
            "estimated_cost": estimated_cost,
            "warnings": warnings,
            "impact_factors": impact_factors,
            "performance_score": max(0, 1.0 - len(impact_factors) * 0.2)
        }
    
    def _extract_table_names(self, sql_query: str) -> List[str]:
        """Extract table names from SQL query"""
        try:
            parsed = sqlparse.parse(sql_query)[0]
            tables = []
            
            # Simple regex-based extraction for demo
            from_pattern = r'\bFROM\s+(\w+)'
            join_pattern = r'\bJOIN\s+(\w+)'
            
            tables.extend(re.findall(from_pattern, sql_query, re.IGNORECASE))
            tables.extend(re.findall(join_pattern, sql_query, re.IGNORECASE))
            
            return list(set(tables))
            
        except Exception as e:
            logger.warning(f"Table extraction error: {e}")
            return []
    
    def _calculate_risk_score(self, security_result: Dict, business_result: Dict, performance_result: Dict) -> float:
        """Calculate overall risk score (0.0 = safe, 1.0 = high risk)"""
        security_score = security_result.get("security_score", 0.0)
        business_score = business_result.get("business_score", 0.0)
        performance_score = performance_result.get("performance_score", 0.0)
        
        # Weighted combination (security is most important)
        weights = {"security": 0.5, "business": 0.3, "performance": 0.2}
        
        weighted_score = (
            security_score * weights["security"] +
            business_score * weights["business"] +
            performance_score * weights["performance"]
        )
        
        # Invert score (higher score = lower risk)
        risk_score = 1.0 - weighted_score
        
        return min(max(risk_score, 0.0), 1.0)
    
    def _generate_optimization_suggestions(
        self, 
        sql_query: str, 
        security_result: Dict, 
        business_result: Dict, 
        performance_result: Dict
    ) -> List[str]:
        """Generate optimization and security suggestions"""
        suggestions = []
        
        # Security suggestions
        if security_result["violations"]:
            suggestions.append("Review query for potential security vulnerabilities")
            suggestions.append("Use parameterized queries to prevent SQL injection")
        
        # Performance suggestions
        if "SELECT *" in sql_query.upper():
            suggestions.append("Select only the columns you need instead of using SELECT *")
        
        if performance_result["impact_level"] == "high":
            suggestions.append("Consider adding appropriate indexes for better performance")
            suggestions.append("Review query structure to reduce complexity")
        
        if "ORDER BY" in sql_query.upper() and "LIMIT" not in sql_query.upper():
            suggestions.append("Add LIMIT clause to ORDER BY queries for better performance")
        
        # Business suggestions
        if len(business_result["warnings"]) > 2:
            suggestions.append("Simplify query structure for better maintainability")
        
        return suggestions[:5]  # Limit to top 5 suggestions
    
    def get_validation_metrics(self) -> Dict[str, Any]:
        """Get validation and security metrics"""
        total_validations = self.metrics["validations_performed"]
        
        return {
            "total_validations": total_validations,
            "queries_blocked": self.metrics["queries_blocked"],
            "security_violations": self.metrics["security_violations"],
            "performance_warnings": self.metrics["performance_warnings"],
            "block_rate": self.metrics["queries_blocked"] / max(total_validations, 1),
            "security_violation_rate": self.metrics["security_violations"] / max(total_validations, 1),
            "system_health": "healthy" if self.metrics["security_violations"] == 0 else "warning"
        }
    
    def update_security_level(self, new_level: SecurityLevel):
        """Update security validation level"""
        self.security_level = new_level
        logger.info(f"Security level updated to: {new_level}")
    
    def add_custom_business_rule(self, rule_name: str, rule_config: Dict[str, Any]):
        """Add custom business rule"""
        self.business_rules[rule_name] = rule_config
        logger.info(f"Added custom business rule: {rule_name}")

# Global trust and safety engine
trust_safety_engine = TrustSafetyEngine(SecurityLevel.NORMAL)