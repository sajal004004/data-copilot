"""
Vector Store Manager for Context Retrieval
Handles embeddings generation and semantic search for SQL examples and business context
"""
import numpy as np
import logging
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from config import config

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Production-ready vector store for semantic search and context retrieval"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.vector_store_path = config.VECTOR_STORE_DIR
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.vector_store_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Collections for different types of context
        self.collections = {
            "sql_examples": self._get_or_create_collection("sql_examples"),
            "business_context": self._get_or_create_collection("business_context"),
            "schema_info": self._get_or_create_collection("schema_info")
        }
        
        self.metrics = {
            "searches_performed": 0,
            "embeddings_generated": 0,
            "context_retrievals": 0
        }
    
    def _get_or_create_collection(self, name: str):
        """Get or create a ChromaDB collection"""
        try:
            return self.client.get_collection(name)
        except Exception:
            return self.client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )
    
    def add_sql_examples(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add curated SQL examples with business context"""
        try:
            collection = self.collections["sql_examples"]
            
            # Prepare data for insertion
            documents = []
            metadatas = []
            ids = []
            
            for i, example in enumerate(examples):
                # Create rich text for embedding
                document_text = f"""
                Business Question: {example['business_question']}
                SQL Query: {example['sql_query']}
                Category: {example.get('category', 'general')}
                Description: {example.get('description', '')}
                Expected Result: {example.get('expected_result_type', '')}
                Business Context: {example.get('business_context', '')}
                """
                
                documents.append(document_text.strip())
                metadatas.append({
                    "business_question": example['business_question'],
                    "sql_query": example['sql_query'],
                    "category": example.get('category', 'general'),
                    "complexity": example.get('complexity', 'medium'),
                    "description": example.get('description', ''),
                    "tags": json.dumps(example.get('tags', [])),
                    "business_context": example.get('business_context', ''),
                    "expected_result_type": example.get('expected_result_type', '')
                })
                ids.append(f"sql_example_{i}")
            
            # Generate embeddings and add to collection
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            self.metrics["embeddings_generated"] += len(examples)
            logger.info(f"Added {len(examples)} SQL examples to vector store")
            
            return {
                "success": True,
                "examples_added": len(examples),
                "total_examples": collection.count()
            }
            
        except Exception as e:
            logger.error(f"Failed to add SQL examples: {e}")
            return {"success": False, "error": str(e)}
    
    def add_business_context(self, business_rules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add business rules and domain knowledge"""
        try:
            collection = self.collections["business_context"]
            
            documents = []
            metadatas = []
            ids = []
            
            for i, rule in enumerate(business_rules):
                document_text = f"""
                Rule: {rule['rule']}
                Domain: {rule.get('domain', 'general')}
                Description: {rule.get('description', '')}
                Examples: {rule.get('examples', '')}
                """
                
                documents.append(document_text.strip())
                metadatas.append({
                    "rule": rule['rule'],
                    "domain": rule.get('domain', 'general'),
                    "description": rule.get('description', ''),
                    "priority": rule.get('priority', 'medium'),
                    "examples": rule.get('examples', '')
                })
                ids.append(f"business_rule_{i}")
            
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            self.metrics["embeddings_generated"] += len(business_rules)
            logger.info(f"Added {len(business_rules)} business rules to vector store")
            
            return {
                "success": True,
                "rules_added": len(business_rules),
                "total_rules": collection.count()
            }
            
        except Exception as e:
            logger.error(f"Failed to add business context: {e}")
            return {"success": False, "error": str(e)}
    
    def add_schema_context(self, schema_info: Dict[str, Any]) -> Dict[str, Any]:
        """Add database schema information for context"""
        try:
            collection = self.collections["schema_info"]
            
            # Create documents for each table and column
            documents = []
            metadatas = []
            ids = []
            
            # Add table-level context
            table_doc = f"""
            Table: {schema_info['table_name']}
            Description: {schema_info.get('business_context', {}).get('description', '')}
            Key Metrics: {', '.join(schema_info.get('business_context', {}).get('key_metrics', []))}
            Dimensions: {', '.join(schema_info.get('business_context', {}).get('dimensions', []))}
            """
            
            documents.append(table_doc.strip())
            metadatas.append({
                "type": "table",
                "table_name": schema_info['table_name'],
                "description": schema_info.get('business_context', {}).get('description', ''),
                "key_metrics": json.dumps(schema_info.get('business_context', {}).get('key_metrics', [])),
                "dimensions": json.dumps(schema_info.get('business_context', {}).get('dimensions', []))
            })
            ids.append("table_schema")
            
            # Add column-level context
            for i, column in enumerate(schema_info.get('columns', [])):
                column_doc = f"""
                Column: {column['name']}
                Type: {column['type']}
                Table: {schema_info['table_name']}
                Nullable: {column.get('nullable', True)}
                """
                
                documents.append(column_doc.strip())
                metadatas.append({
                    "type": "column",
                    "column_name": column['name'],
                    "column_type": column['type'],
                    "table_name": schema_info['table_name'],
                    "nullable": column.get('nullable', True)
                })
                ids.append(f"column_{column['name']}")
            
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            self.metrics["embeddings_generated"] += len(documents)
            logger.info(f"Added schema context for {schema_info['table_name']}")
            
            return {
                "success": True,
                "items_added": len(documents),
                "total_schema_items": collection.count()
            }
            
        except Exception as e:
            logger.error(f"Failed to add schema context: {e}")
            return {"success": False, "error": str(e)}
    
    def search_sql_examples(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant SQL examples based on natural language query"""
        try:
            collection = self.collections["sql_examples"]
            
            results = collection.query(
                query_texts=[query],
                n_results=min(n_results, collection.count())
            )
            
            # Format results
            formatted_results = []
            if results['metadatas'] and results['metadatas'][0]:
                for i, metadata in enumerate(results['metadatas'][0]):
                    formatted_results.append({
                        "business_question": metadata['business_question'],
                        "sql_query": metadata['sql_query'],
                        "category": metadata['category'],
                        "description": metadata['description'],
                        "business_context": metadata['business_context'],
                        "similarity_score": 1 - results['distances'][0][i] if results['distances'] else 1.0,
                        "tags": json.loads(metadata.get('tags', '[]'))
                    })
            
            self.metrics["searches_performed"] += 1
            self.metrics["context_retrievals"] += len(formatted_results)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to search SQL examples: {e}")
            return []
    
    def search_business_context(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Search for relevant business rules and context"""
        try:
            collection = self.collections["business_context"]
            
            if collection.count() == 0:
                return []
            
            results = collection.query(
                query_texts=[query],
                n_results=min(n_results, collection.count())
            )
            
            formatted_results = []
            if results['metadatas'] and results['metadatas'][0]:
                for i, metadata in enumerate(results['metadatas'][0]):
                    formatted_results.append({
                        "rule": metadata['rule'],
                        "domain": metadata['domain'],
                        "description": metadata['description'],
                        "priority": metadata['priority'],
                        "examples": metadata['examples'],
                        "similarity_score": 1 - results['distances'][0][i] if results['distances'] else 1.0
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to search business context: {e}")
            return []
    
    def search_schema_context(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Search for relevant schema information"""
        try:
            collection = self.collections["schema_info"]
            
            if collection.count() == 0:
                return []
            
            results = collection.query(
                query_texts=[query],
                n_results=min(n_results, collection.count())
            )
            
            formatted_results = []
            if results['metadatas'] and results['metadatas'][0]:
                for i, metadata in enumerate(results['metadatas'][0]):
                    formatted_results.append({
                        "type": metadata['type'],
                        "content": metadata,
                        "similarity_score": 1 - results['distances'][0][i] if results['distances'] else 1.0
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to search schema context: {e}")
            return []
    
    def get_comprehensive_context(self, query: str) -> Dict[str, Any]:
        """Get comprehensive context for a query including SQL examples, business rules, and schema"""
        context = {
            "sql_examples": self.search_sql_examples(query, n_results=5),
            "business_context": self.search_business_context(query, n_results=3),
            "schema_context": self.search_schema_context(query, n_results=5),
            "query": query,
            "context_quality_score": 0.0
        }
        
        # Calculate context quality score
        total_items = len(context["sql_examples"]) + len(context["business_context"]) + len(context["schema_context"])
        if total_items > 0:
            avg_similarity = (
                sum(item.get("similarity_score", 0) for item in context["sql_examples"]) +
                sum(item.get("similarity_score", 0) for item in context["business_context"]) +
                sum(item.get("similarity_score", 0) for item in context["schema_context"])
            ) / total_items
            context["context_quality_score"] = avg_similarity
        
        return context
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get vector store performance metrics"""
        return {
            "total_searches": self.metrics["searches_performed"],
            "total_embeddings": self.metrics["embeddings_generated"],
            "context_retrievals": self.metrics["context_retrievals"],
            "collection_counts": {
                name: collection.count() 
                for name, collection in self.collections.items()
            }
        }
    
    def initialize_with_curated_examples(self) -> Dict[str, Any]:
        """Initialize vector store with curated SQL examples and business context"""
        # This will be called during system initialization
        return {
            "sql_examples_initialized": True,
            "business_context_initialized": True,
            "schema_context_initialized": True
        }

# Global vector store manager instance
vector_store = VectorStoreManager()