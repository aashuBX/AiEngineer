import os

project_dir = "/Users/aashu-kumar-jha/Documents/projects/AiEngineer"

expected_files = [
    # AiAgents
    "AiAgents/pyproject.toml",
    "AiAgents/src/config/settings.py",
    "AiAgents/src/config/llm_providers.py",
    "AiAgents/src/models/state.py",
    "AiAgents/src/models/schemas.py",
    "AiAgents/src/agents/base_agent.py",
    "AiAgents/src/agents/intent/intent_agent.py",
    "AiAgents/src/agents/guardrail/guardrail_agent.py",
    "AiAgents/src/agents/graph_rag/graph_rag_agent.py",
    "AiAgents/src/agents/crm/crm_agent.py",
    "AiAgents/src/agents/faq/faq_agent.py",
    "AiAgents/src/agents/feedback/feedback_agent.py",
    "AiAgents/src/agents/handoff/handoff_agent.py",
    "AiAgents/src/graphs/single_agent_graph.py",
    "AiAgents/src/graphs/multi_agent_graph.py",
    "AiAgents/src/graphs/hierarchical_graph.py",
    "AiAgents/src/graphs/plan_execute_graph.py",
    "AiAgents/src/graphs/map_reduce_graph.py",
    "AiAgents/src/mcp_client/client.py",
    "AiAgents/src/mcp_client/config.py",
    "AiAgents/src/mcp_client/tool_adapter.py",
    "AiAgents/src/a2a/a2a_client.py",
    "AiAgents/src/memory/checkpointer.py",
    "AiAgents/src/memory/conversation_memory.py",
    "AiAgents/src/memory/long_term_memory.py",
    "AiAgents/src/prompts/prompt_templates.py",
    "AiAgents/src/prompts/few_shot.py",
    "AiAgents/src/prompts/chain_of_thought.py",
    "AiAgents/src/prompts/structured_output.py",
    "AiAgents/src/guardrails/input_validator.py",
    "AiAgents/src/guardrails/output_validator.py",
    "AiAgents/src/guardrails/action_validator.py",
    "AiAgents/src/guardrails/safety_config.py",
    "AiAgents/src/human_in_loop/interrupt_handler.py",
    "AiAgents/src/human_in_loop/approval_workflow.py",
    "AiAgents/src/utils/logger.py",
    "AiAgents/src/utils/tracing.py",

    # MCPServer
    "MCPServer/pyproject.toml",
    "MCPServer/src/config/settings.py",
    "MCPServer/src/servers/web_search/server.py",
    "MCPServer/src/servers/database/server.py",
    "MCPServer/src/servers/file_system/server.py",
    "MCPServer/src/servers/api_integration/server.py",
    "MCPServer/src/servers/calculator/server.py",
    "MCPServer/src/servers/weather/server.py",
    "MCPServer/src/backend_api/app.py",
    "MCPServer/src/backend_api/routes/search.py",
    "MCPServer/src/backend_api/routes/data.py",
    "MCPServer/src/backend_api/routes/files.py",
    "MCPServer/src/backend_api/middleware/auth.py",
    "MCPServer/src/backend_api/middleware/rate_limiter.py",
    "MCPServer/src/backend_api/middleware/logging_middleware.py",
    "MCPServer/src/backend_api/models/request_response.py",
    "MCPServer/src/shared/tool_registry.py",
    "MCPServer/src/shared/validators.py",
    "MCPServer/src/shared/error_handler.py",
    "MCPServer/src/gateway/unified_server.py",
    "MCPServer/src/gateway/a2a_server.py",
    "MCPServer/src/gateway/agent_card.py",

    # GenAISystem
    "GenAISystem/pyproject.toml",
    "GenAISystem/src/config/settings.py",
    "GenAISystem/src/ingestion/data_loader.py",
    "GenAISystem/src/ingestion/chunking/recursive_chunker.py",
    "GenAISystem/src/ingestion/chunking/semantic_chunker.py",
    "GenAISystem/src/ingestion/chunking/agentic_chunker.py",
    "GenAISystem/src/ingestion/chunking/document_chunker.py",
    "GenAISystem/src/ingestion/preprocessing/text_cleaner.py",
    "GenAISystem/src/ingestion/preprocessing/metadata_extractor.py",
    "GenAISystem/src/ingestion/pipeline.py",
    "GenAISystem/src/embeddings/embedding_factory.py",
    "GenAISystem/src/embeddings/batch_embedder.py",
    "GenAISystem/src/embeddings/embedding_cache.py",
    "GenAISystem/src/vectorstores/base_store.py",
    "GenAISystem/src/vectorstores/faiss_store.py",
    "GenAISystem/src/vectorstores/qdrant_store.py",
    "GenAISystem/src/vectorstores/chromadb_store.py",
    "GenAISystem/src/vectorstores/pinecone_store.py",
    "GenAISystem/src/vectorstores/store_factory.py",
    "GenAISystem/src/knowledge_graph/neo4j_client.py",
    "GenAISystem/src/knowledge_graph/entity_extractor.py",
    "GenAISystem/src/knowledge_graph/triplet_generator.py",
    "GenAISystem/src/knowledge_graph/graph_builder.py",
    "GenAISystem/src/knowledge_graph/graph_retriever.py",
    "GenAISystem/src/knowledge_graph/graph_visualizer.py",
    "GenAISystem/src/retrieval/vector_retriever.py",
    "GenAISystem/src/retrieval/keyword_retriever.py",
    "GenAISystem/src/retrieval/graph_retriever.py",
    "GenAISystem/src/retrieval/hybrid_retriever.py",
    "GenAISystem/src/retrieval/fusion.py",
    "GenAISystem/src/retrieval/reranker.py",
    "GenAISystem/src/retrieval/query_router.py",
    "GenAISystem/src/generation/response_generator.py",
    "GenAISystem/src/generation/citation_handler.py",
    "GenAISystem/src/generation/streaming_response.py",
    "GenAISystem/src/evaluation/ragas_evaluator.py",
    "GenAISystem/src/evaluation/custom_metrics.py",
    "GenAISystem/src/evaluation/evaluation_pipeline.py",
    "GenAISystem/src/agentic_rag/rag_graph.py",
    "GenAISystem/src/agentic_rag/self_rag.py",
    "GenAISystem/src/agentic_rag/corrective_rag.py",
    "GenAISystem/src/agentic_rag/adaptive_rag.py",
    "GenAISystem/src/api/rag_api.py",
    "GenAISystem/src/api/ingestion_api.py"
]

missing = []
present = []

for file in expected_files:
    if os.path.exists(os.path.join(project_dir, file)):
        present.append(file)
    else:
        missing.append(file)

print(f"Total Expected Files: {len(expected_files)}")
print(f"Present Files: {len(present)}")
print(f"Missing Files: {len(missing)}")
print("-" * 20)
print("MISSING FILES:")
for f in missing:
    print(f)

