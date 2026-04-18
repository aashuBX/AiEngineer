"""
CRM Agent — manages customer data queries using MCP Client tools
to interact with CRM systems (Hubspot, Salesforce, internal DB).
"""

from typing import Any

from langchain_core.messages import AIMessage

from src.agents.base_agent import BaseAgent
from src.models.schemas import AgentConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

_CRM_SYSTEM_PROMPT = """You are a CRM Data Agent with access to customer relationship management tools.

You can:
- Look up customer profiles, order history, and account status
- Retrieve lead and opportunity data
- Update customer records when explicitly authorized
- Summarize customer interaction history

Guidelines:
- Always verify customer identity context before sharing sensitive data
- Summarize data clearly without exposing raw database IDs unnecessarily
- If data is not found, clearly state that rather than guessing
- Format monetary values with proper currency symbols
- Present dates in a human-readable format
"""


class CrmAgent(BaseAgent):
    """Handles CRM queries using MCP tools for data retrieval and updates."""

    def __init__(self):
        super().__init__(
            config=AgentConfig(
                name="CrmAgent",
                description="Manages CRM data access and customer profile queries via MCP tools",
                temperature=0.0,
            )
        )

    @property
    def system_prompt(self) -> str:
        return _CRM_SYSTEM_PROMPT

    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        messages = state.get("messages", [])
        user_query = ""
        for msg in reversed(messages):
            if getattr(msg, "type", "") == "human":
                user_query = msg.content
                break

        if not user_query:
            return {**state, "task_status": "error", "error": "No user query found"}

        logger.info(f"CrmAgent: processing CRM query: {user_query!r}")

        # Use bound MCP tools if available, else fall back to direct LLM
        from src.config.llm_providers import get_default_llm
        llm = get_default_llm()

        # Bind registered MCP tools if any
        if self._tools:
            llm = llm.bind_tools(self._tools)

        prompt_msgs = self.build_messages(user_query, history=messages[:-1])

        try:
            response = await llm.ainvoke(prompt_msgs)

            # Handle tool calls if the LLM wants to use CRM tools
            tool_results = []
            if hasattr(response, "tool_calls") and response.tool_calls:
                from langchain_core.messages import ToolMessage
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_input = tool_call["args"]
                    logger.info(f"CrmAgent invoking tool: {tool_name}({tool_input})")

                    # Find and invoke the tool
                    result_content = f"[Tool {tool_name} result placeholder]"
                    for tool in self._tools:
                        if tool.name == tool_name:
                            try:
                                result_content = await tool.arun(tool_input)
                            except Exception as te:
                                result_content = f"Error: {te}"
                            break

                    tool_results.append({"tool": tool_name, "result": result_content})

                # Get final response after tool use
                final_response = await llm.ainvoke(
                    prompt_msgs + [response, ToolMessage(content=str(tool_results), tool_call_id="crm")]
                )
                answer = final_response.content
            else:
                answer = response.content

            return {
                **state,
                "messages": [AIMessage(content=answer)],
                "task_status": "done",
                "current_agent": "crm",
                "tool_results": tool_results,
            }
        except Exception as e:
            logger.error(f"CrmAgent failed: {e}")
            return {**state, "task_status": "error", "error": str(e)}
