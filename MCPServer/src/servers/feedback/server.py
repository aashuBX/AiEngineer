"""
Feedback MCP Server.
Provides data ingestion tools for the Feedback Agent.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("feedback_server")

@mcp.tool()
def submit_product_review(product_id: str, rating: int, review: str) -> str:
    """Submits a product review for e-commerce."""
    return f"Product review for {product_id} logged with rating {rating} (stub)."

@mcp.tool()
def submit_visit_survey(patient_id: str, satisfaction_score: int, comments: str) -> str:
    """Logs a medical patient satisfaction survey."""
    return f"Visit survey for {patient_id} logged (stub)."

@mcp.tool()
def log_session_feedback(client_id: str, difficulty_rating: int) -> str:
    """Logs post-workout session feedback for a coaching client."""
    return f"Session feedback for {client_id} logged with difficulty {difficulty_rating} (stub)."

@mcp.tool()
def submit_app_feedback(user_id: str, bug_report: bool, description: str) -> str:
    """Submits app feedback or bug reports for finance users."""
    return f"App feedback for {user_id} logged (Bug: {bug_report}) (stub)."

@mcp.tool()
def log_agent_csat(call_id: str, score: int) -> str:
    """Logs Customer Satisfaction (CSAT) score for a support call."""
    return f"CSAT score {score} logged for call {call_id} (stub)."

@mcp.tool()
def log_demo_feedback(lead_email: str, interested: bool, blockers: str) -> str:
    """Logs post-demo feedback for a sales prospect."""
    return f"Demo feedback for {lead_email} logged (Interested: {interested}) (stub)."
