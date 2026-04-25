"""
Handoff MCP Server.
Provides escalation and routing tools for the Handoff Agent.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("handoff_server", description="Handoff and Escalation Tools")

@mcp.tool()
def route_to_returns_specialist(order_id: str, urgency: str) -> str:
    """Routes an e-commerce customer to a returns specialist."""
    return f"Order {order_id} routed to returns specialist with urgency {urgency} (stub)."

@mcp.tool()
def page_on_call_doctor(patient_id: str, symptoms: str) -> str:
    """Pages the on-call doctor for urgent medical issues."""
    return f"On-call doctor paged for patient {patient_id} (stub)."

@mcp.tool()
def request_custom_plan_review(client_id: str) -> str:
    """Requests the coach to manually review a client's plan."""
    return f"Custom plan review requested for coach regarding client {client_id} (stub)."

@mcp.tool()
def escalate_to_fraud_team(account_id: str, transaction_id: str) -> str:
    """Escalates a suspicious finance case to the fraud team."""
    return f"Account {account_id} escalated to fraud team for transaction {transaction_id} (stub)."

@mcp.tool()
def transfer_call_to_tier2(call_id: str, context: str) -> str:
    """Transfers a support call to Tier 2 technical support."""
    return f"Call {call_id} transferred to Tier 2 (stub)."

@mcp.tool()
def assign_to_account_executive(lead_email: str, company_size: int) -> str:
    """Assigns a highly qualified lead to an Account Executive."""
    return f"Lead {lead_email} assigned to AE (stub)."
