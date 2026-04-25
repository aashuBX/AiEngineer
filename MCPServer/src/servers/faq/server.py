"""
FAQ MCP Server.
Provides static policy lookup and rule verification for the FAQ Agent.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("faq_server", description="FAQ and Policy Tools")

@mcp.tool()
def get_shipping_policy(region: str) -> str:
    """Gets shipping timelines and costs for a given region."""
    return f"Shipping policy for region {region} retrieved (stub)."

@mcp.tool()
def get_return_window(product_category: str) -> str:
    """Gets the return window policy for a specific product category."""
    return f"Return window for {product_category} is 30 days (stub)."

@mcp.tool()
def get_clinic_hours(clinic_id: str) -> str:
    """Retrieves operating hours for a medical clinic."""
    return f"Clinic hours for {clinic_id} retrieved (stub)."

@mcp.tool()
def get_insurance_accepted(insurance_name: str) -> str:
    """Checks if a specific insurance is accepted by the network."""
    return f"Insurance {insurance_name} acceptance status checked (stub)."

@mcp.tool()
def get_cancellation_policy() -> str:
    """Gets the session cancellation policy for personal coaching."""
    return "Cancellation policy retrieved (stub)."

@mcp.tool()
def get_equipment_requirements(workout_type: str) -> str:
    """Retrieves the list of equipment required for a coaching workout."""
    return f"Equipment requirements for {workout_type} retrieved (stub)."

@mcp.tool()
def get_fee_schedule(account_type: str) -> str:
    """Gets banking/financial fee schedules."""
    return f"Fee schedule for {account_type} retrieved (stub)."

@mcp.tool()
def get_wire_transfer_instructions(currency: str) -> str:
    """Gets routing and swift instructions for a wire transfer."""
    return f"Wire instructions for {currency} retrieved (stub)."

@mcp.tool()
def get_sla_timelines(issue_severity: str) -> str:
    """Gets expected resolution SLA for support issues."""
    return f"SLA for severity {issue_severity} retrieved (stub)."

@mcp.tool()
def get_enterprise_discount_tiers() -> str:
    """Gets sales discount tiers based on volume."""
    return "Enterprise discount tiers retrieved (stub)."
