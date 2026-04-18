"""Example 03: Human-in-the-Loop with interrupt points."""
from src.config.llm_providers import get_llm
from src.human_in_loop.interrupt_handler import InterruptHandler


def main():
    """Demonstrate human approval gates in agent workflows."""
    llm = get_llm(provider="groq", model="llama-3.3-70b-versatile")
    handler = InterruptHandler()

    print("Human-in-the-Loop Demo")
    print("Agent will pause for approval before executing sensitive actions.")
    print("=" * 50)

    # Simulate an interrupt
    action = {"tool": "send_email", "to": "user@example.com", "subject": "Important Update"}
    approved = handler.request_approval(action, reason="Agent wants to send an email")

    if approved:
        print("✓ Action approved by human. Proceeding...")
    else:
        print("✗ Action rejected by human. Aborting.")


if __name__ == "__main__":
    main()
