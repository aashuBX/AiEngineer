"""Example 05: Agent-to-Agent (A2A) communication."""
import asyncio
from src.a2a.a2a_client import A2AClient


async def main():
    """Demonstrate A2A protocol for inter-agent delegation."""
    print("A2A Communication Demo")
    print("=" * 50)

    client = A2AClient(base_url="http://localhost:8001")

    # Discover remote agent capabilities
    print("Discovering remote agents...")
    capabilities = await client.discover_agents()
    print(f"Available agents: {capabilities}")


if __name__ == "__main__":
    asyncio.run(main())
