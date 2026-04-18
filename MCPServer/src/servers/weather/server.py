"""
Weather MCP Server.
Provides tools to query weather data.
"""

import httpx
from mcp.server.fastmcp import FastMCP

from src.config.settings import settings

mcp = FastMCP("weather_server", description="Real-time weather data")


@mcp.tool()
async def get_current_weather(location: str) -> str:
    """
    Get the current weather for a specific location.

    Args:
        location: City name or zip code.
    """
    if not settings.weather_api_key:
        return "Error: Weather API key not configured."

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "http://api.weatherapi.com/v1/current.json",
                params={"key": settings.weather_api_key, "q": location},
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json()

            loc = data.get("location", {})
            cur = data.get("current", {})

            return (
                f"Weather for {loc.get('name')}, {loc.get('country')}:\n"
                f"Temperature: {cur.get('temp_c')}°C ({cur.get('temp_f')}°F)\n"
                f"Condition: {cur.get('condition', {}).get('text')}\n"
                f"Humidity: {cur.get('humidity')}%\n"
                f"Wind: {cur.get('wind_kph')} kph"
            )
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 400:
            return f"Error: Location '{location}' not found."
        return f"Weather API error: {e}"
    except Exception as e:
        return f"Failed to fetch weather: {e}"
