from typing import Callable, Dict, Any, List

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, description: str, func: Callable, schema: Any = None):
        self._tools[name] = {
            "description": description,
            "func": func,
            "schema": schema
        }

    def get_tool(self, name: str):
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, str]]:
        return [{"name": name, "description": data["description"]} for name, data in self._tools.items()]

tool_registry = ToolRegistry()
