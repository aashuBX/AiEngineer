"""
Run All MCP Servers — Start the unified gateway and backend API.
"""

import subprocess
import sys
import os
import signal
import time

# Colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

processes = []


def start_process(name: str, cmd: list, port: int = None):
    """Start a subprocess and track it."""
    port_info = f" (port {port})" if port else ""
    print(f"{GREEN}Starting {name}{port_info}...{RESET}")
    proc = subprocess.Popen(
        cmd,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    processes.append((name, proc))
    return proc


def shutdown(signum=None, frame=None):
    """Gracefully shut down all processes."""
    print(f"\n{YELLOW}Shutting down all servers...{RESET}")
    for name, proc in processes:
        try:
            proc.terminate()
            proc.wait(timeout=5)
            print(f"  {GREEN}✓{RESET} {name} stopped")
        except subprocess.TimeoutExpired:
            proc.kill()
            print(f"  {RED}✗{RESET} {name} killed")
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print(f"\n{GREEN}{'='*50}")
    print("  AI Engineer — MCP Server Launcher")
    print(f"{'='*50}{RESET}\n")

    # 1. Start the FastAPI backend
    start_process(
        "Backend API",
        [sys.executable, "-m", "uvicorn", "src.backend_api.app:app", "--host", "0.0.0.0", "--port", "8001", "--reload"],
        port=8001,
    )

    # 2. Start the unified MCP gateway (stdio mode — for direct agent connections)
    start_process(
        "Unified MCP Gateway",
        [sys.executable, "-m", "src.gateway.unified_server"],
    )

    print(f"\n{GREEN}All servers started!{RESET}")
    print(f"  Backend API:  http://localhost:8001")
    print(f"  API Docs:     http://localhost:8001/docs")
    print(f"  MCP Gateway:  stdio (use MCP Inspector to test)")
    print(f"\n{YELLOW}Press Ctrl+C to stop all servers{RESET}\n")

    # Keep main process alive
    try:
        while True:
            # Check if any process died unexpectedly
            for name, proc in processes:
                if proc.poll() is not None:
                    print(f"{RED}✗ {name} exited with code {proc.returncode}{RESET}")
            time.sleep(2)
    except KeyboardInterrupt:
        shutdown()


if __name__ == "__main__":
    main()
