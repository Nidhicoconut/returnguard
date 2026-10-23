"""Expose ReturnGuard's tools over the Model Context Protocol (stdio).

Run:   python -m app.mcp_server
Then point any MCP client (Claude Desktop, MCP Inspector, langchain-mcp-adapters) at that command.
The web app calls the same functions in-process; this server shows they are protocol-ready.
"""
try:  # mcp 2.x renamed FastMCP to MCPServer
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from . import db, tools

db.init_db()
mcp = _Server("returnguard-tools")


@mcp.tool()
def lookup_order(order_id: str) -> dict:
    """Look up an order (product, category, price, days since delivery). Returns {} if not found."""
    return tools.lookup_order(order_id) or {}


@mcp.tool()
def retrieve_policy(query: str, k: int = 3) -> list[dict]:
    """Retrieve the return-policy clauses most relevant to a question."""
    return tools.retrieve_policy(query, k)


@mcp.tool()
def inspect_product_photo(image_path: str) -> dict:
    """Run the vision pipeline on a product photo and return defect evidence."""
    return tools.run_vision(image_path)


@mcp.tool()
def create_return_ticket(case_id: int, decision: str, summary: str) -> dict:
    """Create a ticket for a return case. decision is approve, reject or escalate."""
    return tools.create_ticket(case_id, decision, summary)


if __name__ == "__main__":
    mcp.run()
