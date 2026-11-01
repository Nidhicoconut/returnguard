"""Expose ReturnGuard's tools over the Model Context Protocol (stdio).

Run:   python -m app.mcp_server
Then point any MCP client (Claude Desktop, MCP Inspector, langchain-mcp-adapters) at that command.
The web app calls the same functions in-process; this server shows they are protocol-ready.
"""
try:  # mcp 2.x renamed FastMCP to MCPServer
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from . import db, preanalysis, tools, vision

db.init_db()
mcp = _Server("returnguard-tools")


@mcp.tool()
def lookup_order(order_id: str) -> dict:
    """Look up a clothing order (product, price, days since delivery, inside the return window). {} if not found."""
    return tools.lookup_order(order_id) or {}


@mcp.tool()
def parse_claim(reason: str) -> dict:
    """Turn a customer's return reason into a structured claim (defect type, location, timing, resolution)."""
    return tools.parse_claim(reason)


@mcp.tool()
def check_photo_quality(image_path: str) -> dict:
    """Quality gate for one photo: resolution, sharpness and exposure, with an instruction if it fails."""
    import cv2

    return preanalysis.assess_quality(cv2.imread(image_path, cv2.IMREAD_COLOR))


@mcp.tool()
def inspect_garment_photo(image_path: str, zoom: bool = False) -> dict:
    """Detect holes and stains on a garment photo. zoom=True re-runs on four zoomed tiles."""
    return vision.analyze(image_path, category="clothing", tiles=zoom)


@mcp.tool()
def retrieve_policy(query: str, k: int = 3) -> list[dict]:
    """Retrieve the clothing return-policy clauses most relevant to a question."""
    return tools.retrieve_policy(query, k)


@mcp.tool()
def create_return_ticket(case_id: int, decision: str, summary: str) -> dict:
    """Create a ticket for a return case. decision is AUTO_APPROVE, REQUEST_MORE_EVIDENCE or ESCALATE_TO_HUMAN."""
    return tools.create_ticket(case_id, decision, summary)


if __name__ == "__main__":
    mcp.run()
