from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
import mcp.types as types

# Yeteneklerimizi import edelim
from radar_skills import RADAR_SKILL_DESCRIPTIONS, generate_synthetic_radar_data

server = Server("fmcw-radar-mcp-server")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    tools = []
    for skill in RADAR_SKILL_DESCRIPTIONS:
        tools.append(
            types.Tool(
                name=skill["name"],
                description=skill["description"],
                inputSchema=skill["inputSchema"]
            )
        )
    return tools

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "generate_synthetic_radar_data":
        target_class = arguments.get("target_class")
        sigma_p = arguments.get("sigma_p", 0.20)
        num_samples = arguments.get("num_samples", 1)
        output_filename = arguments.get("output_filename", "synthetic_data.json")
        
        result = generate_synthetic_radar_data(target_class, sigma_p, num_samples, output_filename)
        return [types.TextContent(type="text", text=result)]
    
    raise ValueError(f"Bilinmeyen yetenek: {name}")

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="fmcw-radar-mcp-server",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())