# ✅ Correct client.py
import base64
from mcp import ClientSession
from mcp.client.sse import sse_client

async def run():

    with open("758.jpg", "rb") as f:
        encoded_file = base64.b64encode(f.read()).decode("utf-8")
    # ✅ Wrap arguments inside "input" key
    arguments = {
        "input": {
            "filename": "6.jpg",
            "file_data": encoded_file
        }
    }

    async with sse_client(url="http://localhost:8000/sse") as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            result = await session.call_tool("car_model_connecter", arguments=arguments)
            print("Server Response:", result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(run())
