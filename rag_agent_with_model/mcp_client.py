# ✅ Correct client.py
import base64
from mcp import ClientSession
from mcp.client.sse import sse_client



SELECT_IMG = "758.jpg"
async def run():

    with open(SELECT_IMG, "rb") as f:
        encoded_file = base64.b64encode(f.read()).decode("utf-8")
    # ✅ Wrap arguments inside "input" key
    arguments = {
        "input": {
            "filename": SELECT_IMG,
            "file_data": encoded_file
        }
    }

    async with sse_client(url="http://localhost:8000/sse") as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            result = await session.call_tool("car_model_connecter", arguments=arguments)
            # print(encoded_file)
            print("Server Response:", result.content[0].text)

if __name__ == "__main__":
    import asyncio
    asyncio.run(run())
