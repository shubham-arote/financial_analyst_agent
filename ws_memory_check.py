"""Server-path memory test: a 2-turn conversation over ONE ws/ask connection."""
import asyncio, json, httpx, websockets

BASE, WS = "http://127.0.0.1:8000", "ws://127.0.0.1:8000"


async def ask(ws, q):
    await ws.send(json.dumps({"question": q}))
    while True:
        ev = json.loads(await ws.recv())
        if ev["type"] == "agent_answer":
            return ev["answer"]
        if ev["type"] == "error":
            return "ERROR: " + ev["error"]


async def main():
    async with httpx.AsyncClient(timeout=60) as cx:
        doc_id = (await cx.post(f"{BASE}/load-sample")).json()["doc_id"]
    async with websockets.connect(f"{WS}/ws/parse/{doc_id}", max_size=None) as ws:
        while True:
            if json.loads(await ws.recv())["type"] in ("indexed", "error"):
                break
    # one connection == one conversation (server assigns a thread_id per connection)
    async with websockets.connect(f"{WS}/ws/ask/{doc_id}", max_size=None) as ws:
        a1 = await ask(ws, "What was the revenue in the Cloud segment?")
        print("Q1: What was the revenue in the Cloud segment?\n   ->", a1[:140])
        a2 = await ask(ws, "What about Hardware?")          # only resolvable via memory
        print("Q2: What about Hardware?   (follow-up)\n   ->", a2[:140])


asyncio.run(main())
