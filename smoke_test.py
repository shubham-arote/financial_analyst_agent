"""End-to-end smoke test against a RUNNING server on 127.0.0.1:8000.
Mirrors exactly what the browser does: load sample -> WS parse -> WS ask.
"""
import asyncio, json, httpx, websockets

BASE = "http://127.0.0.1:8000"
WS = "ws://127.0.0.1:8000"


async def main():
    async with httpx.AsyncClient(timeout=60) as cx:
        h = (await cx.get(f"{BASE}/healthz")).json()
        s = (await cx.get(f"{BASE}/api/status")).json()
        print("healthz:", h)
        print("status :", s["provider"], "| cloud:", s["cloud"])

        r = (await cx.post(f"{BASE}/load-sample")).json()
        doc_id = r["doc_id"]
        print("loaded sample doc_id:", doc_id)

    # --- parse over WS (drain stage events) ---
    stages, blocks, indexed = [], 0, None
    async with websockets.connect(f"{WS}/ws/parse/{doc_id}", max_size=None) as ws:
        while True:
            ev = json.loads(await ws.recv())
            t = ev["type"]
            if t == "stage":
                stages.append(f'{ev["stage"]}:{ev["status"]}')
            elif t == "block":
                blocks += 1
            elif t == "indexed":
                indexed = ev["chunks"]
                break
            elif t == "error":
                print("PARSE ERROR:", ev); return
    print(f"parse: {blocks} blocks, stages={stages[:6]}..., index chunks={indexed}")

    # --- ask 3 questions over WS ---
    questions = [
        "What was the revenue?",
        "What is the operating margin?",
        "What is the capital of France?",   # out-of-scope -> should abstain
    ]
    async with websockets.connect(f"{WS}/ws/ask/{doc_id}", max_size=None) as ws:
        for q in questions:
            await ws.send(json.dumps({"question": q}))
            nodes = []
            while True:
                ev = json.loads(await ws.recv())
                t = ev["type"]
                if t == "agent_node":
                    nodes.append(ev.get("node") + (f'({ev.get("verdict")})' if ev.get("verdict") else ""))
                elif t == "agent_answer":
                    srcs = ev.get("sources", [])
                    cite = ", ".join(f'p{x.get("page")}' for x in srcs) or "(none)"
                    print(f"\nQ: {q}\n   nodes: {' -> '.join(nodes)}")
                    print(f"   A: {ev['answer'][:160]}")
                    print(f"   sources: {cite}")
                    break
                elif t == "error":
                    print(f"\nQ: {q}\n   ERROR: {ev['error']}"); break
    print("\nSMOKE TEST OK")


asyncio.run(main())
