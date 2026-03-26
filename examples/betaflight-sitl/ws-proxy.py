#!/usr/bin/env python3
"""WebSocket-to-TCP proxy for Betaflight Configurator PWA -> SITL.

Usage:
    python3 ws-proxy.py
    Then connect Configurator to: ws://127.0.0.1:5762
"""
import asyncio
import socket
import websockets

BF_HOST = "127.0.0.1"
BF_PORT = 5761
WS_PORT = 5762


async def proxy(ws):
    peer = ws.remote_address
    # Raw TCP socket (not asyncio) for reliable I/O
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    tcp.settimeout(2.0)
    try:
        tcp.connect((BF_HOST, BF_PORT))
    except OSError as e:
        print(f"[{peer}] Cannot reach SITL: {e}")
        return

    print(f"[{peer}] Connected (subprotocol={ws.subprotocol})")
    loop = asyncio.get_event_loop()
    closed = False

    async def ws_to_tcp():
        nonlocal closed
        try:
            async for msg in ws:
                if isinstance(msg, str):
                    msg = msg.encode()
                print(f"  WS->TCP {len(msg)}B: {msg[:20].hex()}")
                await loop.run_in_executor(None, tcp.sendall, msg)
        except Exception as e:
            print(f"  WS->TCP done: {e}")
        finally:
            closed = True

    async def tcp_to_ws():
        nonlocal closed
        try:
            while not closed:
                try:
                    data = await loop.run_in_executor(None, tcp.recv, 4096)
                except socket.timeout:
                    continue
                if not data:
                    print("  TCP closed by Betaflight")
                    break
                print(f"  TCP->WS {len(data)}B: {data[:20].hex()}")
                await ws.send(data)
        except Exception as e:
            print(f"  TCP->WS done: {e}")
        finally:
            closed = True

    try:
        await asyncio.gather(ws_to_tcp(), tcp_to_ws())
    finally:
        tcp.close()
        print(f"[{peer}] Disconnected")


def accept_subprotocol(connection, subprotocols):
    if subprotocols:
        return subprotocols[0]
    return None


async def main():
    async with websockets.serve(
        proxy,
        "127.0.0.1",
        WS_PORT,
        compression=None,
        max_size=None,
        ping_interval=None,
        select_subprotocol=accept_subprotocol,
    ) as server:
        for s in server.sockets:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"WS proxy ready: ws://127.0.0.1:{WS_PORT} -> tcp://{BF_HOST}:{BF_PORT}")
        await asyncio.Future()


asyncio.run(main())
