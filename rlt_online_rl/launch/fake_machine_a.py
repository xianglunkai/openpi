#!/usr/bin/env python3
"""
cd /path/to/rlt_online_rl
conda activate rlt_online_rl310
python launch/fake_machine_a.py
"""

from __future__ import annotations

import numpy as np
from openpi_client import msgpack_numpy
from websockets.sync.server import serve

HOST = "127.0.0.1"
PORT = 8000
Z_DIM = 2048
PROPRIO_DIM = 7
CHUNK_LEN = 10
ACTION_DIM = 7

# Small absolute-action nudge that does not return to the start pose,
# so reset behavior is easy to observe on the robot.
MOVE_OFFSET = np.array([0.02, 0.02, 0.02, 0.02, -0.02, -0.02, 0.0], dtype=np.float32)
MOVE_PROFILE = np.linspace(0.2, 1.0, CHUNK_LEN, dtype=np.float32)

packer = msgpack_numpy.Packer()


def build_payload(observation: dict) -> dict:
    state = np.asarray(observation.get("state", []), dtype=np.float32).reshape(-1)

    z_rl = np.zeros((Z_DIM,), dtype=np.float32)

    proprio = np.zeros((PROPRIO_DIM,), dtype=np.float32)
    if state.size > 0:
        n = min(PROPRIO_DIM, state.size)
        proprio[:n] = state[:n]

    base_action = np.zeros((ACTION_DIM,), dtype=np.float32)
    if state.size > 0:
        n = min(ACTION_DIM, state.size)
        base_action[:n] = state[:n]

    ref_chunk = np.repeat(base_action[None, :], CHUNK_LEN, axis=0)
    ref_chunk += MOVE_PROFILE[:, None] * MOVE_OFFSET[None, :]

    return {
        "z_rl": z_rl,
        "proprio": proprio,
        "ref_chunk": ref_chunk,
    }


def handler(ws) -> None:
    ws.send(packer.pack({"server": "fake-machine-a", "mode": "nudge-no-return"}))
    while True:
        try:
            raw = ws.recv()
        except Exception:
            return
        observation = msgpack_numpy.unpackb(raw)
        ws.send(packer.pack(build_payload(observation)))


def main() -> None:
    print(f"[fake_a] serving ws://{HOST}:{PORT}", flush=True)
    with serve(handler, HOST, PORT, max_size=None, compression=None) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
