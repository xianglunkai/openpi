"""Test client for RLT WebSocket server - validates output matches RL client format.

Usage:
    python scripts/test_rlt_client.py --host localhost --port 8000
"""

import argparse
import logging

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-steps", type=int, default=3)
    args = parser.parse_args()

    from openpi_client import websocket_client_policy

    logging.info(f"Connecting to ws://{args.host}:{args.port}")
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    metadata = client.get_server_metadata()
    logging.info(f"Server metadata: {metadata}")

    for step in range(args.num_steps):
        obs = {
            "images": {
                "base_0_rgb": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                "left_wrist_0_rgb": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                "right_wrist_0_rgb": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            },
            "state": np.random.randn(7).astype(np.float32),
            "prompt": "pick up the box",
        }

        logging.info(f"--- Step {step} ---")
        result = client.infer(obs)

        # Validate expected keys
        expected_keys = ["z_rl", "proprio", "ref_chunk"]
        for key in expected_keys:
            if key in result:
                val = result[key]
                logging.info(f"  {key}: shape={val.shape}, dtype={val.dtype}, range=[{val.min():.3f}, {val.max():.3f}]")
            else:
                logging.error(f"  {key}: MISSING!")

        # Validate shapes
        z_rl = result.get("z_rl")
        proprio = result.get("proprio")
        ref_chunk = result.get("ref_chunk")

        checks = []
        if z_rl is not None:
            checks.append(("z_rl shape", z_rl.shape == (2048,), f"expected (2048,), got {z_rl.shape}"))
        if proprio is not None:
            checks.append(("proprio shape", proprio.shape == (32,), f"expected (32,), got {proprio.shape}"))
        if ref_chunk is not None:
            checks.append(("ref_chunk shape", ref_chunk.shape == (50, 7), f"expected (10, 7), got {ref_chunk.shape}"))

        all_ok = True
        for name, ok, msg in checks:
            if ok:
                logging.info(f"  CHECK {name}: OK")
            else:
                logging.error(f"  CHECK {name}: FAIL - {msg}")
                all_ok = False

        timing = result.get("policy_timing", {})
        logging.info(f"  timing: {timing}")
        logging.info(f"  all keys: {list(result.keys())}")

        if all_ok:
            logging.info(f"  >>> Step {step}: ALL CHECKS PASSED")
        else:
            logging.error(f"  >>> Step {step}: SOME CHECKS FAILED")

    logging.info("Test complete!")


if __name__ == "__main__":
    main()
