"""Test batch inference for RLT WebSocket server.

Compares:
  1. N sequential single requests
  2. 1 batch request with N observations

Usage:
    python scripts/test_rlt_batch.py --host localhost --port 8000 --batch-size 10
"""

import argparse
import logging
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def make_fake_obs():
    return {
        "images": {
            "base_0_rgb": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "left_wrist_0_rgb": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "right_wrist_0_rgb": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        },
        "state": np.random.randn(7).astype(np.float32),
        "prompt": "pick up the box",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2, help="Warmup requests before timing")
    args = parser.parse_args()

    from openpi_client import websocket_client_policy

    logging.info(f"Connecting to ws://{args.host}:{args.port}")
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    metadata = client.get_server_metadata()
    logging.info(f"Server metadata: {metadata}")
    supports_batch = metadata.get("supports_batch", False)
    logging.info(f"Server supports batch: {supports_batch}")

    B = args.batch_size
    obs_list = [make_fake_obs() for _ in range(B)]

    # Warmup (JIT compilation)
    logging.info(f"Warming up with {args.warmup} single requests...")
    for i in range(args.warmup):
        client.infer(obs_list[0])
    logging.info("Warmup done")

    # Test 1: Sequential single requests
    logging.info(f"\n=== Test 1: {B} sequential single requests ===")
    t0 = time.monotonic()
    single_results = []
    for i in range(B):
        result = client.infer(obs_list[i])
        single_results.append(result)
    t_sequential = time.monotonic() - t0
    logging.info(f"Sequential: {B} requests in {t_sequential*1000:.1f}ms ({t_sequential/B*1000:.1f}ms/sample)")

    # Validate single results
    for i, r in enumerate(single_results):
        assert "z_rl" in r, f"Single result {i} missing z_rl"
        assert r["z_rl"].shape == (2048,), f"Single result {i} z_rl shape {r['z_rl'].shape}"
        assert r["ref_chunk"].shape[1] == 7, f"Single result {i} ref_chunk shape {r['ref_chunk'].shape}"
    logging.info(f"  All {B} single results validated OK")

    # Test 2: Batch request
    if supports_batch:
        logging.info(f"\n=== Test 2: 1 batch request with {B} observations ===")
        batch_payload = {"batch": obs_list}
        t0 = time.monotonic()
        batch_response = client.infer(batch_payload)
        t_batch = time.monotonic() - t0

        batch_results = batch_response["batch_results"]
        total_ms = batch_response.get("total_infer_ms", 0)
        per_ms = batch_response.get("per_sample_infer_ms", 0)
        logging.info(f"Batch: {B} samples in {t_batch*1000:.1f}ms ({t_batch/B*1000:.1f}ms/sample)")
        logging.info(f"  Server reported: total={total_ms:.1f}ms, per_sample={per_ms:.1f}ms")

        # Validate batch results
        assert len(batch_results) == B, f"Expected {B} results, got {len(batch_results)}"
        for i, r in enumerate(batch_results):
            assert "z_rl" in r, f"Batch result {i} missing z_rl"
            assert r["z_rl"].shape == (2048,), f"Batch result {i} z_rl shape {r['z_rl'].shape}"
            assert r["ref_chunk"].shape[1] == 7, f"Batch result {i} ref_chunk shape {r['ref_chunk'].shape}"
        logging.info(f"  All {B} batch results validated OK")

        # Compare
        speedup = t_sequential / t_batch
        logging.info("\n=== Comparison ===")
        logging.info(f"Sequential: {t_sequential*1000:.1f}ms total, {t_sequential/B*1000:.1f}ms/sample")
        logging.info(f"Batch:      {t_batch*1000:.1f}ms total, {t_batch/B*1000:.1f}ms/sample")
        logging.info(f"Speedup:    {speedup:.1f}x")

        # Verify results are consistent (same input should give same output)
        for i in range(B):
            z_diff = np.abs(single_results[i]["z_rl"] - batch_results[i]["z_rl"]).max()
            ref_diff = np.abs(single_results[i]["ref_chunk"] - batch_results[i]["ref_chunk"]).max()
            logging.info(f"  Sample {i}: z_rl max_diff={z_diff:.6f}, ref_chunk max_diff={ref_diff:.6f}")
    else:
        logging.warning("Server does not support batch inference. Skipping batch test.")

    logging.info("\nTest complete!")


if __name__ == "__main__":
    main()
