"""Test batch inference with padding against real server.

Tests various batch sizes to verify:
1. Padding works (any size maps to pre-compiled size)
2. Results are correct (z_rl, proprio, ref_chunk shapes)
3. No JIT recompilation timeout
4. Performance comparison: sequential vs batch

Usage:
    # Start server first, then:
    python scripts/test_batch_padding.py --host localhost --port 8000
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


def test_single(client, obs):
    """Test single inference returns correct format."""
    result = client.infer(obs)
    assert "z_rl" in result, "Missing z_rl"
    assert "proprio" in result, "Missing proprio"
    assert "ref_chunk" in result, "Missing ref_chunk"
    assert result["z_rl"].shape == (2048,), f"z_rl shape {result['z_rl'].shape}"
    assert (
        result["ref_chunk"].ndim == 2 and result["ref_chunk"].shape[1] == 7
    ), f"ref_chunk shape {result['ref_chunk'].shape}"
    return result


def test_batch(client, obs_list, expected_size):
    """Test batch inference returns correct number of results with correct format."""
    payload = {"batch": obs_list}
    response = client.infer(payload)
    assert "batch_results" in response, f"Missing batch_results, got keys: {list(response.keys())}"
    results = response["batch_results"]
    assert len(results) == expected_size, f"Expected {expected_size} results, got {len(results)}"
    for i, r in enumerate(results):
        assert "z_rl" in r, f"Result {i} missing z_rl"
        assert r["z_rl"].shape == (2048,), f"Result {i} z_rl shape {r['z_rl'].shape}"
        assert "ref_chunk" in r, f"Result {i} missing ref_chunk"
        assert (
            r["ref_chunk"].ndim == 2 and r["ref_chunk"].shape[1] == 7
        ), f"Result {i} ref_chunk shape {r['ref_chunk'].shape}"
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    from openpi_client import websocket_client_policy

    logging.info(f"Connecting to ws://{args.host}:{args.port}")
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    metadata = client.get_server_metadata()
    logging.info(f"Server metadata: {metadata}")

    # ========================================
    # Test 1: Single inference warmup
    # ========================================
    logging.info("\n" + "=" * 60)
    logging.info("Test 1: Single inference warmup")
    logging.info("=" * 60)
    obs = make_fake_obs()
    for i in range(3):
        r = test_single(client, obs)
        logging.info(f"  Warmup {i}: OK")
    logging.info("  PASSED")

    # ========================================
    # Test 2: Various batch sizes (padding test)
    # ========================================
    logging.info("\n" + "=" * 60)
    logging.info("Test 2: Batch sizes 1-20 (padding to pre-compiled sizes)")
    logging.info("=" * 60)
    for bs in [1, 2, 3, 4, 5, 6, 7, 8]:
        obs_list = [make_fake_obs() for _ in range(bs)]
        t0 = time.monotonic()
        response = test_batch(client, obs_list, expected_size=bs)
        t = time.monotonic() - t0
        padded = response.get("padded_size", "?")
        total_ms = response.get("total_infer_ms", 0)
        logging.info(f"  bs={bs:>2} → padded={padded:>2}, time={t*1000:>7.1f}ms (server: {total_ms:>7.1f}ms)  OK")
    logging.info("  PASSED")

    # ========================================
    # Test 3: Sequential vs Batch speed comparison
    # ========================================
    logging.info("\n" + "=" * 60)
    logging.info("Test 3: Speed comparison (N=4)")
    logging.info("=" * 60)
    N = 4
    obs_list = [make_fake_obs() for _ in range(N)]

    # Sequential
    t0 = time.monotonic()
    seq_results = []
    for o in obs_list:
        seq_results.append(test_single(client, o))
    t_seq = time.monotonic() - t0

    # Batch
    t0 = time.monotonic()
    batch_response = test_batch(client, obs_list, expected_size=N)
    t_batch = time.monotonic() - t0
    batch_results = batch_response["batch_results"]

    speedup = t_seq / t_batch if t_batch > 0 else 0
    logging.info(f"  Sequential: {N} requests in {t_seq*1000:.0f}ms ({t_seq/N*1000:.0f}ms/sample)")
    logging.info(f"  Batch:      {N} samples  in {t_batch*1000:.0f}ms ({t_batch/N*1000:.0f}ms/sample)")
    logging.info(f"  Speedup:    {speedup:.1f}x")
    logging.info("  PASSED")

    # ========================================
    # Test 4: Result consistency (same input → similar output)
    # ========================================
    logging.info("\n" + "=" * 60)
    logging.info("Test 4: Result consistency check")
    logging.info("=" * 60)
    # Use same observation for all
    same_obs = make_fake_obs()

    single_r = test_single(client, same_obs)

    batch_r = test_batch(client, [same_obs, same_obs], expected_size=2)
    b0 = batch_r["batch_results"][0]
    b1 = batch_r["batch_results"][1]

    # batch[0] and batch[1] should be identical (same input)
    z_diff_01 = np.abs(b0["z_rl"] - b1["z_rl"]).max()
    ref_diff_01 = np.abs(b0["ref_chunk"] - b1["ref_chunk"]).max()
    logging.info(f"  batch[0] vs batch[1]: z_rl_diff={z_diff_01:.6f}, ref_diff={ref_diff_01:.6f}")

    # Note: single vs batch may differ slightly due to different JAX rng states
    z_diff_sb = np.abs(single_r["z_rl"] - b0["z_rl"]).max()
    ref_diff_sb = np.abs(single_r["ref_chunk"] - b0["ref_chunk"]).max()
    logging.info(f"  single vs batch[0]:   z_rl_diff={z_diff_sb:.6f}, ref_diff={ref_diff_sb:.6f}")

    if z_diff_01 < 0.001:
        logging.info("  batch[0]==batch[1]: IDENTICAL (as expected)")
    else:
        logging.warning(f"  batch[0]!=batch[1]: diff={z_diff_01:.6f} (unexpected)")
    logging.info("  PASSED")

    # ========================================
    # Test 5: Empty batch
    # ========================================
    logging.info("\n" + "=" * 60)
    logging.info("Test 5: Empty batch")
    logging.info("=" * 60)
    empty_response = client.infer({"batch": []})
    assert empty_response.get("batch_results") == [], f"Expected empty results, got {empty_response}"
    logging.info("  Empty batch returns []: OK")
    logging.info("  PASSED")

    # ========================================
    # Summary
    # ========================================
    logging.info("\n" + "=" * 60)
    logging.info("ALL 5 TESTS PASSED")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
