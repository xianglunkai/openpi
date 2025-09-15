import json
from pathlib import Path
import numpy as np
import h5py
from tqdm import tqdm
import os

class NonIdleRangeExtractor:
    def __init__(self, hdf5_dir, keep_ranges_path, min_idle_len=7, min_non_idle_len=16, filter_last_n_in_ranges=10, static_delta_threshold=1e-3):
        self.hdf5_dir = os.path.expanduser(hdf5_dir)
        self.keep_ranges_path = os.path.expanduser(keep_ranges_path)
        self.min_idle_len = min_idle_len
        self.min_non_idle_len = min_non_idle_len
        self.filter_last_n_in_ranges = filter_last_n_in_ranges
        self.keep_ranges_map = {}
       
        self.hdf5_files = [str(p) for p in Path(self.hdf5_dir).glob("*.hdf5")]
        # print(f"Loading HDF5 files from {self.hdf5_files}")
        
        self.static_delta_threshold = static_delta_threshold  # Threshold to consider action change as static
        
        self._load_existing()

    def _load_existing(self):
        if Path(self.keep_ranges_path).exists():
            with Path(self.keep_ranges_path).open("r") as f:
                self.keep_ranges_map = json.load(f)
            print(f"Resuming from {len(self.keep_ranges_map)} episodes already processed")

    def extract(self):
        for hdf5_path in self.hdf5_files:
            with h5py.File(hdf5_path, "r") as f:
                key = f"{os.path.basename(hdf5_path)}"
                print(f"Processing episode: {key}")
                ep_group = f
                if key in self.keep_ranges_map:
                    continue
                actions = ep_group['action'][:]  # shape: (num_steps, num_joints)
                is_idle_array = np.hstack(
                    [np.array([False]), np.all(np.abs(actions[1:] - actions[:-1]) < self.static_delta_threshold, axis=1)]
                )
                is_idle_padded = np.concatenate([[False], is_idle_array, [False]])
                is_idle_diff = np.diff(is_idle_padded.astype(int))
                is_idle_true_starts = np.where(is_idle_diff == 1)[0]
                is_idle_true_ends = np.where(is_idle_diff == -1)[0]
                true_segment_masks = (is_idle_true_ends - is_idle_true_starts) >= self.min_idle_len
                is_idle_true_starts = is_idle_true_starts[true_segment_masks]
                is_idle_true_ends = is_idle_true_ends[true_segment_masks]
                keep_mask = np.ones(len(actions), dtype=bool)
                for start, end in zip(is_idle_true_starts, is_idle_true_ends, strict=True):
                    keep_mask[start:end] = False
                keep_padded = np.concatenate([[False], keep_mask, [False]])
                keep_diff = np.diff(keep_padded.astype(int))
                keep_true_starts = np.where(keep_diff == 1)[0]
                keep_true_ends = np.where(keep_diff == -1)[0]
                true_segment_masks = (keep_true_ends - keep_true_starts) >= self.min_non_idle_len
                keep_true_starts = keep_true_starts[true_segment_masks]
                keep_true_ends = keep_true_ends[true_segment_masks]
                self.keep_ranges_map[key] = []
                for start, end in zip(keep_true_starts, keep_true_ends, strict=True):
                    self.keep_ranges_map[key].append((int(start), int(end) - self.filter_last_n_in_ranges))
        print("Done!")
        with Path(self.keep_ranges_path).open("w") as f:
            json.dump(self.keep_ranges_map, f)

    def verify(self):
        with Path(self.keep_ranges_path).open("r") as f:
            self.keep_ranges_map = json.load(f)
        print(f"Saved {len(self.keep_ranges_map)} episodes with non-idle ranges")
        for hdf5_path in self.hdf5_files:
            with h5py.File(hdf5_path, "r") as f:
                key = f"{os.path.basename(hdf5_path)}"
                print(f"Verifying episode: {key}")
                ep_group = f
                if key not in self.keep_ranges_map:
                    print(f"No keep ranges for episode {key}")
                    continue
                actions = ep_group['action'][:]  # shape: (num_steps, num_joints)
                print(f"Total actions shape: {actions.shape}")
                keep_mask = np.zeros(actions.shape[0], dtype=bool)
                for start, end in self.keep_ranges_map[key]:
                    keep_mask[start:end] = True
                validate_actions = actions[keep_mask]
                print(f"Total kept actions shape for episode {key}: {validate_actions.shape}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract non-idle action ranges from HDF5 files.")
    parser.add_argument("--hdf5_dir", type=str, required=True, help="Directory containing HDF5 files.")
    parser.add_argument("--keep_ranges_path", type=str, required=True, help="Path to save the keep ranges JSON file.")
    parser.add_argument("--min_idle_len", type=int, default=7, help="Minimum length of idle segments to filter out.")
    parser.add_argument("--min_non_idle_len", type=int, default=16, help="Minimum length of non-idle segments to keep.")
    parser.add_argument("--filter_last_n_in_ranges", type=int, default=10, help="Number of frames to filter out at the end of each keep range.")
    parser.add_argument("--static_delta_threshold", type=float, default=1e-3, help="Threshold to consider action change as static.")
    args = parser.parse_args()
    
    extractor = NonIdleRangeExtractor(
        hdf5_dir=args.hdf5_dir,
        keep_ranges_path=args.keep_ranges_path,
        min_idle_len=args.min_idle_len,
        min_non_idle_len=args.min_non_idle_len,
        filter_last_n_in_ranges=args.filter_last_n_in_ranges,
        static_delta_threshold=args.static_delta_threshold
    )
    extractor.extract()
    extractor.verify()