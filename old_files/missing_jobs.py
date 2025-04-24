import os
from itertools import product

# === Parameters for your jobs ===
datasets = [
    "barbell_2d", "sine_wave_2d", "star_2d", "radial_segment_2d",
    "rectangle_2d", "barbell_3d", "radial_segment_3d", "saddle_3d"
]
noise_folders = ["fuzziness_000", "fuzziness_003", "fuzziness_005", "fuzziness_007"]
models = ["hhcart_a", "hhcart_d", "randcart", "oc1", "wodt", "co2", "cart", "ridge_cart"]
seeds = [0, 1, 2]

# === Construct all expected filenames ===
expected_filenames = {
    f"{ds}_{noise}_{model}_seed{seed}.csv"
    for ds, noise, seed, model in product(datasets, noise_folders, seeds, models)
}

# === Scan the actual output folder ===
base_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "_data", "depth_sweep_batch_results", "delftblue_fuzziness_runs")
)
print(f"📂 Checking folder: {base_dir}")

if not os.path.isdir(base_dir):
    raise RuntimeError(f"Output folder not found: {base_dir}")

found_filenames = set(f for f in os.listdir(base_dir) if f.endswith(".csv"))

# === Compare ===
missing = expected_filenames - found_filenames

# === Report ===
print(f"\n✅ Found {len(found_filenames)} / {len(expected_filenames)} files")
print(f"❌ Missing {len(missing)} files:\n")

for name in sorted(missing):
    print(name)

# === Save missing jobs as CSV-style lines ===
# === Save missing jobs in comma-separated format ===
missing_path = os.path.join(os.path.dirname(__file__), "missing_jobs.txt")
with open(missing_path, "w") as f:
    for filename in sorted(missing):
        name = filename.replace(".csv", "")
        if "_seed" not in name:
            print(f"[!] Skipping malformed filename (no seed): {filename}")
            continue
        parts = name.split("_")
        try:
            # Dataset always ends with _2d or _3d
            dataset_parts = []
            for i, part in enumerate(parts):
                dataset_parts.append(part)
                if part.endswith("2d") or part.endswith("3d"):
                    break
            dataset = "_".join(dataset_parts)

            folder = "_".join(parts[i+1:i+3])        # fuzziness_XXX
            model = "_".join(parts[i+3:-1])          # can be hhcart_a or similar
            seed = parts[-1].replace("seed", "")     # seed number

            f.write(f"{dataset},{folder},{seed},{model}\n")
        except Exception as e:
            print(f"[!] Could not parse filename: {filename} ({e})")

print(f"\n📄 Saved missing filenames to: {missing_path}")
