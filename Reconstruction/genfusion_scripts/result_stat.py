import os
import json
import numpy as np


def calculate_metrics_stats(base_path):
    # Initialize lists to store metrics and folder names
    ssim_values = []
    psnr_values = []
    lpips_values = []
    folder_names = []

    # Walk through all subdirectories
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file == "results.json":
                json_path = os.path.join(root, file)
                try:
                    with open(json_path, "r") as f:
                        data = json.load(f)

                    # Get folder name
                    folder_name = os.path.basename(root)
                    folder_names.append(folder_name)

                    # Get the first (and should be only) key in the results dict
                    test_key = list(data.keys())[0]
                    metrics = data[test_key]

                    ssim_values.append(metrics["SSIM"])
                    psnr_values.append(metrics["PSNR"])
                    lpips_values.append(metrics["LPIPS"])
                except Exception as e:
                    print(f"Error reading {json_path}: {str(e)}")
                    continue

    # Calculate statistics
    stats = {
        "SSIM": {"mean": np.mean(ssim_values), "values": ssim_values},
        "PSNR": {"mean": np.mean(psnr_values), "values": psnr_values},
        "LPIPS": {"mean": np.mean(lpips_values), "values": lpips_values},
        "folders": folder_names,
        "num_samples": len(ssim_values),
    }

    return stats


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python result_stat.py <base_path>")
        sys.exit(1)

    base_path = sys.argv[1]
    stats = calculate_metrics_stats(base_path)

    print("\nResults Statistics:")
    print(f"Number of samples: {stats['num_samples']}")

    # Print header
    print("\n{:<30} {:>10} {:>10} {:>10}".format("Folder", "PSNR", "SSIM", "LPIPS"))
    print("-" * 62)

    # Print each row
    for i in range(stats["num_samples"]):
        print(
            "{:<30} {:>10.4f} {:>10.4f} {:>10.4f}".format(
                stats["folders"][i],
                stats["PSNR"]["values"][i],
                stats["SSIM"]["values"][i],
                stats["LPIPS"]["values"][i],
            )
        )

    # Print means at the bottom
    print("-" * 62)
    print(
        "{:<30} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            "Mean", stats["PSNR"]["mean"], stats["SSIM"]["mean"], stats["LPIPS"]["mean"]
        )
    )
