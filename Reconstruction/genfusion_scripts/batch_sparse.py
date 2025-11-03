import os
import sys
import threading
import subprocess
from pathlib import Path
from datetime import datetime
import logging
import time

base_dir = "./data_sparse"
experiment_name = "E5_sparse_mip_360"

# Parse result directory from command line if provided, otherwise use auto-increment
if len(sys.argv) >= 3 and sys.argv[2].startswith("--result_dir="):
    result_dir_base = sys.argv[2].split("=")[1]
else:
    # Get the latest baseline count by checking existing directories
    latest_count = 0
    for dirname in os.listdir("./output"):
        if dirname.startswith(experiment_name):
            try:
                count = int(dirname.split("_")[-1])
                latest_count = max(latest_count, count)
            except ValueError:
                continue

    result_dir_base = f"./output/{experiment_name}_{latest_count + 1}"

# Create logs directory if it doesn't exist
os.makedirs("./logs", exist_ok=True)
os.makedirs(f"./logs/{experiment_name}/details", exist_ok=True)
os.makedirs(result_dir_base, exist_ok=True)

# Set up logging
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"./logs/{experiment_name}/{current_time}.log"

# Configure logging to write to both file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
)

# Create a logger
logger = logging.getLogger(__name__)

# Create locks for each GPU
gpu_locks = {}


def run_task(gpu_id, hash_id):
    # Acquire lock for this GPU
    with gpu_locks[gpu_id]:
        # Set data_dir based on whether hash_id is in drone_hash_ids
        data_dir = f"{base_dir}/{hash_id}"

        result_dir = f"./{result_dir_base}/{hash_id}"
        os.makedirs(result_dir, exist_ok=True)

        # First command: training
        train_cmd = (
            f"CUDA_VISIBLE_DEVICES={gpu_id} python train.py "
            f"--data_dir {data_dir} "
            f"-m {result_dir} "
            f"--iterations 7000 "
            f"--test_iterations 7_000 "
            f"--diffusion_ckpt ./diffusion_ckpt/epoch=59-step=34000.ckpt "
            f"--diffusion_config ./generation_infer.yaml "
            f"--num_frames 16 "
            f"--outpaint_type sparse "
            f"--start_diffusion_iter 1000 "
            f"--depth_loss "
            f"--sparse_view 6 "
            f"--diffusion_resize_width 960 "
            f"--diffusion_resize_height 512 "
            f"--diffusion_crop_width 960 "
            f"--diffusion_crop_height 512 "
            f"--mono_depth "
            f"--repair "
            f"--densify_from_iter 100 "
            f"--diffusion_until 7000 "
            f"--diffusion_every 1000 "
            f"--densify_until_iter 5000 "
            f"--densification_interval 500 "
            f"--opacity_reset_interval 3100 "
            f"--lambda_dist 10.0 "
            f"--lambda_dssim 0.5 "
            f"--lambda_reg 1.0 "
            f"--unconditional_guidance_scale 3.2 "
        )
        train_cmd += f" --port {4500 + gpu_id}"

        logger.info(f"Starting training task on GPU {gpu_id} for {hash_id}")

        log_file_path = f"logs/{experiment_name}/details/{hash_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        try:
            # Execute training command
            with open(log_file_path, "w") as log_file:
                subprocess.run(
                    train_cmd, shell=True, check=True, stdout=log_file, stderr=log_file
                )

            logger.info(
                f"Successfully completed training on GPU {gpu_id} for {hash_id}"
            )

            # Second command: rendering
            render_cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python render.py -m {result_dir} --data_dir {data_dir}"
            logger.info(f"Starting rendering task on GPU {gpu_id} for {hash_id}")

            # Execute rendering command with the same log file
            with open(log_file_path, "a") as log_file:  # 'a' for append mode
                subprocess.run(
                    render_cmd, shell=True, check=True, stdout=log_file, stderr=log_file
                )

            logger.info(
                f"Successfully completed rendering on GPU {gpu_id} for {hash_id}"
            )

            # Third command: metrics
            metrics_cmd = (
                f"CUDA_VISIBLE_DEVICES={gpu_id} python metrics.py -m {result_dir}"
            )
            logger.info(f"Starting metrics calculation on GPU {gpu_id} for {hash_id}")

            # Execute metrics command with the same log file
            with open(log_file_path, "a") as log_file:
                subprocess.run(
                    metrics_cmd,
                    shell=True,
                    check=True,
                    stdout=log_file,
                    stderr=log_file,
                )

            logger.info(
                f"Successfully completed metrics calculation on GPU {gpu_id} for {hash_id}"
            )
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Task failed on GPU {gpu_id} for {hash_id}")
            return False


def main():
    if len(sys.argv) < 2:
        logger.error(
            "Usage: python batch_full_frame.py gpu_id1,gpu_id2,... [--result_dir=path]"
        )
        sys.exit(1)

    # Parse GPU IDs
    gpu_ids = [int(x) for x in sys.argv[1].split(",")]
    n_gpus = len(gpu_ids)

    # Initialize locks for each GPU
    for gpu_id in gpu_ids:
        gpu_locks[gpu_id] = threading.Lock()

    logger.info(f"Starting batch processing with GPUs: {gpu_ids}")

    # Get all hash directories from mip_360
    mip_360_dir = base_dir

    hash_ids = []
    if os.path.exists(mip_360_dir):
        hash_ids = [
            d
            for d in os.listdir(mip_360_dir)
            if os.path.isdir(os.path.join(mip_360_dir, d)) and d != "train"
        ]

    # Filter out hash_ids that already have results.json
    filtered_hash_ids = []
    for hash_id in hash_ids:
        result_dir = f"./{result_dir_base}/{hash_id}"
        result_file = os.path.join(result_dir, "results.json")
        if not os.path.exists(result_file):
            filtered_hash_ids.append(hash_id)
        else:
            logger.info(f"Skipping {hash_id} as results.json already exists")

    hash_ids = filtered_hash_ids
    if not hash_ids:
        logger.info("No tasks to process - all results.json files exist")
        sys.exit(0)
    # Track results
    results = {}

    # Create threads for each task
    threads = []
    for i, hash_id in enumerate(hash_ids):
        gpu_idx = i % n_gpus  # Cycle through available GPUs
        t = threading.Thread(
            target=lambda: results.update(
                {hash_id: run_task(gpu_ids[gpu_idx], hash_id)}
            )
        )
        threads.append(t)
        t.start()
        # Add a small delay between thread starts
        time.sleep(1)

    # Wait for all threads to complete
    for t in threads:
        t.join()

    # Log final results
    success_count = sum(1 for success in results.values() if success)
    logger.info(f"Completed all tasks. {success_count}/{len(hash_ids)} succeeded.")

    # Log failed tasks
    failed_hashes = [hash_id for hash_id, success in results.items() if not success]
    if failed_hashes:
        logger.warning("Failed hashes:")
        for hash_id in failed_hashes:
            logger.warning(f"- {hash_id}")


if __name__ == "__main__":
    main()
