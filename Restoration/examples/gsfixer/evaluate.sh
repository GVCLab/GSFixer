export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=

python examples/gsfixer/evaluate_cogvideox_video_to_video_benchmark.py \
    --model_name PATH-to-CogVideoX-5b-I2V \
    --transformer_path PATH-to-GSFixer \
    --dinov2_ckpt PATH-to-dinov2-with-registers-large \
    --vggt_ckpt PATH-to-vggt-model.pt \
    --base_folder PATH-to-DL3DV_Res_benchmark \
    --ref_folders PATH-to-DL3DV_benchmark \
    --num_views 3 \
    --outpath "./output_gsfixer_DL3DV_Res_benchmark_results" \
    --scene_name "./examples/gsfixer/DL3DV-Res_scene_names.txt"