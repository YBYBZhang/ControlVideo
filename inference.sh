python inference.py \
    --prompt "A striking mallard floats effortlessly on the sparkling pond." \
    --condition "depth" \
    --video_path "data/mallard-water.mp4" \
    --output_path "outputs/" \
    --video_length 15 \
    --smoother_steps 19 20 \
    --width 512 \
    --height 512 \
    # --is_long_video