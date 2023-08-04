python inference.py \
    --prompt "A striking mallard floats effortlessly on the sparkling pond." \
    --condition "depth_midas" \
    --video_path "data/mallard-water.mp4" \
    --output_path "outputs/" \
    --video_length 15 \
    --smoother_steps 19 20 \
    --width 512 \
    --height 512 \
    --frame_rate 2 \
    --version v10 \
    # --is_long_video