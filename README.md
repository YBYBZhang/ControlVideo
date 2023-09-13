# ControlVideo

Official pytorch implementation of "ControlVideo: Training-free Controllable Text-to-Video Generation"

[![arXiv](https://img.shields.io/badge/arXiv-2305.13077-b31b1b.svg)](https://arxiv.org/abs/2305.13077)
[![Project](https://img.shields.io/badge/Project-Website-orange)](https://controlvideov1.github.io/)
[![HuggingFace demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Yabo/ControlVideo)
[![Replicate](https://replicate.com/cjwbw/controlvideo/badge)](https://replicate.com/cjwbw/controlvideo) 
![visitors](https://visitor-badge.laobi.icu/badge?page_id=YBYBZhang/ControlVideo)

<p align="center">
<img src="assets/overview.png" width="1080px"/> 
<br>
<em>ControlVideo adapts ControlNet to the video counterpart without any finetuning, aiming to directly inherit its high-quality and consistent generation </em>
</p>

## News
* [07/16/2023] Add [HuggingFace demo](https://huggingface.co/spaces/Yabo/ControlVideo)!
* [07/11/2023] Support [ControlNet 1.1](https://github.com/lllyasviel/ControlNet-v1-1-nightly) based version! 
* [05/28/2023] Thank [chenxwh](https://github.com/chenxwh), add a [Replicate demo](https://replicate.com/cjwbw/controlvideo)!
* [05/25/2023] Code [ControlVideo](https://github.com/YBYBZhang/ControlVideo/) released!
* [05/23/2023] Paper [ControlVideo](https://arxiv.org/abs/2305.13077) released!

## Setup

### 1. Download Weights
All pre-trained weights are downloaded to `checkpoints/` directory, including the pre-trained weights of [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5), ControlNet 1.0 conditioned on [canny edges](https://huggingface.co/lllyasviel/sd-controlnet-canny), [depth maps](https://huggingface.co/lllyasviel/sd-controlnet-depth), [human poses](https://huggingface.co/lllyasviel/sd-controlnet-openpose), and ControlNet 1.1 in [here](https://huggingface.co/lllyasviel). 
The `flownet.pkl` is the weights of [RIFE](https://github.com/megvii-research/ECCV2022-RIFE).
The final file tree likes:

```none
checkpoints
├── stable-diffusion-v1-5
├── sd-controlnet-canny
├── sd-controlnet-depth
├── sd-controlnet-openpose
├── ...
├── flownet.pkl
```
### 2. Requirements

```shell
conda create -n controlvideo python=3.10
conda activate controlvideo
pip install -r requirements.txt
```
Note: `xformers` is recommended to save memory and running time. `controlnet-aux` is updated to version 0.0.6.

## Inference

To perform text-to-video generation, just run this command in `inference.sh`:
```bash
python inference.py \
    --prompt "A striking mallard floats effortlessly on the sparkling pond." \
    --condition "depth" \
    --video_path "data/mallard-water.mp4" \
    --output_path "outputs/" \
    --video_length 15 \
    --smoother_steps 19 20 \
    --width 512 \
    --height 512 \
    --frame_rate 2 \
    --version v10 \
    # --is_long_video
```
where `--video_length` is the length of synthesized video, `--condition` represents the type of structure sequence,
`--smoother_steps` determines at which timesteps to perform smoothing, `--version` selects the version of ControlNet (e.g., `v10` or `v11`), and `--is_long_video` denotes whether to enable efficient long-video synthesis.

## Visualizations

### ControlVideo on depth maps

<table class="center">
<tr>
  <td width=30% align="center"><img src="assets/depth/A_charming_flamingo_gracefully_wanders_in_the_calm_and_serene_water,_its_delicate_neck_curving_into_an_elegant_shape..gif" raw=true></td>
	<td width=30% align="center"><img src="assets/depth/A_striking_mallard_floats_effortlessly_on_the_sparkling_pond..gif" raw=true></td>
  <td width=30% align="center"><img src="assets/depth/A_gigantic_yellow_jeep_slowly_turns_on_a_wide,_smooth_road_in_the_city..gif" raw=true></td>
</tr>
<tr>
  <td width=30% align="center">"A charming flamingo gracefully wanders in the calm and serene water, its delicate neck curving into an elegant shape."</td>
  <td width=30% align="center">"A striking mallard floats effortlessly on the sparkling pond."</td>
  <td width=30% align="center">"A gigantic yellow jeep slowly turns on a wide, smooth road in the city."</td>
</tr>
 <tr>
	<td width=30% align="center"><img src="assets/depth/A_sleek_boat_glides_effortlessly_through_the_shimmering_river,_van_gogh_style..gif" raw=true></td>
  <td width=30% align="center"><img src="assets/depth/A_majestic_sailing_boat_cruises_along_the_vast,_azure_sea..gif" raw=true></td>
	<td width=30% align="center"><img src="assets/depth/A_contented_cow_ambles_across_the_dewy,_verdant_pasture..gif" raw=true></td>
</tr>
<tr>
  <td width=30% align="center">"A sleek boat glides effortlessly through the shimmering river, van gogh style."</td>
  <td width=30% align="center">"A majestic sailing boat cruises along the vast, azure sea."</td>
  <td width=30% align="center">"A contented cow ambles across the dewy, verdant pasture."</td>
</tr>
</table>

### ControlVideo on canny edges

<table class="center">
<tr>
  <td width=30% align="center"><img src="assets/canny/A_young_man_riding_a_sleek,_black_motorbike_through_the_winding_mountain_roads..gif" raw=true></td>
  <td width=30% align="center"><img src="assets/canny/A_white_swan_moving_on_the_lake,_cartoon_style..gif" raw=true></td>
	<td width=30% align="center"><img src="assets/canny/A_dusty_old_jeep_was_making_its_way_down_the_winding_forest_road,_creaking_and_groaning_with_each_bump_and_turn..gif" raw=true></td>
</tr>
<tr>
  <td width=30% align="center">"A young man riding a sleek, black motorbike through the winding mountain roads."</td>
  <td width=30% align="center">"A white swan movingon the lake, cartoon style."</td>
  <td width=30% align="center">"A dusty old jeep was making its way down the winding forest road, creaking and groaning with each bump and turn."</td>
</tr>
 <tr>
  <td width=30% align="center"><img src="assets/canny/A_shiny_red_jeep_smoothly_turns_on_a_narrow,_winding_road_in_the_mountains..gif" raw=true></td>
  <td width=30% align="center"><img src="assets/canny/A_majestic_camel_gracefully_strides_across_the_scorching_desert_sands..gif" raw=true></td>
	<td width=30% align="center"><img src="assets/canny/A_fit_man_is_leisurely_hiking_through_a_lush_and_verdant_forest..gif" raw=true></td>
</tr>
<tr>
  <td width=30% align="center">"A shiny red jeep smoothly turns on a narrow, winding road in the mountains."</td>
  <td width=30% align="center">"A majestic camel gracefully strides across the scorching desert sands."</td>
  <td width=30% align="center">"A fit man is leisurely hiking through a lush and verdant forest."</td>
</tr>
</table>


### ControlVideo on human poses

<table class="center">
<tr>
  <td width=25% align="center"><img src="assets/pose/James_bond_moonwalk_on_the_beach,_animation_style.gif" raw=true></td>
  <td width=25% align="center"><img src="assets/pose/Goku_in_a_mountain_range,_surreal_style..gif" raw=true></td>
	<td width=25% align="center"><img src="assets/pose/Hulk_is_jumping_on_the_street,_cartoon_style.gif" raw=true></td>
  <td width=25% align="center"><img src="assets/pose/A_robot_dances_on_a_road,_animation_style.gif" raw=true></td>
</tr>
<tr>
  <td width=25% align="center">"James bond moonwalk on the beach, animation style."</td>
  <td width=25% align="center">"Goku in a mountain range, surreal style."</td>
  <td width=25% align="center">"Hulk is jumping on the street, cartoon style."</td>
  <td width=25% align="center">"A robot dances on a road, animation style."</td>
</tr></table>

### Long video generation

<table class="center">
<tr>
  <td width=60% align="center"><img src="assets/long/A_steamship_on_the_ocean,_at_sunset,_sketch_style.gif" raw=true></td>
	<td width=40% align="center"><img src="assets/long/Hulk_is_dancing_on_the_beach,_cartoon_style.gif" raw=true></td>
</tr>
<tr>
  <td width=60% align="center">"A steamship on the ocean, at sunset, sketch style."</td>
  <td width=40% align="center">"Hulk is dancing on the beach, cartoon style."</td>
</tr>
</table>

## Citation
If you make use of our work, please cite our paper.
```bibtex
@article{zhang2023controlvideo,
  title={ControlVideo: Training-free Controllable Text-to-Video Generation},
  author={Zhang, Yabo and Wei, Yuxiang and Jiang, Dongsheng and Zhang, Xiaopeng and Zuo, Wangmeng and Tian, Qi},
  journal={arXiv preprint arXiv:2305.13077},
  year={2023}
}
```

## Acknowledgement
This work repository borrows heavily from [Diffusers](https://github.com/huggingface/diffusers), [ControlNet](https://github.com/lllyasviel/ControlNet), [Tune-A-Video](https://github.com/showlab/Tune-A-Video), and [RIFE](https://github.com/megvii-research/ECCV2022-RIFE).
The code of HuggingFace demo borrows from [fffiloni/ControlVideo](https://huggingface.co/spaces/fffiloni/ControlVideo).
Thanks for their contributions!

There are also many interesting works on video generation: [Tune-A-Video](https://github.com/showlab/Tune-A-Video), [Text2Video-Zero](https://github.com/Picsart-AI-Research/Text2Video-Zero), [Follow-Your-Pose](https://github.com/mayuelala/FollowYourPose), [Control-A-Video](https://github.com/Weifeng-Chen/control-a-video), et al.
