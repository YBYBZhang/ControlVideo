import gradio as gr
import os
import shutil
import subprocess
import cv2
import numpy as np
import math

from huggingface_hub import snapshot_download

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


model_ids = [
    'runwayml/stable-diffusion-v1-5',
    'lllyasviel/sd-controlnet-depth', 
    'lllyasviel/sd-controlnet-canny', 
    'lllyasviel/sd-controlnet-openpose',
    "lllyasviel/control_v11p_sd15_softedge",
    "lllyasviel/control_v11p_sd15_scribble",
    "lllyasviel/control_v11p_sd15s2_lineart_anime",
    "lllyasviel/control_v11p_sd15_lineart",
    "lllyasviel/control_v11f1p_sd15_depth",
    "lllyasviel/control_v11p_sd15_canny",
    "lllyasviel/control_v11p_sd15_openpose",
    "lllyasviel/control_v11p_sd15_normalbae"
]


for model_id in model_ids:
    model_name = model_id.split('/')[-1]
    snapshot_download(model_id, cache_dir=f'checkpoints/{model_name}')

def load_model(model_id):
    local_dir = f'checkpoints/stable-diffusion-v1-5'
    # Check if the directory exists
    if os.path.exists(local_dir):
        # Delete the directory if it exists
        shutil.rmtree(local_dir)

    model_name = model_id.split('/')[-1]
    snapshot_download(model_id, local_dir=f'checkpoints/{model_name}')
    os.rename(f'checkpoints/{model_name}', f'checkpoints/stable-diffusion-v1-5')
    return "model loaded"

def get_frame_count(filepath):
    if filepath is not None:
        video = cv2.VideoCapture(filepath) 
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
        video.release()

        # LIMITS
        if frame_count > 100 :
            frame_count = 100 # limit to 100 frames to avoid cuDNN errors

        return gr.update(maximum=frame_count)

    else:
        return gr.update(value=1, maximum=100 )

def get_video_dimension(filepath):
    video = cv2.VideoCapture(filepath)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    return width, height, fps, frame_count

def resize_video(input_vid, output_vid, width, height, fps):
    print(f"RESIZING ...")
    # Open the input video file
    video = cv2.VideoCapture(input_vid)

    # Create a VideoWriter object to write the resized video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for the output video
    output_video = cv2.VideoWriter(output_vid, fourcc, fps, (width, height))

    while True:
        # Read a frame from the input video
        ret, frame = video.read()
        if not ret:
            break

        # Resize the frame to the desired dimensions
        resized_frame = cv2.resize(frame, (width, height))

        # Write the resized frame to the output video file
        output_video.write(resized_frame)

    # Release the video objects
    video.release()
    output_video.release()
    print(f"RESIZE VIDEO DONE!")
    return output_vid

def make_nearest_multiple_of_32(number):
    remainder = number % 32
    if remainder <= 16:
        number -= remainder
    else:
        number += 32 - remainder
    return number 

def change_video_fps(input_path):
    print(f"CHANGING FIANL OUTPUT FPS")
    cap = cv2.VideoCapture(input_path)
    # Check if the final file already exists
    if os.path.exists('output_video.mp4'):
        # Delete the existing file
        os.remove('output_video.mp4')
    output_path = 'output_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_fps = 12
    output_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(output_path, fourcc, output_fps, output_size)
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Write the current frame to the output video multiple times to increase the frame rate
        for _ in range(output_fps // 8):
            out.write(frame)
        
        frame_count += 1
        print(f'Processed frame {frame_count}')

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return 'output_video.mp4'

def run_inference(prompt, video_path, version_condition, video_length, seed):
    
    seed = math.floor(seed)
    o_width = get_video_dimension(video_path)[0]
    o_height = get_video_dimension(video_path)[1]
    version, condition = version_condition.split("+")

    # Prepare dimensions
    if o_width > 512 :
        # Calculate the new height while maintaining the aspect ratio
        n_height = int(o_height / o_width * 512)
        n_width = 512
    else:
        n_height = o_height
        n_width = o_width

    # Make sure new dimensions are multipe of 32
    r_width = make_nearest_multiple_of_32(n_width)
    r_height = make_nearest_multiple_of_32(n_height)
    print(f"multiple of 32 sizes : {r_width}x{r_height}")

    # Get FPS of original video input
    original_fps = get_video_dimension(video_path)[2] 
    if original_fps > 12 :
        print(f"FPS is too high: {original_fps}")          
        target_fps = 12
    else : 
        target_fps = original_fps
    print(f"NEW INPUT FPS: {target_fps}, NEW LENGTH: {video_length}")
    
    # Check if the resized file already exists
    if os.path.exists('resized.mp4'):
        # Delete the existing file
        os.remove('resized.mp4')
    
    resized = resize_video(video_path, 'resized.mp4', r_width, r_height, target_fps)
    resized_video_fcount = get_video_dimension(resized)[3]
    print(f"RESIZED VIDEO FRAME COUNT: {resized_video_fcount}")

    # Make sure new total frame count is enough to handle chosen video length
    if video_length > resized_video_fcount :
        video_length = resized_video_fcount
    # video_length = int((target_fps * video_length) / original_fps)
    
    output_path = 'output/'
    os.makedirs(output_path, exist_ok=True)
            
    # Check if the file already exists
    if os.path.exists(os.path.join(output_path, f"result.mp4")):
        # Delete the existing file
        os.remove(os.path.join(output_path, f"result.mp4"))

    print(f"RUNNING INFERENCE ...")
    if video_length > 16:
        command = f"python inference.py --prompt '{prompt}' --condition '{condition}' --video_path '{resized}' --output_path '{output_path}' --temp_video_name 'result' --width {r_width} --height {r_height} --seed {seed} --video_length {video_length} --smoother_steps 19 20 --version {version} --is_long_video"
    else:
        command = f"python inference.py --prompt '{prompt}' --condition '{condition}' --video_path '{resized}' --output_path '{output_path}' --temp_video_name 'result'  --width {r_width} --height {r_height} --seed {seed} --video_length {video_length} --smoother_steps 19 20 --version {version} "
    
    try:
        subprocess.run(command, shell=True)
    except cuda.Error as e:
        return f"CUDA Error: {e}", None
    except RuntimeError as e:
        return f"Runtime Error: {e}", None

    # Construct the video path
    video_path_output = os.path.join(output_path, f"result.mp4")

    # Resize to original video input size
    #o_width = get_video_dimension(video_path)[0]
    #o_height = get_video_dimension(video_path)[1]
    #resize_video(video_path_output, 'resized_final.mp4', o_width, o_height, target_fps)

    # Check generated video FPS
    gen_fps = get_video_dimension(video_path_output)[2] 
    print(f"GEN VIDEO FPS: {gen_fps}")
    final = change_video_fps(video_path_output)
    print(f"FINISHED !")
    
    return final
    # return final, gr.Group.update(visible=True)
 

css="""
#col-container {max-width: 810px; margin-left: auto; margin-right: auto;}
.animate-spin {
  animation: spin 1s linear infinite;
}
@keyframes spin {
  from {
      transform: rotate(0deg);
  }
  to {
      transform: rotate(360deg);
  }
}
#share-btn-container {
  display: flex; 
  padding-left: 0.5rem !important; 
  padding-right: 0.5rem !important; 
  background-color: #000000; 
  justify-content: center; 
  align-items: center; 
  border-radius: 9999px !important; 
  max-width: 13rem;
}
#share-btn-container:hover {
  background-color: #060606;
}
#share-btn {
  all: initial; 
  color: #ffffff;
  font-weight: 600; 
  cursor:pointer; 
  font-family: 'IBM Plex Sans', sans-serif; 
  margin-left: 0.5rem !important; 
  padding-top: 0.5rem !important; 
  padding-bottom: 0.5rem !important;
  right:0;
}
#share-btn * {
  all: unset;
}
#share-btn-container div:nth-child(-n+2){
  width: auto !important;
  min-height: 0px !important;
}
#share-btn-container .wrap {
  display: none !important;
}
#share-btn-container.hidden {
  display: none!important;
}
img[src*='#center'] { 
    display: block;
    margin: auto;
}
"""
with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("""
            <h1 style="text-align: center;">controlvideo: Training-free Controllable Text-to-Video Generation</h1>
            <p style="text-align: center;">
                [<a href="https://arxiv.org/abs/2305.13077" style="color:blue;">arXiv</a>] 
                [<a href="https://github.com/YBYBZhang/ControlVideo" style="color:blue;">GitHub</a>]
            </p>
            <p style="text-align: center;"> controlvideo adapts ControlNet to the video counterpart without any finetuning, aiming to directly inherit its high-quality and consistent generation. </p>            
        """)
        
        with gr.Column():
            with gr.Row():
                video_path = gr.Video(label="Input video", source="upload", type="filepath", visible=True, elem_id="video-in")
                video_res = gr.Video(label="result", elem_id="video-out")
                
                # with gr.Column():
                #     video_res = gr.Video(label="result", elem_id="video-out")
                #     with gr.Group(elem_id="share-btn-container", visible=False) as share_group:
                #         community_icon = gr.HTML(community_icon_html)
                #         loading_icon = gr.HTML(loading_icon_html)
                #         share_button = gr.Button("Share to community", elem_id="share-btn")
            with gr.Row():
                chosen_model = gr.Dropdown(label="Diffusion model (*1.5)", choices=['runwayml/stable-diffusion-v1-5','nitrosocke/Ghibli-Diffusion'], value="runwayml/stable-diffusion-v1-5", allow_custom_value=True)
                model_status = gr.Textbox(label="status")
                load_model_btn = gr.Button("load model (optional)")
            prompt = gr.Textbox(label="prompt", info="If you loaded a custom model, do not forget to include Prompt trigger", elem_id="prompt-in")
            with gr.Column():
                video_length = gr.Slider(label="Video length", info="How many frames do you want to process ? For demo purpose, max is set to 24", minimum=1, maximum=12, step=1, value=2)
                with gr.Row():
                    # version = gr.Dropdown(label="ControlNet version", choices=["v10", "v11"], value="v10")
                    version_condition = gr.Dropdown(label="ControlNet version + Condition", 
                                                    choices=["v10+depth_midas", "v10+canny", "v10+openpose", "v11+softedge_pidinet", "v11+softedge_pidsafe",
                                                    "v11+softedge_hed", "v11+softedge_hedsafe", "v11+scribble_hed", "v11+scribble_pidinet", "v11+lineart_anime",
                                                    "v11+lineart_coarse", "v11+lineart_realistic", "v11+depth_midas", "v11+depth_leres", "v11+depth_leres++", 
                                                    "v11+depth_zoe", "v11+canny", "v11+openpose", "v11+openpose_face", "v11+openpose_faceonly", "v11+openpose_full", 
                                                    "v11+openpose_hand", "v11+normal_bae"], value="v10+depth_midas")
                    seed = gr.Number(label="seed", value=42)
            submit_btn = gr.Button("Submit")
        
            
            gr.Examples(
                examples=[["James bond moonwalks on the beach.", "./data/moonwalk.mp4", 'v10+openpose', 15, 42],
                          ["A striking mallard floats effortlessly on the sparkling pond.", "./data/mallard-water.mp4", "v11+depth_midas", 15, 42]],
                fn=run_inference,
                inputs=[prompt,
                         video_path,
                         version_condition,
                         video_length,
                         seed,
                        ],
                # outputs=[video_res, share_group],
                outputs=video_res,
                cache_examples=False
            )
                
    # share_button.click(None, [], [], _js=share_js)
    load_model_btn.click(fn=load_model, inputs=[chosen_model], outputs=[model_status], queue=False)
    video_path.change(fn=get_frame_count,
                      inputs=[video_path],
                      outputs=[video_length],
                      queue=False
                     )
    submit_btn.click(fn=run_inference, 
                     inputs=[prompt,
                             video_path,
                             version_condition,
                             video_length,
                             seed,
                            ],
                    outputs=video_res)

demo.queue(max_size=12).launch(share=True)