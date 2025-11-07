import os
import cv2
import torch
import configargparse
from model import DeblurGAN
from utils import selectDevice, tensorToImage, imageToTensor


if __name__ == "__main__":
    
    # Select parameters for testing
    arg = configargparse.ArgumentParser()
    arg.add_argument('--dataset_path', type=str, default='test_videos', help='Dataset path.')
    arg.add_argument('--log_dir', type=str, default='deblurGAN_bs1_lr0.0001_numresblocks9_lambdaG100_lambdaD10', help='Name of the folder where the files of checkpoints and precision and loss values are stored.')
    arg.add_argument('--checkpoint', type=str, default='checkpoint_36_best_g.pth',help='Checkpoint to use')
    arg.add_argument('--num_resblocks', type=int, default=9, help='Number of residual blocks for the generator.')
    arg.add_argument('--GPU', type=bool, default=True, help='True to train the model in the GPU.')
    args = arg.parse_args()
    
    device = selectDevice(args)
    
    generator = DeblurGAN(n_resblocks=args.num_resblocks)
    state_dict = torch.load(os.path.join(args.log_dir, "checkpoints", args.checkpoint), map_location=device)
    generator.load_state_dict(state_dict)
    generator.to(device)
    generator.eval()
    
    video_paths = []
    
    for root, _, files in os.walk(os.path.join(args.dataset_path, "original")):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov')):
                video_paths.append(os.path.join(root, file))
    
    kernel_size = 5
    sigma = 7.0
    output_dir_blurred = os.path.join(args.dataset_path, "blurred")
    output_dir_deblurred = os.path.join(args.dataset_path, "deblurred")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    with torch.no_grad():
        
        for video_path in video_paths:
            
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"Could not open video: {video_path}")
                continue
    
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_name = os.path.splitext(os.path.basename(video_path))[0]
    
            # Read first frame to get resolution of the video
            ret, frame = cap.read()
            if not ret:
                print(f"Error reading first frame from video: {video_path}")
                continue
    
            height, width = frame.shape[:2]
    
            # Reset video to the first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
            output_path_blurred = os.path.join(output_dir_blurred, video_name)
            out_video_blurred = cv2.VideoWriter(output_path_blurred, fourcc, fps, (width, height))
            
            output_path_deblurred = os.path.join(output_dir_deblurred, video_name)
            out_video_deblurred = cv2.VideoWriter(output_path_deblurred, fourcc, fps, (width, height))
    
            for _ in range(total_frames):
    
                ret, frame = cap.read()
                if not ret:
                    break
    
                frame_blurred = cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigmaX=sigma)
                tensor = torch.unsqueeze(imageToTensor(frame_blurred), dim=0).to(device)
    
                out_tensor = generator(tensor)
                out_image = tensorToImage(out_tensor)
    
                out_video_blurred.write(frame_blurred)
                out_video_deblurred.write(out_image)
    
            cap.release()
            out_video_blurred.release()
            out_video_deblurred.release()

