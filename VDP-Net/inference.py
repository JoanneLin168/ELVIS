import argparse
import os
import torch
import tqdm
import cv2
import numpy as np

from torchvision import transforms
from PIL import Image

from src.models import build_model
from src.dataset.noise import generate_noise, reshape_noise_params


def get_arguments():
    parser = argparse.ArgumentParser(description='VDP-Net inference script.')
    parser.add_argument('--input', required=True, help='Path to input normal-light video')
    parser.add_argument('--reference', required=True, help='Path to low-light reference video')
    parser.add_argument('--output', required=False, help='Path to save output video', default='output')
    parser.add_argument('--checkpoint', required=True, type=str, help='Path to checkpoint to load')
    parser.add_argument('--num_frames', default=5, type=int, help='Number of frames to sample from each video (network, not dataset)')
    parser.add_argument('--patch_size', default=256, type=int, help='Patch size for training')
    parser.add_argument('--device', default='cuda:0', help='Device to run the model on')
    args = parser.parse_args()
    return args

def main(args):
    device = args.device
    model_dir = os.path.dirname(os.path.dirname(args.checkpoint))

    # Build model
    model = build_model(args=args)
    model.to(args.device)
    model.eval()

    transform = transforms.Compose([transforms.ToTensor()])
    transform_ref = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop((args.patch_size, args.patch_size)),
    ])

    # Get input video frames
    input_frames = []
    input_filenames = sorted(os.listdir(args.input))
    for filename in sorted(os.listdir(args.input)):
        # Only allow num_frames frames because of blur
        if len(input_frames) == args.num_frames:
            break

        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(args.input, filename)
            img = Image.open(img_path).convert('RGB')
            img = transform(img)
            input_frames.append(img)
    input_frames = torch.stack(input_frames).to(device)

    # Get clip from reference video
    ref_frames = []
    for filename in sorted(os.listdir(args.reference)):
        if len(ref_frames) == args.num_frames:
            break

        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(args.reference, filename)
            img = Image.open(img_path).convert('RGB')
            img = transform_ref(img)
            ref_frames.append(img)
    ref_frames = torch.stack(ref_frames).unsqueeze(0).to(device)  # [num_frames, C, H, W]

    # Estimate noise parameters
    v_pred = None
    with torch.no_grad():
        v_pred = model.estimate_noise(ref_frames)
    v_pred_np = v_pred.squeeze(0).cpu().numpy().tolist()  # Convert to list for printing
    print("Predictions:", v_pred_np)


    print("Applying noise onto input image...")
    # Create noise dict
    # noise_params = v_pred.unsqueeze(1).repeat(1, 1, 1)
    print(v_pred.shape)
    noise_params = v_pred
    num_classes = v_pred.shape[-1]
    # noise_params = noise_params.view(-1, num_classes)
    noise_dict = reshape_noise_params(noise_params, num_frames=args.num_frames)

    # Apply onto input video
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    noisy_frames = []
    input_frames = input_frames.unsqueeze(0) # (1, num_frames, C, H, W)
    noisy_frames = generate_noise(input_frames, noise_dict, num_frames=args.num_frames, device=device)

    noisy_frames_np = []
    for i, noisy_frame in enumerate(noisy_frames.squeeze(0)):
        frame_np = noisy_frame.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
        frame_np = (frame_np * 255).clip(0, 255).astype(np.uint8)
        noisy_frames_np.append(frame_np)
        frame_pil = Image.fromarray(frame_np)
        frame_pil.save(os.path.join(output_dir, input_filenames[i]))
    
    # Create mp4
    output_video_path = os.path.join(output_dir, 'noisy_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width = frame_np.shape[0], frame_np.shape[1]
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 30, (width, height))
    for frame in noisy_frames_np:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame)
    video_writer.release()
    print(f"Noisy video saved to {output_video_path}")

if __name__ == "__main__":
    args = get_arguments()
    main(args)