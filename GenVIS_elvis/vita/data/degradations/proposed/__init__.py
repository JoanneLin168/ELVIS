import torch
from .arch import VDPNet
from .noise import reshape_noise_params, generate_noise

def build_model(args=None):
    print("Building model...")

    if args is not None:
        model = VDPNet(in_channels=3, out_channels=3, args=args, num_frames=args.num_frames)
        if args.checkpoint:
            print(f"Loading model from checkpoint: {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        return model
    else:
        print("No arguments provided, using default parameters for model instantiation.")
        return VDPNet()
