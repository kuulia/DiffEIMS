import torch
import argparse
'''
This script extracts the encoder weights from a PyTorch Lightning checkpoint (.ckpt) 
and saves them as a standalone .pt file. This is useful when you want to re-use or fine-tune 
the encoder separately from the rest of the model.
'''


def main(ckpt_path: str, pt_path: str):
    """
    Extract encoder weights from a PyTorch Lightning checkpoint and save them as a standalone .pt file.

    This function is useful when you want to decouple a pretrained encoder from a larger model for
    reuse or fine-tuning. It assumes that the encoder parameters in the checkpoint's state_dict
    are prefixed with "encoder." and strips this prefix before saving.

    Args:
        ckpt_path (str): Path to the input .ckpt file (PyTorch Lightning checkpoint).
        pt_path (str): Path where the extracted encoder weights will be saved as a .pt file.

    Behavior:
        - Loads the .ckpt file using torch.load().
        - Accesses the "state_dict" key from the checkpoint.
        - Filters out keys that start with "encoder." and removes the prefix.
        - Saves the resulting dictionary to the specified .pt path.

    Example:
        >>> main("checkpoints/my_model.ckpt", "encoder.pt")
        Saved encoder weights to encoder.pt
    """
    ckpt = torch.load(ckpt_path, map_location='cpu')

    # Extract encoder weights
    encoder_weights = {
        k.replace("encoder.", ""): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("encoder.")
    }

    # Save as .pt
    torch.save(encoder_weights, pt_path)
    print(f"Saved encoder weights to {pt_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract encoder weights from checkpoint.")
    parser.add_argument("ckpt_path", type=str, help="Path to the .ckpt file")
    parser.add_argument("pt_path", type=str, help="Output path for .pt file")

    args = parser.parse_args()
    main(args.ckpt_path, args.pt_path)
