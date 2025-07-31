"""
This script extracts the encoder and/or decoder weights from a PyTorch Lightning checkpoint (.ckpt) 
and saves them as standalone .pt files. This is useful when you want to re-use, fine-tune, or 
analyze specific parts of a trained model independently.

Supported prefixes:
- Encoder weights are assumed to be prefixed with "encoder.".
- Decoder weights may be prefixed with either "decoder." or "model." (commonly used in Lightning modules).
"""

import torch
import argparse
import os

def extract_weights(ckpt_path: str, output_dir: str, extract_encoder: bool = True, extract_decoder: bool = True):
    """
    Extract encoder and/or decoder weights from a PyTorch Lightning checkpoint and save them as standalone .pt files.

    This function decouples specific components of a Lightning-trained model, making it easier to 
    re-use parts of the model architecture such as the encoder or decoder independently of the rest.

    Args:
        ckpt_path (str): Path to the input .ckpt file (PyTorch Lightning checkpoint).
        output_dir (str): Directory where extracted .pt files will be saved.
        extract_encoder (bool): If True, extracts weights from keys starting with "encoder.".
        extract_decoder (bool): If True, extracts weights from keys starting with "decoder." or "model.".

    Behavior:
        - Loads the checkpoint using torch.load().
        - Extracts relevant keys based on the specified component(s).
        - Removes the component prefix (e.g., "encoder.", "decoder.", or "model.") from the keys.
        - Saves each extracted component's weights to a separate .pt file in the output directory.

    Example usage:
        >>> extract_weights("checkpoints/epoch=4.ckpt", "weights/", extract_encoder=True, extract_decoder=True)
        Saved encoder weights to weights/encoder.pt
        Saved decoder weights to weights/decoder.pt
    """
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if extract_encoder:
        try:
            encoder_weights = {
                k.replace("encoder.", "", 1): v
                for k, v in state_dict.items()
                if k.startswith("encoder.")
            }
            encoder_path = os.path.join(output_dir, "encoder.pt")
            torch.save(encoder_weights, encoder_path)
            print(f"Saved encoder weights to {encoder_path}")
        except:
            print(f'Failed to extract encoder weights')

    if extract_decoder:
        try:
            decoder_weights = {}
            for k, v in state_dict.items():
                if k.startswith("decoder."):
                    new_key = k.replace("decoder.", "", 1)
                    decoder_weights[new_key] = v
                elif k.startswith("model."):
                    new_key = k.replace("model.", "", 1)
                    decoder_weights[new_key] = v
            decoder_path = os.path.join(output_dir, "decoder.pt")
            torch.save(decoder_weights, decoder_path)
            print(f"Saved decoder weights to {decoder_path}")
        except:
            print(f'Failed to extract decoder weights')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract encoder and/or decoder weights from a Lightning checkpoint.")
    parser.add_argument("ckpt_path", type=str, help="Path to the .ckpt file")
    parser.add_argument("output_dir", type=str, help="Directory to save the extracted .pt files")
    parser.add_argument("--encoder", action="store_true", help="Extract encoder weights")
    parser.add_argument("--decoder", action="store_true", help="Extract decoder weights")

    args = parser.parse_args()

    # Default to extracting both if neither flag is set
    if not args.encoder and not args.decoder:
        args.encoder = args.decoder = True

    extract_weights(
        ckpt_path=args.ckpt_path,
        output_dir=args.output_dir,
        extract_encoder=args.encoder,
        extract_decoder=args.decoder,
    )