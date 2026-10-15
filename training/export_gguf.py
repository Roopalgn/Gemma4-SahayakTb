"""
SAHAYAK-TB — GGUF export script (standalone).

Use this if you need to export outside of the fine-tuning notebook.

Usage:
  python export_gguf.py --adapter-path ./sahayak_tb_lora --quantization q4_k_m --output-dir ./gguf_output
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Export fine-tuned SAHAYAK-TB to GGUF")
    parser.add_argument("--adapter-path", type=str, default="sahayak_tb_lora",
                        help="Path to saved LoRA adapter")
    parser.add_argument("--quantization", type=str, default="q4_k_m",
                        choices=["f16", "q8_0", "q5_k_m", "q4_k_m", "q3_k_m"],
                        help="Quantization method")
    parser.add_argument("--output-dir", type=str, default="gguf_output")
    parser.add_argument("--push-to-hub", type=str, default="",
                        help="HuggingFace repo to push to (e.g., username/sahayak-tb)")
    args = parser.parse_args()

    from unsloth import FastModel

    print(f"Loading adapter from {args.adapter_path}...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.adapter_path,
        max_seq_length=4096,
        load_in_4bit=True,
    )

    print(f"Exporting to GGUF ({args.quantization})...")
    model.save_pretrained_gguf(
        args.output_dir,
        tokenizer,
        quantization_method=args.quantization,
    )
    print(f"Saved to {args.output_dir}/")

    if args.push_to_hub:
        print(f"Pushing to HuggingFace: {args.push_to_hub}...")
        model.push_to_hub_gguf(
            args.push_to_hub,
            tokenizer,
            quantization_method=args.quantization,
        )
        print("Done.")


if __name__ == "__main__":
    main()
