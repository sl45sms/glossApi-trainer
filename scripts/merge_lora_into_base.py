#!/usr/bin/env python3

# This script merges a LoRA adapter into a base model 
# and saves the resulting standalone model.
# It uses the PEFT library to perform the merging. 
# The merged model is saved in a specified output directory along with the tokenizer.

import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
	parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a base model.")
	parser.add_argument("--base_model", required=True, help="Base model id or local path")
	parser.add_argument("--adapter_dir", required=True, help="Directory containing the LoRA adapter")
	parser.add_argument("--output_dir", required=True, help="Directory for the merged standalone model")
	parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
	parser.add_argument("--device_map", default="auto", help="Transformers device_map value")
	return parser.parse_args()


def resolve_dtype(name: str):
	if name == "bfloat16":
		return torch.bfloat16
	if name == "float16":
		return torch.float16
	return torch.float32


def main() -> None:
	args = parse_args()
	os.makedirs(args.output_dir, exist_ok=True)

	base_model = AutoModelForCausalLM.from_pretrained(
		args.base_model,
		torch_dtype=resolve_dtype(args.dtype),
		device_map=args.device_map,
	)
	model = PeftModel.from_pretrained(base_model, args.adapter_dir)
	merged_model = model.merge_and_unload()
	merged_model.save_pretrained(args.output_dir, safe_serialization=True)

	tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)
	tokenizer.save_pretrained(args.output_dir)

	print(f"Merged model saved at: {args.output_dir}")


if __name__ == "__main__":
	main()