#!/usr/bin/env python3

# This script runs a smoke test inference prompt against a merged Apertus model.
# It loads the merged model and tokenizer from a specified directory,
# and generates a response to a given prompt, printing the result as JSON.


import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_PROMPT = (
	"Δώσε ΜΟΝΟ valid JSON με πεδία question και answer. "
	"Θέμα: ποιος είναι ο σκοπός του Βιβλίου του Δασκάλου;"
)


def parse_args():
	parser = argparse.ArgumentParser(description="Run one inference prompt against a merged Apertus model.")
	parser.add_argument("--model_dir", required=True, help="Merged model directory")
	parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Single prompt to generate from")
	parser.add_argument("--max_new_tokens", type=int, default=256)
	parser.add_argument("--temperature", type=float, default=0.2)
	parser.add_argument("--top_p", type=float, default=0.95)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
	model = AutoModelForCausalLM.from_pretrained(
		args.model_dir,
		torch_dtype=torch.bfloat16,
		device_map="auto",
	)
	model.eval()

	messages = [{"role": "user", "content": args.prompt}]
	encoded = tokenizer.apply_chat_template(
		messages,
		add_generation_prompt=True,
		return_dict=True,
		return_tensors="pt",
	)
	encoded = encoded.to(model.device)
	input_ids = encoded["input_ids"]
	attention_mask = encoded.get("attention_mask")

	with torch.inference_mode():
		output_ids = model.generate(
			input_ids=input_ids,
			attention_mask=attention_mask,
			max_new_tokens=args.max_new_tokens,
			do_sample=args.temperature > 0,
			temperature=args.temperature,
			top_p=args.top_p,
			pad_token_id=tokenizer.pad_token_id,
			eos_token_id=tokenizer.eos_token_id,
		)

	generated_ids = output_ids[0][input_ids.shape[-1]:]
	text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

	result = {
		"prompt": args.prompt,
		"response": text,
	}
	print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
	main()