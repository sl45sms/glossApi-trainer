#!/usr/bin/env python3

import os
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import ModelConfig, ScriptArguments, SFTConfig, SFTTrainer, TrlParser, get_peft_config


def _looks_like_local_path(dataset_name: str) -> bool:
	return dataset_name.startswith(("/", "./", "../", "~")) or Path(dataset_name).exists()


def _find_split_file(dataset_dir: Path, split_name: str) -> Optional[Path]:
	for suffix in ("jsonl", "json"):
		candidate = dataset_dir / f"{split_name}.{suffix}"
		if candidate.is_file():
			return candidate
	return None


def _load_local_or_hub_dataset(script_args: ScriptArguments, training_args: SFTConfig):
	dataset_name = str(script_args.dataset_name)
	expanded_name = os.path.expandvars(os.path.expanduser(dataset_name))
	dataset_path = Path(expanded_name)

	if not _looks_like_local_path(dataset_name):
		dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
	else:
		if not dataset_path.exists():
			raise FileNotFoundError(f"Local dataset path does not exist: {dataset_path}")

		data_files = {}
		if dataset_path.is_file():
			data_files[script_args.dataset_train_split] = str(dataset_path)
		else:
			train_file = _find_split_file(dataset_path, script_args.dataset_train_split)
			if train_file is None:
				raise FileNotFoundError(
					f"Missing {script_args.dataset_train_split}.jsonl or {script_args.dataset_train_split}.json in {dataset_path}"
				)
			data_files[script_args.dataset_train_split] = str(train_file)

			eval_file = _find_split_file(dataset_path, script_args.dataset_test_split)
			if eval_file is not None:
				data_files[script_args.dataset_test_split] = str(eval_file)

		dataset = load_dataset("json", data_files=data_files)

	if training_args.eval_strategy != "no" and script_args.dataset_test_split not in dataset:
		raise ValueError(
			f"Evaluation is enabled but split '{script_args.dataset_test_split}' is not available. "
			"Provide a validation file or set eval_strategy=no."
		)

	return dataset


def main(script_args: ScriptArguments, training_args: SFTConfig, model_args: ModelConfig) -> None:
	store_base_dir = "./"

	model = AutoModelForCausalLM.from_pretrained(
		model_args.model_name_or_path,
		dtype=model_args.dtype,
		use_cache=False if training_args.gradient_checkpointing else True,
		attn_implementation=model_args.attn_implementation,
	)

	tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
	tokenizer.pad_token = tokenizer.eos_token

	dataset = _load_local_or_hub_dataset(script_args, training_args)
	eval_dataset = None
	if training_args.eval_strategy != "no":
		eval_dataset = dataset[script_args.dataset_test_split]

	trainer = SFTTrainer(
		model=model,
		args=training_args,
		train_dataset=dataset[script_args.dataset_train_split],
		eval_dataset=eval_dataset,
		processing_class=tokenizer,
		peft_config=get_peft_config(model_args),
	)

	trainer.train()
	trainer.save_model(os.path.join(store_base_dir, training_args.output_dir))
	if training_args.push_to_hub:
		trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
	parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
	script_args, training_args, model_args, _ = parser.parse_args_and_config(return_remaining_strings=True)
	main(script_args, training_args, model_args)
