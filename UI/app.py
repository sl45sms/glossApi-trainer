#!/usr/bin/env python3

import argparse
import concurrent.futures
import html
import inspect
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


TEXTBOX_SIGNATURE = inspect.signature(gr.Textbox.__init__)
TEXTBOX_SUPPORTS_BUTTONS = "buttons" in TEXTBOX_SIGNATURE.parameters
TEXTBOX_SUPPORTS_SHOW_COPY_BUTTON = "show_copy_button" in TEXTBOX_SIGNATURE.parameters
BLOCKS_LAUNCH_SUPPORTS_CSS = "css" in inspect.signature(gr.Blocks.launch).parameters
BLOCKS_LAUNCH_SUPPORTS_SHOW_API = "show_api" in inspect.signature(gr.Blocks.launch).parameters


DEFAULT_SHARED_PROMPT = (
	"Δώσε ΜΟΝΟ valid JSON με πεδία question και answer. "
	"Θέμα: ποιος είναι ο σκοπός του Βιβλίου του Δασκάλου;"
)

DEFAULT_SINGLE_PROMPT = "Γράψε εδώ prompt μόνο για αυτό το μοντέλο."
DEFAULT_MAX_NEW_TOKENS = 160
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0

CUSTOM_CSS = """
:root {
	--surface-0: #f5efe2;
	--surface-1: rgba(255, 250, 241, 0.88);
	--surface-2: rgba(245, 233, 210, 0.92);
	--ink-0: #201715;
	--ink-1: #59463f;
	--ink-2: #7b655d;
	--hero-ink: #2a1c17;
	--editor-bg: #1d1f25;
	--editor-bg-soft: #252831;
	--editor-ink: #f6efe2;
	--editor-muted: #b7ab98;
	--editor-border: rgba(246, 239, 226, 0.14);
	--accent-base: #0c7c59;
	--accent-base-soft: rgba(12, 124, 89, 0.14);
	--accent-merged: #bb4d00;
	--accent-merged-soft: rgba(187, 77, 0, 0.14);
	--shadow-soft: 0 24px 70px rgba(63, 31, 18, 0.12);
	--radius-xl: 24px;
	--radius-lg: 18px;
	--font-sans: "IBM Plex Sans", "Noto Sans", "Liberation Sans", sans-serif;
	--font-serif: "Source Serif 4", "Iowan Old Style", Georgia, serif;
}

body, .gradio-container {
	background:
		radial-gradient(circle at top left, rgba(12, 124, 89, 0.10), transparent 28%),
		radial-gradient(circle at top right, rgba(187, 77, 0, 0.10), transparent 26%),
		linear-gradient(180deg, #f8f2e7 0%, #efe6d5 100%);
	color: var(--ink-0);
	font-family: var(--font-sans);
}

.gradio-container {
	max-width: 1480px !important;
	padding: 24px 20px 40px !important;
}

.app-shell {
	background: linear-gradient(180deg, rgba(255, 252, 246, 0.86), rgba(255, 249, 239, 0.94));
	border: 1px solid rgba(89, 70, 63, 0.12);
	border-radius: 32px;
	box-shadow: var(--shadow-soft);
	overflow: hidden;
	padding: 20px;
	backdrop-filter: blur(14px);
}

.hero-panel {
	background:
		linear-gradient(140deg, rgba(12, 124, 89, 0.10), transparent 32%),
		linear-gradient(220deg, rgba(187, 77, 0, 0.12), transparent 36%),
		#fffaf0;
	border: 1px solid rgba(89, 70, 63, 0.10);
	border-radius: 28px;
	padding: 28px 24px 20px 24px;
	margin-bottom: 18px;
}

.hero-panel h1 {
	font-family: var(--font-serif);
	font-size: clamp(2rem, 4vw, 3.7rem);
	line-height: 0.95;
	letter-spacing: -0.03em;
	margin: 0 0 8px 0;
	color: var(--hero-ink) !important;
	text-shadow: 0 1px 0 rgba(255, 255, 255, 0.55);
}

.hero-panel p,
.hero-panel li,
.meta-strip,
.status-strip {
	color: var(--ink-1);
	font-size: 0.97rem;
	line-height: 1.55;
}

.hero-panel strong,
.meta-card,
.meta-card div,
.meta-card strong,
.status-strip,
.status-strip strong {
	color: var(--hero-ink) !important;
}

.meta-grid {
	display: grid;
	grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
	gap: 12px;
	margin-top: 18px;
}

.meta-card {
	background: rgba(255, 255, 255, 0.62);
	border: 1px solid rgba(89, 70, 63, 0.12);
	border-radius: 18px;
	padding: 14px 16px;
}

.meta-card strong {
	display: block;
	font-size: 0.8rem;
	text-transform: uppercase;
	letter-spacing: 0.08em;
	margin-bottom: 8px;
	color: #6f554b;
}

.panel-card {
	background: var(--surface-1);
	border: 1px solid rgba(89, 70, 63, 0.12);
	border-radius: var(--radius-xl);
	padding: 16px;
	box-shadow: 0 12px 30px rgba(51, 28, 16, 0.07);
	margin-top: 12px;
	margin-bottom: 12px;
	backdrop-filter: blur(10px);
}

.panel-card.base-lane {
	background:
		linear-gradient(180deg, rgba(12, 124, 89, 0.08), transparent 42%),
		var(--surface-1);
	border-color: rgba(12, 124, 89, 0.26);
	box-shadow: 0 18px 42px rgba(12, 124, 89, 0.08);
}

.panel-card.merged-lane {
	background:
		linear-gradient(180deg, rgba(187, 77, 0, 0.08), transparent 42%),
		var(--surface-1);
	border-color: rgba(187, 77, 0, 0.28);
	box-shadow: 0 18px 42px rgba(187, 77, 0, 0.09);
}

.section-title {
	font-family: var(--font-serif);
	font-size: 1.45rem;
	margin: 6px 0 2px 0;
	color: var(--hero-ink) !important;
	text-shadow: 0 1px 0 rgba(255, 255, 255, 0.45);
}

.lane-title {
	font-family: var(--font-serif);
	font-size: 1.35rem;
	margin-bottom: 4px;
	color: var(--hero-ink) !important;
	text-shadow: 0 1px 0 rgba(255, 255, 255, 0.45);
}

.status-strip {
	background: rgba(255, 255, 255, 0.66);
	border: 1px solid rgba(89, 70, 63, 0.10);
	border-radius: 16px;
	padding: 10px 14px;
	margin-top: 8px;
	margin-bottom: 8px;
}

.lane-chip {
	display: inline-block;
	padding: 6px 10px;
	border-radius: 999px;
	font-size: 0.78rem;
	font-weight: 600;
	letter-spacing: 0.03em;
	margin-bottom: 8px;
}

.lane-chip.base {
	background: var(--accent-base-soft);
	color: var(--accent-base);
}

.lane-chip.merged {
	background: var(--accent-merged-soft);
	color: var(--accent-merged);
}

.gr-button-primary {
	background: linear-gradient(135deg, #1d8c68, #0c7c59) !important;
	border: 0 !important;
	box-shadow: 0 10px 26px rgba(12, 124, 89, 0.18) !important;
}

.gr-button-secondary {
	background: linear-gradient(135deg, #c96a1f, #bb4d00) !important;
	border: 0 !important;
	box-shadow: 0 10px 26px rgba(187, 77, 0, 0.18) !important;
	color: white !important;
}

.gr-button {
	border-radius: 999px !important;
	min-height: 44px !important;
	font-weight: 600 !important;
	letter-spacing: 0.01em;
}

.gr-form, .gr-box, .gr-group, .gr-accordion {
	border-color: rgba(89, 70, 63, 0.12) !important;
}

.gradio-container .prose h1,
.gradio-container .prose h2,
.gradio-container .prose h3,
.gradio-container .prose h4,
.gradio-container .prose strong,
.gradio-container label,
.gradio-container .form label,
.gradio-container .gr-markdown,
.gradio-container .gr-accordion summary,
.gradio-container .gr-accordion button,
.gradio-container .gr-accordion * {
	color: var(--hero-ink);
}

textarea,
input,
.gr-textbox textarea,
.gr-textbox input {
	font-family: var(--font-sans) !important;
	background: linear-gradient(180deg, var(--editor-bg-soft), var(--editor-bg)) !important;
	color: var(--editor-ink) !important;
	border: 1px solid var(--editor-border) !important;
	box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03), 0 8px 22px rgba(8, 10, 14, 0.18) !important;
	caret-color: var(--editor-ink) !important;
}

textarea::placeholder,
input::placeholder,
.gr-textbox textarea::placeholder,
.gr-textbox input::placeholder {
	color: var(--editor-muted) !important;
	opacity: 1 !important;
}

.gr-textbox textarea:focus,
.gr-textbox input:focus,
textarea:focus,
input:focus {
	border-color: rgba(255, 183, 107, 0.55) !important;
	box-shadow: 0 0 0 1px rgba(255, 183, 107, 0.24), 0 10px 24px rgba(8, 10, 14, 0.24) !important;
}

.gradio-container button[aria-label="copy"],
.gradio-container button[aria-label*="Copy"],
.gradio-container .gr-copy-button {
	color: var(--editor-ink) !important;
	background: rgba(255, 255, 255, 0.08) !important;
	border: 1px solid rgba(255, 255, 255, 0.10) !important;
}

footer {
	display: none !important;
}
"""


def discover_default_merged_model() -> str:
	scratch_dir = os.environ.get("SCRATCH")
	if not scratch_dir:
		return "./output/apertus_lora_proper_merged"

	candidates = [
		os.path.join(scratch_dir, "glossapi-trainer/output/apertus_lora_proper_merged"),
		os.path.join(scratch_dir, "glossapi-trainer/output/apertus_lora_short100_merged"),
		os.path.join(scratch_dir, "glossapi-trainer/output/apertus_lora_smoke_merged"),
	]
	for candidate in candidates:
		if os.path.isdir(candidate):
			return candidate
	return candidates[0]


def default_device_for(index: int) -> Optional[str]:
	if torch.cuda.is_available():
		device_count = torch.cuda.device_count()
		if index < device_count:
			return f"cuda:{index}"
		if device_count > 0:
			return "cuda:0"
	return None


def resolve_dtype(name: str):
	if name == "bfloat16":
		return torch.bfloat16
	if name == "float16":
		return torch.float16
	return torch.float32


def parse_bool(value: str) -> bool:
	return value.strip().lower() in {"1", "true", "yes", "on"}


def format_path(value: str) -> str:
	return html.escape(value or "-")


def textbox_copy_kwargs() -> dict:
	if TEXTBOX_SUPPORTS_BUTTONS:
		return {"buttons": ["copy"]}
	if TEXTBOX_SUPPORTS_SHOW_COPY_BUTTON:
		return {"show_copy_button": True}
	return {}


@dataclass
class GenerationResult:
	text: str
	seconds: float
	device: str
	token_count: int
	error: Optional[str] = None


@dataclass
class ModelRuntime:
	label: str
	model_ref: str
	preferred_device: Optional[str]
	dtype_name: str
	trust_remote_code: bool
	attn_implementation: Optional[str] = None
	tokenizer: Optional[AutoTokenizer] = None
	model: Optional[AutoModelForCausalLM] = None
	load_lock: threading.Lock = field(default_factory=threading.Lock)
	generate_lock: threading.Lock = field(default_factory=threading.Lock)

	def _device_map(self):
		if not self.preferred_device or self.preferred_device == "auto":
			return "auto"
		return {"": self.preferred_device}

	def ensure_loaded(self) -> None:
		if self.model is not None and self.tokenizer is not None:
			return

		with self.load_lock:
			if self.model is not None and self.tokenizer is not None:
				return

			tokenizer = AutoTokenizer.from_pretrained(
				self.model_ref,
				trust_remote_code=self.trust_remote_code,
			)
			if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
				tokenizer.pad_token = tokenizer.eos_token

			target_dtype = resolve_dtype(self.dtype_name)
			load_kwargs = {
				"dtype": target_dtype,
				"device_map": self._device_map(),
				"low_cpu_mem_usage": True,
				"trust_remote_code": self.trust_remote_code,
			}
			if self.attn_implementation:
				load_kwargs["attn_implementation"] = self.attn_implementation

			model = AutoModelForCausalLM.from_pretrained(
				self.model_ref,
				**load_kwargs,
			)
			model = model.to(dtype=target_dtype)
			model.eval()

			self.tokenizer = tokenizer
			self.model = model

	def primary_device(self) -> str:
		if self.model is None:
			return self.preferred_device or "auto"
		try:
			return str(self.model.device)
		except Exception:
			try:
				return str(next(self.model.parameters()).device)
			except StopIteration:
				return self.preferred_device or "unknown"

	def generate(self, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> GenerationResult:
		prompt = prompt.strip()
		if not prompt:
			raise ValueError("Το prompt είναι κενό.")

		self.ensure_loaded()
		assert self.tokenizer is not None
		assert self.model is not None

		start_time = time.perf_counter()
		with self.generate_lock:
			messages = [{"role": "user", "content": prompt}]
			encoded = self.tokenizer.apply_chat_template(
				messages,
				add_generation_prompt=True,
				return_dict=True,
				return_tensors="pt",
			)
			model_device = self.primary_device()
			encoded = encoded.to(model_device)

			generate_kwargs = {
				"input_ids": encoded["input_ids"],
				"attention_mask": encoded.get("attention_mask"),
				"max_new_tokens": int(max_new_tokens),
				"do_sample": temperature > 0,
				"pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
				"eos_token_id": self.tokenizer.eos_token_id,
			}
			if temperature > 0:
				generate_kwargs["temperature"] = float(temperature)
				generate_kwargs["top_p"] = float(top_p)

			with torch.inference_mode():
				output_ids = self.model.generate(**generate_kwargs)

			generated_ids = output_ids[0][encoded["input_ids"].shape[-1]:]
			text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
			token_count = int(generated_ids.shape[-1])

		elapsed = time.perf_counter() - start_time
		return GenerationResult(text=text, seconds=elapsed, device=model_device, token_count=token_count)


def build_status_message(runtime_label: str, message: str) -> str:
	return f"<div class='status-strip'><strong>{html.escape(runtime_label)}</strong> {message}</div>"


def run_single(runtime: ModelRuntime, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> Tuple[str, str]:
	try:
		result = runtime.generate(prompt, max_new_tokens, temperature, top_p)
		tokens_per_second = result.token_count / result.seconds if result.seconds > 0 else 0.0
		status = build_status_message(
			runtime.label,
			f"ολοκλήρωσε σε {result.seconds:.2f}s στο {html.escape(result.device)} "
			f"για {result.token_count} tokens ({tokens_per_second:.1f} tok/s).",
		)
		return result.text, status
	except Exception as exc:
		error_text = f"[ERROR]\n{exc}"
		status = build_status_message(runtime.label, f"απέτυχε: {html.escape(str(exc))}")
		return error_text, status


def run_parallel(
	base_runtime: ModelRuntime,
	merged_runtime: ModelRuntime,
	prompt: str,
	max_new_tokens: int,
	temperature: float,
	top_p: float,
) -> Tuple[str, str, str]:
	final_state = None
	for state in run_parallel_stream(
		base_runtime,
		merged_runtime,
		prompt,
		max_new_tokens,
		temperature,
		top_p,
	):
		final_state = state
	assert final_state is not None
	return final_state


def run_parallel_stream(
	base_runtime: ModelRuntime,
	merged_runtime: ModelRuntime,
	prompt: str,
	max_new_tokens: int,
	temperature: float,
	top_p: float,
):
	prompt = prompt.strip()
	if not prompt:
		raise gr.Error("Γράψε πρώτα ένα κοινό prompt.")

	base_output = ""
	merged_output = ""
	statuses = [build_status_message("Parallel run", "τρέχει και στα δύο μοντέλα. Τα αποτελέσματα θα εμφανιστούν μόλις είναι έτοιμα.")]
	yield base_output, merged_output, "\n".join(statuses)

	with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
		futures = {
			executor.submit(runtime.generate, prompt, max_new_tokens, temperature, top_p): runtime
			for runtime in (base_runtime, merged_runtime)
		}
		for future in concurrent.futures.as_completed(futures):
			runtime = futures[future]
			try:
				result = future.result()
				tokens_per_second = result.token_count / result.seconds if result.seconds > 0 else 0.0
				statuses.append(
					build_status_message(
						runtime.label,
						f"ολοκλήρωσε σε {result.seconds:.2f}s στο {html.escape(result.device)} "
						f"για {result.token_count} tokens ({tokens_per_second:.1f} tok/s).",
					)
				)
				if runtime is base_runtime:
					base_output = result.text
				else:
					merged_output = result.text
			except Exception as exc:
				if runtime is base_runtime:
					base_output = f"[ERROR]\n{exc}"
				else:
					merged_output = f"[ERROR]\n{exc}"
				statuses.append(build_status_message(runtime.label, f"απέτυχε: {html.escape(str(exc))}"))

			yield base_output, merged_output, "\n".join(statuses)


def clear_common() -> Tuple[str, str, str, str]:
	return "", "", "", ""


def clear_single() -> Tuple[str, str, str]:
	return "", "", ""


def build_app(args: argparse.Namespace):
	base_runtime = ModelRuntime(
		label=args.base_label,
		model_ref=args.base_model,
		preferred_device=args.base_device,
		dtype_name=args.base_dtype,
		trust_remote_code=args.trust_remote_code,
		attn_implementation=args.attn_implementation,
	)
	merged_runtime = ModelRuntime(
		label=args.merged_label,
		model_ref=args.merged_model,
		preferred_device=args.merged_device,
		dtype_name=args.merged_dtype,
		trust_remote_code=args.trust_remote_code,
		attn_implementation=args.attn_implementation,
	)

	def compare_models(prompt, tokens, temp, nucleus):
		yield from run_parallel_stream(
			base_runtime,
			merged_runtime,
			prompt,
			int(tokens),
			float(temp),
			float(nucleus),
		)

	meta_html = f"""
	<div class='hero-panel'>
		<div class='lane-chip base'>Parallel compare</div>
		<h1>Apertus Dual Console</h1>
		<p>
			Ένα κοινό prompt στην κορυφή στέλνει ταυτόχρονα ερώτηση στο base <strong>{html.escape(args.base_label)}</strong>
			και στο <strong>{html.escape(args.merged_label)}</strong>. Πιο κάτω έχεις και δύο ανεξάρτητες λωρίδες,
			ώστε να γράφεις διαφορετικό prompt σε κάθε μοντέλο όταν θέλεις one-off έλεγχο.
		</p>
		<div class='meta-grid'>
			<div class='meta-card'>
				<strong>Base model</strong>
				<div class='meta-strip'>{format_path(args.base_model)}</div>
				<div class='meta-strip'>device: {format_path(args.base_device or 'auto')}</div>
				<div class='meta-strip'>dtype: {format_path(args.base_dtype)}</div>
			</div>
			<div class='meta-card'>
				<strong>Merged model</strong>
				<div class='meta-strip'>{format_path(args.merged_model)}</div>
				<div class='meta-strip'>device: {format_path(args.merged_device or 'auto')}</div>
				<div class='meta-strip'>dtype: {format_path(args.merged_dtype)}</div>
			</div>
			<div class='meta-card'>
				<strong>Serve endpoint</strong>
				<div class='meta-strip'>{format_path(args.host)}:{args.port}</div>
				<div class='meta-strip'>πρώτο request φορτώνει weights και θέλει λίγη υπομονή</div>
			</div>
		</div>
	</div>
	"""

	with gr.Blocks(title="Apertus Dual Console") as demo:
		with gr.Column(elem_classes=["app-shell"]):
			gr.HTML(meta_html)

			with gr.Group(elem_classes=["panel-card"]):
				gr.Markdown("## Κοινό Prompt")
				shared_prompt = gr.Textbox(
					label="Μία ερώτηση για τα δύο μοντέλα",
					lines=5,
					value=DEFAULT_SHARED_PROMPT,
					placeholder="Γράψε το κοινό prompt εδώ...",
				)
				with gr.Row():
					shared_send = gr.Button("Ρώτα και τα δύο μοντέλα", variant="primary")
					shared_clear = gr.Button("Καθαρισμός κοινής σύγκρισης")

				with gr.Accordion("Generation settings", open=False):
					with gr.Row():
						max_new_tokens = gr.Slider(
							minimum=64,
							maximum=1024,
							step=32,
							value=args.max_new_tokens,
							label="Max new tokens",
						)
						temperature = gr.Slider(
							minimum=0.0,
							maximum=1.2,
							step=0.05,
							value=args.temperature,
							label="Temperature",
						)
						top_p = gr.Slider(
							minimum=0.1,
							maximum=1.0,
							step=0.05,
							value=args.top_p,
							label="Top-p",
						)

				with gr.Row(equal_height=True):
					with gr.Column(elem_classes=["panel-card", "base-lane"]):
						gr.HTML("<div class='lane-chip base'>Base</div><div class='lane-title'>Apertus 8B</div>")
						shared_base_output = gr.Textbox(
							label=args.base_label,
							lines=20,
							**textbox_copy_kwargs(),
						)
					with gr.Column(elem_classes=["panel-card", "merged-lane"]):
						gr.HTML("<div class='lane-chip merged'>Merged</div><div class='lane-title'>Trained + merged model</div>")
						shared_merged_output = gr.Textbox(
							label=args.merged_label,
							lines=20,
							**textbox_copy_kwargs(),
						)
				shared_status = gr.HTML()

			with gr.Row(equal_height=True):
				with gr.Column(elem_classes=["panel-card", "base-lane"]):
					gr.HTML("<div class='lane-chip base'>Single model lane</div><div class='section-title'>Μόνο Apertus 8B</div>")
					base_prompt = gr.Textbox(
						label="Prompt μόνο για το base model",
						lines=6,
						value=DEFAULT_SINGLE_PROMPT,
					)
					with gr.Row():
						base_send = gr.Button("Ρώτα μόνο το Apertus 8B", variant="primary")
						base_clear = gr.Button("Καθαρισμός")
					base_output = gr.Textbox(
						label="Απάντηση Apertus 8B",
						lines=18,
						**textbox_copy_kwargs(),
					)
					base_status = gr.HTML()

				with gr.Column(elem_classes=["panel-card", "merged-lane"]):
					gr.HTML("<div class='lane-chip merged'>Single model lane</div><div class='section-title'>Μόνο merged model</div>")
					merged_prompt = gr.Textbox(
						label="Prompt μόνο για το merged model",
						lines=6,
						value=DEFAULT_SINGLE_PROMPT,
					)
					with gr.Row():
						merged_send = gr.Button("Ρώτα μόνο το merged", variant="secondary")
						merged_clear = gr.Button("Καθαρισμός")
					merged_output = gr.Textbox(
						label="Απάντηση merged model",
						lines=18,
						**textbox_copy_kwargs(),
					)
					merged_status = gr.HTML()

		shared_send.click(
			fn=compare_models,
			inputs=[shared_prompt, max_new_tokens, temperature, top_p],
			outputs=[shared_base_output, shared_merged_output, shared_status],
		)

		base_send.click(
			fn=lambda prompt, tokens, temp, nucleus: run_single(
				base_runtime,
				prompt,
				int(tokens),
				float(temp),
				float(nucleus),
			),
			inputs=[base_prompt, max_new_tokens, temperature, top_p],
			outputs=[base_output, base_status],
		)

		merged_send.click(
			fn=lambda prompt, tokens, temp, nucleus: run_single(
				merged_runtime,
				prompt,
				int(tokens),
				float(temp),
				float(nucleus),
			),
			inputs=[merged_prompt, max_new_tokens, temperature, top_p],
			outputs=[merged_output, merged_status],
		)

		shared_clear.click(
			fn=clear_common,
			outputs=[shared_prompt, shared_base_output, shared_merged_output, shared_status],
		)
		base_clear.click(fn=clear_single, outputs=[base_prompt, base_output, base_status])
		merged_clear.click(fn=clear_single, outputs=[merged_prompt, merged_output, merged_status])

	demo.queue(default_concurrency_limit=8, api_open=False)
	return demo


def parse_args() -> argparse.Namespace:
	legacy_shared_dtype = os.environ.get("APERTUS_UI_DTYPE", "")
	dtype_choices = ["bfloat16", "float16", "float32"]

	parser = argparse.ArgumentParser(description="Run a dual-model Apertus comparison UI.")
	parser.add_argument("--host", default=os.environ.get("APERTUS_UI_HOST", "0.0.0.0"))
	parser.add_argument("--port", type=int, default=int(os.environ.get("APERTUS_UI_PORT", "8631")))
	parser.add_argument("--base-model", default=os.environ.get("APERTUS_BASE_MODEL", "swiss-ai/Apertus-8B-Instruct-2509"))
	parser.add_argument("--merged-model", default=os.environ.get("APERTUS_MERGED_MODEL", discover_default_merged_model()))
	parser.add_argument("--base-label", default=os.environ.get("APERTUS_BASE_LABEL", "Apertus 8B"))
	parser.add_argument("--merged-label", default=os.environ.get("APERTUS_MERGED_LABEL", "Merged model"))
	parser.add_argument("--base-device", default=os.environ.get("APERTUS_BASE_DEVICE", default_device_for(0) or "auto"))
	parser.add_argument("--merged-device", default=os.environ.get("APERTUS_MERGED_DEVICE", default_device_for(1) or default_device_for(0) or "auto"))
	parser.add_argument("--base-dtype", default=os.environ.get("APERTUS_BASE_DTYPE", legacy_shared_dtype or "float32"), choices=dtype_choices)
	parser.add_argument("--merged-dtype", default=os.environ.get("APERTUS_MERGED_DTYPE", legacy_shared_dtype or "bfloat16"), choices=dtype_choices)
	parser.add_argument("--dtype", dest="legacy_dtype", default="", choices=["", *dtype_choices], help=argparse.SUPPRESS)
	parser.add_argument("--attn-implementation", default=os.environ.get("APERTUS_UI_ATTN_IMPLEMENTATION", ""))
	parser.add_argument("--trust-remote-code", action="store_true", default=parse_bool(os.environ.get("APERTUS_TRUST_REMOTE_CODE", "false")))
	parser.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("APERTUS_UI_MAX_NEW_TOKENS", str(DEFAULT_MAX_NEW_TOKENS))))
	parser.add_argument("--temperature", type=float, default=float(os.environ.get("APERTUS_UI_TEMPERATURE", str(DEFAULT_TEMPERATURE))))
	parser.add_argument("--top-p", type=float, default=float(os.environ.get("APERTUS_UI_TOP_P", str(DEFAULT_TOP_P))))
	args = parser.parse_args()
	if args.legacy_dtype:
		args.base_dtype = args.legacy_dtype
		args.merged_dtype = args.legacy_dtype
	return args


def main() -> None:
	args = parse_args()
	demo = build_app(args)
	launch_kwargs = {
		"server_name": args.host,
		"server_port": args.port,
		"share": False,
		"inbrowser": False,
	}
	if BLOCKS_LAUNCH_SUPPORTS_CSS:
		launch_kwargs["css"] = CUSTOM_CSS
	if BLOCKS_LAUNCH_SUPPORTS_SHOW_API:
		launch_kwargs["show_api"] = False

	demo.launch(
		**launch_kwargs,
	)


if __name__ == "__main__":
	main()