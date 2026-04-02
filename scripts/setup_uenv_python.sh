#!/bin/bash

# This script sets up a Python virtual environment inside a specified uenv image,
# and installs the necessary dependencies for training and inference with the Apertus model.
# It also clones the Apertus finetuning recipes repository if it doesn't already exist.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
uenv_image="${UENV_IMAGE:-prgenv-gnu/24.11:v1}"
uenv_view="${UENV_VIEW:-default}"
venv_dir="${VENV_DIR:-${repo_root}/.venv-uenv}"
apertus_repo_dir="${APERTUS_REPO_DIR:-${repo_root}/external/apertus-finetuning-recipes}"
apertus_repo_url="${APERTUS_REPO_URL:-https://github.com/swiss-ai/apertus-finetuning-recipes.git}"
torch_install_command="${TORCH_INSTALL_COMMAND:-}"

if [[ -z "${torch_install_command}" ]]; then
	torch_install_command='python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu128'
fi

mkdir -p "$(dirname "${apertus_repo_dir}")"

if [[ -e "${apertus_repo_dir}" && ! -d "${apertus_repo_dir}/.git" ]]; then
	echo "APERTUS_REPO_DIR exists but is not a git checkout: ${apertus_repo_dir}" >&2
	exit 1
fi

export repo_root
export venv_dir
export apertus_repo_dir
export apertus_repo_url
export torch_install_command

uenv image pull "${uenv_image}"
uenv run "${uenv_image}" --view="${uenv_view}" -- bash -lc '
set -euo pipefail

if [[ ! -d "${apertus_repo_dir}/.git" ]]; then
	git clone --depth 1 "${apertus_repo_url}" "${apertus_repo_dir}"
fi

python3 -m venv "${venv_dir}"
source "${venv_dir}/bin/activate"
python -m pip install --upgrade pip uv

if ! python -c "import torch" >/dev/null 2>&1; then
	if [[ -n "${torch_install_command}" ]]; then
		eval "${torch_install_command}"
	else
		echo "PyTorch is not installed in ${venv_dir}." >&2
		echo "Set TORCH_INSTALL_COMMAND to a Clariden-compatible install command and rerun." >&2
		exit 1
	fi
fi

uv pip install -r "${apertus_repo_dir}/requirements.txt"
python -m compileall "${repo_root}/scripts"
python - <<"PY"
import torch
print(f"Using torch {torch.__version__}")
PY
'
