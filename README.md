# Re-Imagining Multimodal Instruction Tuning: A Representation View

🔥🔥🔥 Our [arxiv](https://arxiv.org/abs/2503.00723) version is currently available. Please check it out! 🔥🔥🔥

<div align="center">
  <img src=".\images\mainfig.png">
</div>
<p align="center">
 Figure1: Overview of our MRT approach. Representation editors ψ ∈ {ψV , ψc, ψP , ψS} are the only tunable
parameters while the entire model remains completely frozen. During fine-tuning, we jointly edit
the visual representations in the vision encoder, the cross-modality layer, and the prefix and suffix
of textual-oriented fraction in the multimodal representations in the LLM. These editors efficiently
and effectively optimize the model representations during multimodal instruction tuning.
</p>

## Installation

```bash
git clone https://github.com/comeandcode/MRT.git
cd MRT

conda create -n mrt python=3.9 -y
conda activate mrt
pip install packaging
pip install -e . --no-cache-dir
pip install numpy==1.26.4
pip install ninja
pip install transformers==4.31.0
pip install torch==2.0.1
pip install flash-attn==2.6.3 --no-build-isolation --no-cache-dir

```
## LLaVA_align Weights

The weigth for stage-1 LLaVA is [liuhaotian/llava-pretrain-vicuna-7b-v1.3](https://huggingface.co/liuhaotian/llava-pretrain-vicuna-7b-v1.3) and  [lmsys/vicuna-7b-v1.3](https://huggingface.co/lmsys/vicuna-7b-v1.3) please download it for MRT.

## Usage

```bash
# Train
sh train.sh

# Eval
sh eval.sh

```
## License


Copyright 2025 Re-Imagining Multimodal Instruction Tuning: A Representation View Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

