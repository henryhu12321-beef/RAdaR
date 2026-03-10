# RAdaR: RL-Native Adaptive Reasoning with Capability-Aware Data Curation for VLMs
## 📝 Abstract
Vision-Language Models (VLMs) excel in complex reasoning tasks but are often constrained by the issue of overthinking and underthinking, limiting their applicability in real-world scenarios. Existing adaptive reasoning approaches face critical challenges, including data scarcity, catastrophic forgetting, and sensitivity to prompts. To address these limitations, we propose AdaR, an RL-native framework for adaptive reasoning with a two-stage training process: In Stage 1, the model is trained to produce outputs that follow the prescribed formats of the two reasoning modes, Thinking and Instant, using a curriculum-style formatting strategy. In Stage 2, we model adaptive reasoning as a reasoning mode selection problem and train the model to dynamically choose appropriate strategies for each input by predicting the corresponding mode-control start token. In addition, we introduce a capability-aware data construction pipeline that provides highly discriminative supervision for adaptive reasoning in VLMs. Experimental results demonstrate that RAdaR achieves a significant reduction in reasoning overhead while improving accuracy by up to 7.6% over the base model and 14.7% over SOTA methods, respectively. To ensure reproducibility and promote further research, we will release the code, datasets, and model weights.

![Three-stage pipeline](figures/framework.png)

## 🚀 Getting Started

## 🔧 0. Before You Start
### Installation
1. Clone the repository:
```bash
git clone https://github.com/henryhu12321-beef/RAdaR.git
cd RAdaR
```

2. Create a conda environment and install dependencies:
```bash
conda create -n radar python=3.10 -y
conda activate radar

# Install core dependencies (adjust torch version based on your CUDA)
pip install torch==2.8.0 torchvision==0.23.0
pip install transformers==4.57.1 sglang==0.5.5.post1 vllm==0.11.0
pip install flash-attn --no-build-isolation

# Install AReaL package in editable mode
pip install -e .
```

## 🚀 1. Stage 1.1
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m areal.launcher.local examples/vlm/radar_stage1_1_train.py --config examples/vlm/radar_gspo_stage1_1_bs32_rollout32.yaml
```

## 🚀 2. Stage 1.2
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m areal.launcher.local examples/vlm/radar_stage1_2_train.py --config examples/vlm/radar_gspo_stage1_2_bs32_rollout32.yaml
```

## 🚀 3. Stage 2
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m areal.launcher.local examples/vlm/radar_stage2_train.py --config examples/vlm/radar_gspo_stage2_bs32_rollout32.yaml
```

## 📖 Resources

- [Installation](https://inclusionai.github.io/AReaL/tutorial/installation.html)
- [Quickstart](https://inclusionai.github.io/AReaL/tutorial/quickstart.html)
- [CLI Configurations](https://inclusionai.github.io/AReaL/cli_reference.html)
- [Asynchronous RL Explained](https://inclusionai.github.io/AReaL/algorithms/async.html)
- [Fine-Tuning Large MoE](https://inclusionai.github.io/AReaL/tutorial/megatron.html)
- [Agentic RL](https://inclusionai.github.io/AReaL/tutorial/agentic_rl.html)
- [Debugging Best Practices](https://inclusionai.github.io/AReaL/best_practices/debugging.html)
- [Handling OOM Issues](https://inclusionai.github.io/AReaL/best_practices/handling_oom.html)

## 📄 Citation

```bibtex
@misc{fu2025areal,
      title={AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning},
      author={Wei Fu and Jiaxuan Gao and Xujie Shen and Chen Zhu and Zhiyu Mei and Chuyi He and Shusheng Xu and Guo Wei and Jun Mei and Jiashu Wang and Tongkai Yang and Binhang Yuan and Yi Wu},
      year={2025},
      eprint={2505.24298},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.24298},
}
```
