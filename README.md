# Fed-GRPO
Fed-GRPO: Reward-Signal-Driven Federated Group Relative Policy Optimization

## Installation

Our code is based on Python 3.12 and PyTorch 2.8.0.
You can install all the dependencies with the following command:
```shell
conda create -n fed-grpo python=3.12
conda activate fed-grpo
pip install -r requirements.txt
```

## Training

You can train an LLM with Fed-GRPO using the unified training script:

```shell
accelerate launch \
    --main_process_port 29500 \
    --num_processes=8 \
    src/train.py --config config/train.yaml --method fed_grpo
```

A SLURM submission script is also provided:
```shell
sbatch slurm/submit_train.submit
```

## Evaluation

To evaluate a trained checkpoint on a math reasoning benchmark:

```shell
python src/eval.py \
    --checkpoint /path/to/checkpoint \
    --config config/eval.yaml \
    --dataset math500 \
    --num_gpus 2 \
    --output_dir /path/to/output
```

A SLURM submission script that evaluates all checkpoints on all datasets is also provided:
```shell
sbatch slurm/submit_eval.submit
```

## Acknowledgement

We would like to thank the authors for releasing the public repositories: [TRL](https://github.com/huggingface/trl) and [vLLM](https://github.com/vllm-project/vllm).
