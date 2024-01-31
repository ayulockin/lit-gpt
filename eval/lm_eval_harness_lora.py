import os
import json
import sys
import wandb
from pathlib import Path
from typing import List, Literal, Optional

import lightning as L
import torch
from lightning.fabric.plugins import BitsandbytesPrecision
from lm_eval.base import BaseLM

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
from lm_eval_harness import EvalHarnessBase

from lit_gpt import Tokenizer
from lit_gpt.lora import GPT, Config, merge_lora_weights
from lit_gpt.utils import check_valid_checkpoint_dir, get_default_supported_precision, load_lora_checkpoint


def generate_prompt(example):
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""

    if example["input"]:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:"
        )
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{example['instruction']}\n\n### Response:"
    )


lora_r = 4
lora_alpha = 16
lora_dropout = 0.05
lora_query = True
lora_key = True
lora_value = True
lora_projection = False
lora_mlp = False
lora_head = False


class EvalHarnessLoRA(EvalHarnessBase):
    def __init__(
        self,
        fabric: L.Fabric, model: GPT, tokenizer: Tokenizer,
        input: str = "",
        batch_size=1,
    ):
        super(BaseLM, self).__init__()
        self.input = input
        self.fabric = fabric
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size_per_gpu = batch_size
        with fabric.init_tensor():
            model.set_kv_cache(batch_size=batch_size)

    def tok_encode(self, string: str):
        sample = {"instruction": string, "input": self.input}
        prompt = generate_prompt(sample)
        return super().tok_encode(prompt)
    
@torch.inference_mode()
def run_eval_harness(
    checkpoint_dir: Path,
    lora_path: Path,
    input: str = "",
    precision: Optional[str] = None,
    eval_tasks: List[str] = ["arc_challenge", "piqa", "hellaswag", "hendrycksTest-*"],
    num_fewshot: int = 0,
    bootstrap_iters: int = 100000,
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8"]] = None,
    save_filepath: Optional[Path] = None,
    log_to_wandb: bool = True,
    limit: Optional[int] = None,
    no_cache: bool = True,
):
    if precision is None:
        precision = get_default_supported_precision(training=False)

    plugins = None
    if quantize is not None and quantize.startswith("bnb."):
        if "mixed" in precision:
            raise ValueError("Quantization and mixed precision is not supported.")
        dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
        plugins = BitsandbytesPrecision(quantize[4:], dtype)
        precision = None

    fabric = L.Fabric(devices=1, precision=precision, plugins=plugins)

    check_valid_checkpoint_dir(checkpoint_dir)
    tokenizer = Tokenizer(checkpoint_dir)

    with open(checkpoint_dir / "lit_config.json") as fp:
        config_params = dict(
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            to_query=lora_query,
            to_key=lora_key,
            to_value=lora_value,
            to_projection=lora_projection,
            to_mlp=lora_mlp,
            to_head=lora_head,
        )
        config_params.update(**json.load(fp))
        config = Config(**config_params)

    checkpoint_path = checkpoint_dir / "lit_model.pth"

    print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    with fabric.init_module(empty_init=True):
        model = GPT(config)

    load_lora_checkpoint(fabric, model, checkpoint_path, lora_path)
    model.eval()
    merge_lora_weights(model)
    model = fabric.setup_module(model)

    eval_harness = EvalHarnessLoRA(fabric, model, tokenizer, input, 1)

    results = eval_harness.run_eval(eval_tasks, num_fewshot, limit, bootstrap_iters, no_cache)

    if log_to_wandb:
        config = results["config"]
        config.update(
            dict(
                checkpoint_dir = checkpoint_dir,
                precision = precision,
                batch_size = eval_harness.batch_size,
                eval_tasks = eval_tasks,
                num_fewshot = num_fewshot,
                bootstrap_iters = bootstrap_iters,
                device = eval_harness.device,
                devices = 1,
                quantize = quantize,
                save_filepath = save_filepath,
            )
        )
        run = wandb.init(
            project="llm-finetuning",
            job_type="eval-harness",
            config=results["config"]
        )
        
        results_tmp = results["results"]
        tasks = list(results_tmp.keys())

        for task in tasks:
            wandb.log({f"{task}/{k}": v for k, v in results_tmp[task].items()}, commit=False)
        wandb.log({})

    if save_filepath is None:
        print(results)
    else:
        print(f"Saving results to {str(save_filepath)!r}")
        data = json.dumps(results)
        with open(save_filepath, "w") as fw:
            fw.write(data)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    CLI(run_eval_harness, as_positional=False)
