# /// script
# dependencies = [
#   "lm-eval[math,ifeval,sentencepiece]",
#   "wandb",
#   "hf_transfer",
# ]
# ///

import lm_eval
from lm_eval.loggers import WandbLogger
import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "model_name",
    help="Name of the model to evaluate, e.g., 'microsoft/phi-2'",
    type=str,
)

args = parser.parse_args()

results = lm_eval.simple_evaluate(
    model="hf",
    model_args={
        "pretrained": args.model_name,
        "trust_remote_code": True,
    },

    # leaderboard (mmlu-pro, bbh, gpqa, math-hard, ifeval, musr)
    tasks=[
      # "boolq",
      # "piqa",
      # "openbookqa",
      # "winogrande",
      # "mmlu",
      # "lambada",
      # "kobest",
      # "kmmlu",
      # "ifeval",
      # "humaneval",
      # "hrm8k",
      # "hellaswag",
      # "haerae",
      # "gsm8k",
      # "gpqa",
      # "eq_bench",
      "leaderboard",
      # "commonsense_qa",
      # "ai2_arc"
    ],

    # too long to run
    # "bbq",
    # "coqa",
    # "drop",

    log_samples=True,
    batch_size="auto"
)

wandb_logger = WandbLogger(
  init_args={
    "project": "fast-eval",
    "job_type": "eval",
    "name": f"eval-{datetime.datetime.now().strftime('%Y%m%d')}-{args.model_name}"
  },
)
wandb_logger.post_init(results)
wandb_logger.log_eval_result()
wandb_logger.log_eval_samples(results["samples"])