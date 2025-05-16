import ast
import logging
import subprocess
from typing import Any

import numpy as np


def run_llama_embedding(checkpoint_path: str,
                        prompt: str) -> np.ndarray[Any, np.dtype[np.float32]]:
    cmd = f'llama-embedding -m {checkpoint_path} -p "{prompt}" \
            --pooling none --no-warmup --embd-normalize -1 \
            --batch-size 512 --ctx-size 512 \
            --embd-output-format array'

    logging.info(f"Running llama.cpp: {cmd}")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to run llama-embedding:{e.stderr}")
        raise e
    try:
        embeddings = np.array(ast.literal_eval(result.stdout)).astype(
            np.float32)
    except Exception as e:
        logging.error(f"Failed to parse llama-embedding output:{e}")
        raise e
    return embeddings
