import ast
import logging
import subprocess

import numpy as np


def run_llama_embedding(checkpoint_path, prompt:str)->np.ndarray:
    cmd = f'llama-embedding -m {checkpoint_path} -p "{prompt}" --pooling none --embd-normalize -1 --no-warmup --batch-size 512 --ctx-size 512 --embd-output-format array'
    logging.info(f"Running llama.cpp: {cmd}")
    result = subprocess.run(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        embeddings = np.array(ast.literal_eval(result.stdout)).astype(np.float32)
    except Exception as e:
        logging.error(f"Failed to run llama-embedding: {e}, {result.stderr}")
        raise e
    return embeddings