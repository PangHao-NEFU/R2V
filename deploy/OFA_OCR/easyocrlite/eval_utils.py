# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

import string
import math
import json
from itertools import chain
import os

import torch
import torch.distributed as dist
from OFA_OCR.fairseq import utils

from data import data_utils
from tasks.nlg_tasks.gigaword import fix_tokenization


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.bos, generator.eos}


def decode_fn(x, tgt_dict, bpe, generator, tokenizer=None):
    x = tgt_dict.string(x.int().cpu(), extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator))
    if bpe is not None:
        x = bpe.decode(x)
    if tokenizer is not None:
        x = tokenizer.decode(x)
    return x


def eval_ocr(task, generator, models, sample, **kwargs):
    gen_out = task.inference_step(generator, models, sample)
    hyps, refs, results = [], [], []
    for i, sample_id in enumerate(sample["id"].tolist()):
        decode_tokens = decode_fn(gen_out[i][0]["tokens"], task.tgt_dict, task.bpe, generator).strip()
        hyps.append(decode_tokens.strip().replace(" ", ""))
        if sample["target"]:
            refs.append(
                decode_fn(
                    utils.strip_pad(sample["target"][i], task.tgt_dict.pad()),
                    task.tgt_dict, task.bpe, generator
                )
                .strip()
                .replace(" ", "")
            )
        results.append(
            {
                "image_id": str(sample_id),
                "ocr": decode_tokens.strip().replace(" ", ""),
            }
        )
    if refs:
        acc = [1.0 if hyp == ref else 0.0 for hyp, ref in zip(hyps, refs)]
    else:
        acc = None

    return results, acc


def eval_step(task, generator, models, sample, **kwargs):
    if task.cfg._name == "ocr":
        return eval_ocr(task, generator, models, sample, **kwargs)
    else:
        raise NotImplementedError
