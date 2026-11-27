#!/usr/bin/env python3
"""GPTQ W4 (group 128, symmetric, no act-order) of the Qwen3-8B-Base target, for Marlin.

  python scripts/gptq_quantize.py --out checkpoints/qwen3-8b-base-gptq-w4g128

Calibration: the 128 x 512-token RedPajama-V2 calibration sequences in
data/redpajama_v2/calibration.jsonl (Qwen2.5/Qwen3 share the BPE vocabulary).
Only the decoder linear layers are quantized; embeddings, norms and lm_head stay BF16.
The checkpoint is loaded by scripts/quant_spec.py (target scheme 'gptq-w4g128').
"""
import argparse
import json
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from acceleratellm.config import load_config

CALIBRATION = 'data/redpajama_v2/calibration.jsonl'


@contextmanager
def preserve_initializers():
    """GPTQModel's loader replaces torch.nn.init functions; restore them afterwards."""
    from transformers import modeling_utils
    names = ('kaiming_uniform_', 'uniform_', 'normal_')
    originals = {name: getattr(torch.nn.init, name) for name in names}
    flag = modeling_utils._init_weights
    try:
        yield
    finally:
        for name, function in originals.items():
            setattr(torch.nn.init, name, function)
        modeling_utils._init_weights = flag


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=Path('checkpoints/qwen3-8b-base-gptq-w4g128'))
    parser.add_argument('--config', default='configs/experiments/speculative_compiled.yaml')
    args = parser.parse_args()
    if (args.out / 'artifact.json').exists():
        print(f'GPTQ_CHECKPOINT_EXISTS={args.out}', flush=True)
        return
    from gptqmodel import GPTQModel, QuantizeConfig
    from gptqmodel.utils import BACKEND
    spec = load_config(args.config)['models']['target']
    source = snapshot_download(spec['model_id'], revision=spec['revision'], local_files_only=True)
    rows = [json.loads(line)['input_ids'] for line in Path(CALIBRATION).read_text().splitlines()]
    calibration = [{'input_ids': ids, 'attention_mask': [1] * len(ids)} for ids in rows]
    qconfig = QuantizeConfig(bits=4, group_size=128, sym=True, desc_act=False, damp_percent=0.01,
                             damp_auto_increment=0., format='gptq', device='cuda:0', lm_head=False,
                             true_sequential=True, parallel_packing=False)
    torch.manual_seed(42)
    with preserve_initializers():
        wrapper = GPTQModel.load(source, quantize_config=qconfig, torch_dtype=torch.bfloat16,
                                 attn_implementation='sdpa', trust_remote_code=False, local_files_only=True)
    started = time.perf_counter()
    wrapper.quantize(calibration, batch_size=1, calibration_enable_gpu_cache=True, backend=BACKEND.AUTO)
    seconds = time.perf_counter() - started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{args.out.name}-', dir=args.out.parent))
    try:
        wrapper.save(str(staging))
        artifact = {'source_model': spec, 'bits': 4, 'group_size': 128, 'sym': True, 'desc_act': False,
                    'damp_percent': 0.01, 'calibration': CALIBRATION, 'calibration_sequences': len(rows),
                    'sequence_length': len(rows[0]), 'activation_dtype': 'bfloat16', 'lm_head_quantized': False,
                    'quantization_seconds': seconds, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                    'checkpoint_bytes': sum(p.stat().st_size for p in staging.glob('*.safetensors'))}
        (staging / 'artifact.json').write_text(json.dumps(artifact, indent=2) + '\n')
        staging.rename(args.out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f'GPTQ_DONE={args.out} seconds={seconds:.0f}', flush=True)


if __name__ == '__main__':
    main()
