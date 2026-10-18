"""Compatibility entry point; method implementation lives in ours/method/."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ours.method.training import (
    ablation_penalty, export_model, load_pools, padded, support,
    targets_without_decoder_bias, train, validation,
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'method', 'model', 'codes', 'train-pool', 'validation-pool', 'variant', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--bits', type=int, required=True)
    args = parser.parse_args()
    train(args.config, args.method, args.model, args.codes, args.train_pool,
          args.validation_pool, args.bits, args.variant, args.output)
