"""Preserve the existing two experts and initialize an untrained recovery expert."""

import argparse
import json

from unilab.algos.torch.distill.recovery_integration import extend_recovery_student_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    print(
        json.dumps(
            extend_recovery_student_checkpoint(args.source, args.output, seed=args.seed), indent=2
        )
    )


if __name__ == "__main__":
    main()
