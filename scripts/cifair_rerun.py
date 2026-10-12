""" Reruns the final runs of a scratch cell from a finished cifair_gem.py run with a different epoch budget.

The chosen setting and the seeds are the run's own; only the number of epochs changes. Each final run writes
final_{cell}_seed{k}.json under --out and is skipped when it exists.

Example (from the gem/ directory, one split per GPU):
    python scripts/cifair_rerun.py --split 0 --cell SL --epochs 600 \\
        --run-dir /workspace/gem_results/cifair/split_0 --out /workspace/gem_results/cifair_sl600/split_0
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import statistics

from cifair_gem import SCRATCH_CELLS, train_once, read_json, write_json, log


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--split', type=int, required=True)
    parser.add_argument('--cell', choices=SCRATCH_CELLS, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--run-dir', required=True, help='--out of the finished cifair_gem.py run')
    parser.add_argument('--out', required=True)
    parser.add_argument('--gpus-per-trial', type=float, default=0.1)
    parser.add_argument('--cpus-per-trial', type=float, default=4)
    parser.add_argument('--load-workers', type=int, default=2)
    parser.add_argument('--ray-cpus', type=int, default=None)
    parser.add_argument('--ray-temp-dir', default=None)
    args = parser.parse_args()

    import ray
    run_args = read_json(os.path.join(args.run_dir, 'args.json'))
    data_root = run_args['data_root']
    split, cell = args.split, args.cell
    setting = read_json(os.path.join(args.run_dir, f'split{split}', f'search_{cell}.json'))
    out = os.path.join(args.out, f'split{split}')
    os.makedirs(out, exist_ok=True)
    pipeline = 'linear' if cell == 'SL' else 'memorywrap'

    gem_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    paths = [gem_root, os.path.join(gem_root, 'scripts'), os.environ.get('PYTHONPATH', '')]
    ray.init(num_cpus=args.ray_cpus, include_dashboard=False, _temp_dir=args.ray_temp_dir,
             runtime_env={'env_vars': {'PYTHONPATH': os.pathsep.join(p for p in paths if p)}})
    remote = ray.remote(num_cpus=args.cpus_per_trial, num_gpus=args.gpus_per_trial)

    def run(seed):
        return train_once(pipeline, setting, {}, data_root, f'trainval{split}', args.epochs, seed,
                          args.load_workers, test_split='test0')

    seeds = range(run_args['final_runs'])
    jobs = {remote(run).remote(seed): seed for seed in seeds
            if not os.path.exists(os.path.join(out, f'final_{cell}_seed{seed}.json'))}
    log(f'split {split}: {len(jobs)} final {cell} runs at {args.epochs} epochs')
    pending = list(jobs)
    while pending:
        done, pending = ray.wait(pending, num_returns=1)
        seed = jobs[done[0]]
        result = ray.get(done[0])
        write_json(os.path.join(out, f'final_{cell}_seed{seed}.json'), result)
        log(f'split {split} {cell} seed {seed}: test {result["accuracy"]:.4f}')
    accs = [read_json(os.path.join(out, f'final_{cell}_seed{seed}.json'))['accuracy'] for seed in seeds]
    log(f'split {split} {cell} at {args.epochs} epochs: {100 * statistics.mean(accs):.2f} ± '
        f'{100 * statistics.stdev(accs):.2f}; all done')


if __name__ == '__main__':
    main()
