""" Purity and coherence of the Memory Wrap cells from a finished cifair_gem.py run.

The final runs of cifair_gem.py do not keep their Memory Wrap weights, so this script retrains them with the
same chosen settings and seeds (SupCon cells reuse the saved per-seed encoders), saves them, and scores them
with memory-wrap-SENN's own purity code (paper/scripts/purity_score.py), as in the SVHN evaluation:
    - single model, all test queries: soft purity and coherence;
    - model pairs, only queries both classify correctly: coherence, soft purity and focus;
    - support-set size (one memory draw on 2,000 test images).
Accuracy is reported from the same retrained checkpoints. Every step writes a JSON file and is skipped when
it exists.

Example (from the gem/ directory, one split per GPU):
    python scripts/cifair_purity.py --split 0 --run-dir /workspace/gem_results/cifair/split_0 \\
        --out /workspace/gem_results/cifair_purity/split_0

To retrain only Scratch MW with a longer schedule (same chosen settings and seeds) and score it against the
SupCon models of an earlier run, add --scratch-epochs 1000 --reuse-supcon /workspace/gem_results/cifair_purity/split_0
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import contextlib
import io
import json
import re

import numpy as np

from cifair_gem import DATASET, train_once, read_json, write_json, log, run_arch

REPO_PAPER = '/workspace/memory-wrap-SENN/paper'
CELLS = {'UMfz': 'SupCon -> MW, frozen', 'UMft': 'SupCon -> MW, fine-tuned', 'SM': 'Scratch + MW'}
PAIRS = [('UMft', 'SM'), ('UMfz', 'SM'), ('UMft', 'UMfz')]


def load_model(setting, path, arch):
    import torch
    from gem.loader import InstanceLoader
    hparams = {k: v for k, v in setting['hparams'].items() if k not in ('pretrained_encoder', 'freeze_encoder')}
    pipeline = InstanceLoader('pipelines').build_instance('memorywrap', lr=setting['lr'],
                                                          weight_decay=setting['decay'], **hparams)
    model = pipeline.create_model(arch, 10, 3)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    return model.cuda().eval(), pipeline


def loaders(pipeline, data_root, split, seed):
    """ memory-wrap-SENN's purity loaders: test batches of 500, memory batches of 100 unaugmented training images. """
    import torch
    from gem.loader import InstanceLoader
    datasets = InstanceLoader('datasets')
    trainval = datasets.build_instance(DATASET, data_root, f'trainval{split}')
    _, test_transform = pipeline.get_data_transforms(trainval)
    trainval.transform = test_transform
    test = datasets.build_instance(DATASET, data_root, 'test0')
    test.transform = test_transform
    test_loader = torch.utils.data.DataLoader(test, batch_size=500, shuffle=False, pin_memory=True)
    mem_loader = torch.utils.data.DataLoader(trainval, batch_size=100, shuffle=True, drop_last=True, pin_memory=True,
                                             generator=torch.Generator().manual_seed(seed))
    return test_loader, mem_loader


def score(job):
    """ Purity metrics for one seed: every single cell, every pair, and support-set sizes. """
    import torch
    sys.path.insert(0, os.path.join(REPO_PAPER, 'scripts'))
    import purity_score as ps  # memory-wrap-SENN/paper/scripts/purity_score.py

    split, seed, settings, paths, data_root, redraws, arch = job
    torch.manual_seed(seed)
    models = {cell: load_model(settings[cell], paths[cell], arch) for cell in CELLS}
    test_loader, mem_loader = loaders(models['SM'][1], data_root, split, seed)
    out = {}
    for cell, (model, _) in models.items():
        soft, coherence = ps.purity(model, test_loader, mem_loader, redraws, 'cuda')
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ps.diagnose(model, test_loader, mem_loader, 'cuda', max_images=2000)
        size = float(re.search(r'Mean support-set size : ([0-9.]+)', buf.getvalue()).group(1))
        out[cell] = {'soft': float(np.mean(soft)), 'coherence': float(np.nanmean(coherence)), 'support_size': size}
    for a, b in PAIRS:
        coh_a, coh_b, foc_a, foc_b, sft_a, sft_b, n = ps.compare_coherence(
            models[a][0], models[b][0], test_loader, mem_loader, redraws, 'cuda')
        out[f'{a}_vs_{b}'] = {'coherence_a': float(np.nanmean(coh_a)), 'coherence_b': float(np.nanmean(coh_b)),
                              'soft_a': float(np.nanmean(sft_a)), 'soft_b': float(np.nanmean(sft_b)),
                              'focus_a': float(np.nanmean(foc_a)), 'focus_b': float(np.nanmean(foc_b)),
                              'both_correct': float(np.mean(n))}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--split', type=int, required=True)
    parser.add_argument('--run-dir', required=True, help='--out of the finished cifair_gem.py run')
    parser.add_argument('--out', required=True)
    parser.add_argument('--redraws', type=int, default=5, help='memory draws per checkpoint (as on SVHN)')
    parser.add_argument('--gpus-per-trial', type=float, default=0.1666)
    parser.add_argument('--cpus-per-trial', type=float, default=4)
    parser.add_argument('--load-workers', type=int, default=2)
    parser.add_argument('--ray-cpus', type=int, default=None)
    parser.add_argument('--ray-temp-dir', default=None)
    parser.add_argument('--scratch-epochs', type=int, default=None, help='Scratch MW epochs (default: the run\'s --epochs)')
    parser.add_argument('--reuse-supcon', default=None,
                        help='--out of an earlier run of this script whose SupCon MW models are linked instead of retrained')
    args = parser.parse_args()

    import ray
    run_args = read_json(os.path.join(args.run_dir, 'args.json'))
    data_root = run_args['data_root']
    arch = run_arch(run_args)
    split = args.split
    src = os.path.join(args.run_dir, f'split{split}')
    out = os.path.join(args.out, f'split{split}')
    os.makedirs(out, exist_ok=True)
    settings = {cell: read_json(os.path.join(src, f'search_{cell}.json')) for cell in CELLS}
    seeds = range(run_args['final_runs'])
    if args.reuse_supcon:
        prev = os.path.join(args.reuse_supcon, f'split{split}')
        for seed in seeds:
            for cell in CELLS:
                if cell == 'SM':
                    continue
                for name in (f'{cell}_seed{seed}.pt', f'final_{cell}_seed{seed}.json'):
                    if not os.path.exists(os.path.join(out, name)):
                        os.symlink(os.path.join(prev, name), os.path.join(out, name))

    gem_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    paths = [gem_root, os.path.join(gem_root, 'scripts'), os.environ.get('PYTHONPATH', '')]
    ray.init(num_cpus=args.ray_cpus, include_dashboard=False, _temp_dir=args.ray_temp_dir,
             runtime_env={'env_vars': {'PYTHONPATH': os.pathsep.join(p for p in paths if p)}})
    remote = ray.remote(num_cpus=args.cpus_per_trial, num_gpus=args.gpus_per_trial)

    # 1. Retrain and save the final Memory Wrap models.
    def retrain(cell, seed):
        model = os.path.join(out, f'{cell}_seed{seed}.pt')
        extra = {}
        epochs = args.scratch_epochs or run_args['epochs']
        if cell != 'SM':
            extra = {'pretrained_encoder': os.path.join(src, f'supcon_trainval_seed{seed}.pt')}
            epochs = run_args['down_epochs']
        result = train_once('memorywrap', settings[cell], extra, data_root, f'trainval{split}', epochs, seed,
                            args.load_workers, arch=arch, save=model + '.part', test_split='test0')
        os.replace(model + '.part', model)
        return result

    jobs = {}
    for seed in seeds:
        for cell in CELLS:
            if not os.path.exists(os.path.join(out, f'final_{cell}_seed{seed}.json')):
                jobs[remote(retrain).remote(cell, seed)] = (cell, seed)
    log(f'split {split}: retraining {len(jobs)} Memory Wrap models')
    pending = list(jobs)
    while pending:
        done, pending = ray.wait(pending, num_returns=1)
        cell, seed = jobs[done[0]]
        result = ray.get(done[0])
        write_json(os.path.join(out, f'final_{cell}_seed{seed}.json'), result)
        log(f'split {split} {cell} seed {seed}: test {result["accuracy"]:.4f}')

    # 2. Purity, coherence and focus with memory-wrap-SENN's code.
    jobs = {}
    for seed in seeds:
        if not os.path.exists(os.path.join(out, f'purity_seed{seed}.json')):
            ckpts = {cell: os.path.join(out, f'{cell}_seed{seed}.pt') for cell in CELLS}
            jobs[remote(score).remote((split, seed, settings, ckpts, data_root, args.redraws, arch))] = seed
    pending = list(jobs)
    while pending:
        done, pending = ray.wait(pending, num_returns=1)
        seed = jobs[done[0]]
        write_json(os.path.join(out, f'purity_seed{seed}.json'), ray.get(done[0]))
        log(f'split {split} purity seed {seed} done')
    log(f'split {split}: all done')


if __name__ == '__main__':
    main()
