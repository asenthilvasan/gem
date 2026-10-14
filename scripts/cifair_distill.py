""" SupCon -> Memory Wrap fine-tuning with a DVP-style distillation term, for several distillation weights.

Fine-tuning adds weight * mean cosine distance between the student encoder and a frozen copy of the SupCon encoder
(on the query and the memory images) to the cross-entropy loss. Weight 0 is the UMft cell, and a large weight
approaches the frozen UMfz cell. Everything else is the finished cifair_gem.py run's: the UMft setting chosen on
val (lr, weight decay, batch size), the per-seed SupCon encoders, the fine-tuning epochs and the seeds.

Per split:
    1. Train and save one model per (weight, seed) on trainval, evaluated on test0.
    2. Score every seed with memory-wrap-SENN's purity code, next to the UMfz and UMft checkpoints that
       cifair_purity.py saved: accuracy, soft purity, coherence and support size on all test queries; coherence,
       soft purity and focus on queries both models classify correctly (against UMft and UMfz); and drift, the
       mean cosine distance between the model's encoder and the SupCon encoder on the test images.
Every step writes a JSON file and is skipped when it exists.

Example (from the gem/ directory, one split per GPU):
    python scripts/cifair_distill.py --split 0 --run-dir /workspace/gem_results/cifair/split_0 \\
        --purity-dir /workspace/gem_results/cifair_purity/split_0 --out /workspace/gem_results/cifair_distill/split_0
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import contextlib
import io
import re

import numpy as np

from cifair_gem import train_once, read_json, write_json, log, run_arch
from cifair_purity import REPO_PAPER, load_model, loaders

BASELINES = ['UMfz', 'UMft']


def cell_name(weight):
    return f'D{weight:g}'


def drift(model, teacher, test_loader):
    """ Mean cosine distance between the model's encoder and the SupCon encoder on the test images. """
    import torch
    total, n = 0.0, 0
    with torch.no_grad():
        for X, _ in test_loader:
            X = X.cuda()
            cos = torch.nn.functional.cosine_similarity(model.forward_encoder(X), teacher.forward_encoder(X), dim=1)
            total += (1 - cos).sum().item()
            n += len(X)
    return total / n


def score(job):
    import torch
    sys.path.insert(0, os.path.join(REPO_PAPER, 'scripts'))
    import purity_score as ps  # memory-wrap-SENN/paper/scripts/purity_score.py

    split, seed, settings, paths, data_root, redraws, arch = job
    torch.manual_seed(seed)
    models = {cell: load_model(settings[cell], path, arch) for cell, path in paths.items()}
    pipeline = models['UMft'][1]
    test_loader, _ = loaders(pipeline, data_root, split, seed)
    # UMfz trains only the head, so its encoder is the seed's SupCon encoder with its BatchNorm statistics.
    teacher = models['UMfz'][0]
    out = {}
    for cell, (model, _) in models.items():
        # Fresh loaders per cell, so every cell is scored on the same memory draws.
        _, mem_loader = loaders(pipeline, data_root, split, seed)
        soft, coherence = ps.purity(model, test_loader, mem_loader, redraws, 'cuda')
        _, mem_loader = loaders(pipeline, data_root, split, seed)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ps.diagnose(model, test_loader, mem_loader, 'cuda', max_images=2000)
        size = float(re.search(r'Mean support-set size : ([0-9.]+)', buf.getvalue()).group(1))
        out[cell] = {'soft': float(np.mean(soft)), 'coherence': float(np.nanmean(coherence)), 'support_size': size,
                     'drift': drift(model, teacher, test_loader)}
    for a in [c for c in models if c not in BASELINES]:
        for b in BASELINES:
            _, mem_loader = loaders(pipeline, data_root, split, seed)
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
    parser.add_argument('--purity-dir', required=True, help='--out of cifair_purity.py, with the UMfz/UMft checkpoints')
    parser.add_argument('--out', required=True)
    parser.add_argument('--weights', type=float, nargs='+', default=[1, 10, 100, 1000])
    parser.add_argument('--seeds', type=int, default=None, help='number of seeds (default: the run\'s final runs)')
    parser.add_argument('--epochs', type=int, default=None, help='fine-tuning epochs (default: the run\'s)')
    parser.add_argument('--redraws', type=int, default=5, help='memory draws per checkpoint (as in cifair_purity.py)')
    parser.add_argument('--gpus-per-trial', type=float, default=0.1666)
    parser.add_argument('--cpus-per-trial', type=float, default=4)
    parser.add_argument('--load-workers', type=int, default=2)
    parser.add_argument('--ray-cpus', type=int, default=None)
    parser.add_argument('--ray-temp-dir', default=None)
    args = parser.parse_args()

    import ray
    from gem.pipelines.memorywrap import MemoryWrapClassifier
    # GEM accepts unknown hyper-parameters silently, so an old pipeline would train plain UMft.
    if 'distill_weight' not in MemoryWrapClassifier.default_hparams():
        raise RuntimeError('gem/pipelines/memorywrap.py has no distill_weight; sync the code first.')
    run_args = read_json(os.path.join(args.run_dir, 'args.json'))
    data_root = run_args['data_root']
    arch = run_arch(run_args)
    split = args.split
    src = os.path.join(args.run_dir, f'split{split}')
    prev = os.path.join(args.purity_dir, f'split{split}')
    out = os.path.join(args.out, f'split{split}')
    os.makedirs(out, exist_ok=True)
    settings = {cell: read_json(os.path.join(src, f'search_{cell}.json')) for cell in BASELINES}
    cells = [cell_name(w) for w in args.weights]
    for cell in cells:
        settings[cell] = settings['UMft']
    seeds = range(args.seeds or run_args['final_runs'])
    epochs = args.epochs or run_args['down_epochs']
    # Finished files are reused, so they must come from the same epochs and memory draws.
    config = {'epochs': epochs, 'redraws': args.redraws}
    config_path = os.path.join(out, 'config.json')
    if os.path.exists(config_path) and read_json(config_path) != config:
        raise RuntimeError(f'{out} holds results for {read_json(config_path)}, not {config}; use another --out.')
    write_json(config_path, config)

    gem_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    paths = [gem_root, os.path.join(gem_root, 'scripts'), os.environ.get('PYTHONPATH', '')]
    ray.init(num_cpus=args.ray_cpus, include_dashboard=False, _temp_dir=args.ray_temp_dir,
             runtime_env={'env_vars': {'PYTHONPATH': os.pathsep.join(p for p in paths if p)}})
    remote = ray.remote(num_cpus=args.cpus_per_trial, num_gpus=args.gpus_per_trial)

    # 1. Fine-tune with distillation and save the models.
    def train(weight, seed):
        model = os.path.join(out, f'{cell_name(weight)}_seed{seed}.pt')
        extra = {'pretrained_encoder': os.path.join(src, f'supcon_trainval_seed{seed}.pt'), 'distill_weight': weight}
        result = train_once('memorywrap', settings['UMft'], extra, data_root, f'trainval{split}', epochs, seed,
                            args.load_workers, arch=arch, save=model + '.part', test_split='test0')
        os.replace(model + '.part', model)
        return {**result, 'distill_weight': weight}

    jobs = {}
    for seed in seeds:
        for weight in args.weights:
            if not os.path.exists(os.path.join(out, f'final_{cell_name(weight)}_seed{seed}.json')):
                jobs[remote(train).remote(weight, seed)] = (cell_name(weight), seed)
    log(f'split {split}: training {len(jobs)} models, {epochs} epochs, weights {args.weights}')
    pending = list(jobs)
    while pending:
        done, pending = ray.wait(pending, num_returns=1)
        cell, seed = jobs[done[0]]
        result = ray.get(done[0])
        write_json(os.path.join(out, f'final_{cell}_seed{seed}.json'), result)
        log(f'split {split} {cell} seed {seed}: test {result["accuracy"]:.4f} ({result["train_minutes"]:.1f} min)')

    # 2. Purity, coherence, focus and drift, next to the UMfz/UMft checkpoints.
    jobs = {}
    for seed in seeds:
        done_path = os.path.join(out, f'purity_seed{seed}.json')
        if not os.path.exists(done_path) or not all(c in read_json(done_path) for c in BASELINES + cells):
            ckpts = {cell: os.path.join(prev, f'{cell}_seed{seed}.pt') for cell in BASELINES}
            ckpts.update({cell: os.path.join(out, f'{cell}_seed{seed}.pt') for cell in cells})
            jobs[remote(score).remote((split, seed, settings, ckpts, data_root, args.redraws, arch))] = seed
    pending = list(jobs)
    while pending:
        done, pending = ray.wait(pending, num_returns=1)
        seed = jobs[done[0]]
        write_json(os.path.join(out, f'purity_seed{seed}.json'), ray.get(done[0]))
        log(f'split {split} purity seed {seed} done')

    # Summary over seeds: accuracy from each cell's own final runs, metrics from the purity files.
    summary = {}
    for cell in BASELINES + cells:
        base = prev if cell in BASELINES else out
        accs = [read_json(os.path.join(base, f'final_{cell}_seed{seed}.json'))['accuracy'] for seed in seeds]
        purity = [read_json(os.path.join(out, f'purity_seed{seed}.json'))[cell] for seed in seeds]
        summary[cell] = {'accuracy': float(np.mean(accs)), 'accuracy_std': float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
                         **{k: float(np.mean([p[k] for p in purity])) for k in purity[0]}}
        s = summary[cell]
        log(f'split {split} {cell}: acc {100 * s["accuracy"]:.2f} ± {100 * s["accuracy_std"]:.2f}, '
            f'coherence {s["coherence"]:.3f}, soft {s["soft"]:.3f}, support {s["support_size"]:.1f}, drift {s["drift"]:.3f}')
    write_json(os.path.join(out, 'summary.json'), summary)
    log(f'split {split}: all done')


if __name__ == '__main__':
    main()
