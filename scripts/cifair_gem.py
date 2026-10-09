""" GEM protocol on ciFAIR-10 for SupCon pretraining with linear and Memory Wrap heads, plus scratch baselines.

Cells (same names as memory-wrap-SENN's svhn_eval.py):
    ULfz  SupCon -> linear head, frozen encoder (linear probe)
    UMfz  SupCon -> Memory Wrap, frozen encoder
    UMft  SupCon -> Memory Wrap, fine-tuned encoder
    SM    Memory Wrap from scratch
    SL    linear head (cross-entropy) from scratch

Per split i (train{i}: 300 images, val{i}: 200, trainval{i}: 500, test0: 10,000), in priority order:
    1. ASHA search for SupCon pretraining on train{i}, scored by a linear probe on val{i}.
    2. Pretrain the chosen SupCon setting on train{i}; this encoder is used for the downstream searches.
    3. ASHA search for each downstream cell (ULfz, UMfz, UMft) on train{i}/val{i}.
    4. Final runs, one per seed: pretrain SupCon on trainval{i}, train every downstream cell on that
       same encoder, evaluate on test0.
    5. SM and SL: ASHA search on train{i}/val{i}, then final runs on trainval{i}, evaluated on test0.

Every search and every final run writes a JSON file into --out and is skipped when it exists,
so rerunning the same command resumes after a crash.

Example (from the gem/ directory):
    python scripts/cifair_gem.py --splits 0 --out results/cifair_gem
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import datetime
import json
import random
import time
import urllib.request

import numpy as np

GEM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

DATASET = 'cifair10'
ARCH = 'mobilenet'
METRIC = 'accuracy_avg5'
SUPCON_CELLS = ['ULfz', 'UMfz', 'UMft']
SCRATCH_CELLS = ['SM', 'SL']
CELL_NAMES = {'ULfz': 'SupCon -> Linear, frozen', 'UMfz': 'SupCon -> MW, frozen',
              'UMft': 'SupCon -> MW, fine-tuned', 'SM': 'Scratch + MW', 'SL': 'Scratch + Linear'}


def log(msg):
    print(f'[{datetime.datetime.now().strftime("%m-%d %H:%M:%S")}] {msg}', flush=True)


def slack(msg):
    url = os.environ.get('SLACK_WEBHOOK')
    if not url:
        return
    try:
        req = urllib.request.Request(url, data=json.dumps({'text': msg}).encode(),
                                     headers={'Content-Type': 'application/json'})
        urllib.request.urlopen(req, timeout=20).read()
    except Exception as e:  # a failed ping must never stop the run
        log(f'slack ping failed: {e}')


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, data):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


# --- search spaces -----------------------------------------------------------

def search_space(cell, args):
    """ Returns (pipeline, sampled hyper-parameters, fixed pipeline hyper-parameters, epochs, grace period). """
    from ray import tune

    # GEM's defaults (scripts/hpo.py) for every supervised stage.
    gem_space = {
        'lr': tune.loguniform(1e-4, 0.1),
        'decay': tune.loguniform(1e-5, 0.1),
        'batch_size': tune.choice([10, 25, 50]),
    }
    # GEM's ciFAIR-10 augmentation (bash_scripts/bench_cifair10.sh).
    basic_aug = {'rand_shift': 4, 'hflip': True}

    if cell == 'supcon':
        space = {
            # Wider than GEM's lr range: SupCon on SVHN-500 was best at 0.25.
            'lr': tune.loguniform(1e-3, 0.5),
            'decay': tune.loguniform(1e-5, 0.1),
            # Larger than GEM's batches: SupCon needs several same-class images per batch.
            'batch_size': tune.choice([64, 128, 256]),
            'hparams': {
                'temperature': tune.loguniform(0.05, 0.5),
                'projection_head': tune.choice(['identity', 'mlp']),
            },
        }
        # No temperature / base_temperature loss scaling, as in memory-wrap-SENN's pretrain_supcon.py.
        fixed = {'loss': 'supcon', 'base_temperature': None, 'hflip': True, 'contrastive_aug': True}
        return 'contrastive', space, fixed, args.epochs, args.grace
    if cell in SUPCON_CELLS:
        pipeline = 'linear' if cell == 'ULfz' else 'memorywrap'
        fixed = {**basic_aug, 'freeze_encoder': cell != 'UMft'}
        return pipeline, dict(gem_space), fixed, args.down_epochs, args.down_grace
    if cell == 'SM':
        return 'memorywrap', dict(gem_space), dict(basic_aug), args.epochs, args.grace
    if cell == 'SL':
        return 'linear', dict(gem_space), dict(basic_aug), args.epochs, args.grace
    raise ValueError(cell)


# --- training ----------------------------------------------------------------

def train_once(pipeline_name, setting, extra_hparams, data_root, train_split, epochs, seed,
               load_workers, save=None, test_split=None):
    """ Trains one model with a chosen setting; optionally saves it and evaluates it on `test_split`. """
    import torch
    from gem.loader import InstanceLoader

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    hparams = {**setting['hparams'], **extra_hparams}
    pipeline = InstanceLoader('pipelines').build_instance(
        pipeline_name, lr=setting['lr'], weight_decay=setting['decay'], **hparams)
    datasets = InstanceLoader('datasets')
    train_data = datasets.build_instance(DATASET, data_root, train_split)
    t0 = time.time()
    model, metrics = pipeline.train(
        train_data, None, batch_size=setting['batch_size'], epochs=epochs, architecture=ARCH,
        show_progress=False, load_workers=load_workers)
    result = {'train_loss': float(metrics['loss'][-1]), 'train_accuracy': float(metrics['accuracy'][-1]),
              'epochs': epochs, 'train_minutes': (time.time() - t0) / 60}
    if save is not None:
        torch.save(model.state_dict(), save)
    if test_split is not None:
        test_data = datasets.build_instance(DATASET, data_root, test_split)
        # Training ran without validation data, so nothing has set the test transform yet; a fresh copy of the
        # training split gives the same normalization statistics.
        fresh_train = datasets.build_instance(DATASET, data_root, train_split)
        test_data.transform = pipeline.get_data_transforms(fresh_train)[1]
        # Batch 500 matches memory-wrap-SENN's test loader; Memory Wrap averages 5 memory draws.
        scores = pipeline.evaluate(model, test_data, batch_size=500)
        result.update({key: float(value) for key, value in scores.items()})
    return result


class Runner:

    def __init__(self, args):
        self.args = args
        self.data_root = os.path.abspath(args.data_root)
        self.out = os.path.abspath(args.out)
        os.makedirs(self.out, exist_ok=True)

    def path(self, split, name):
        d = os.path.join(self.out, f'split{split}')
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, name)

    # One task = one final run; Ray schedules them like trials.
    def remote(self, fn):
        import ray
        return ray.remote(num_cpus=self.args.cpus_per_trial, num_gpus=self.args.gpus_per_trial)(fn)

    def search(self, split, cell, extra_hparams=None):
        """ ASHA search on train{split}/val{split}; returns the chosen setting. """
        out = self.path(split, f'search_{cell}.json')
        if os.path.exists(out):
            return read_json(out)
        from ray import tune
        from gem.ray_classifier import ray_train_classifier

        a = self.args
        pipeline, space, fixed, epochs, grace = search_space(cell, a)
        config = {
            **space,
            'hparams': {**fixed, **(extra_hparams or {}), **space.get('hparams', {})},
            'epochs': epochs,
            'pipeline': pipeline,
            'architecture': ARCH,
            'init_weights': None,
            'dataset': DATASET,
            'data_root': self.data_root,
            'train_split': f'train{split}',
            'test_split': f'val{split}',
            'load_workers': a.load_workers,
        }
        scheduler = tune.schedulers.ASHAScheduler(
            metric=METRIC, mode='max', max_t=epochs, grace_period=grace, reduction_factor=2)
        log(f'split {split} search {cell}: {a.trials} trials, up to {epochs} epochs, grace {grace}')
        t0 = time.time()
        analysis = tune.run(
            ray_train_classifier,
            name=f'split{split}_{cell}',
            config=config,
            num_samples=a.trials,
            scheduler=scheduler,
            resources_per_trial={'cpu': a.cpus_per_trial, 'gpu': a.gpus_per_trial},
            storage_path=os.path.join(self.out, 'ray'),
            raise_on_failed_trial=False,
            verbose=1,
        )
        minutes = (time.time() - t0) / 60
        trials = analysis.trials
        errored = [t for t in trials if t.status == 'ERROR']
        if len(errored) > max(1, len(trials) // 10):
            raise RuntimeError(f'{len(errored)} of {len(trials)} trials failed in split {split} {cell}; '
                               f'see {os.path.join(self.out, "ray")}')
        best = analysis.get_best_trial(METRIC, 'max', 'last')
        trial_epochs = sum((t.last_result or {}).get('training_iteration', 0) for t in trials)
        hparams = {key: value for key, value in best.config['hparams'].items()}
        record = {
            'cell': cell, 'pipeline': pipeline, 'split': split,
            'lr': best.config['lr'], 'decay': best.config['decay'], 'batch_size': best.config['batch_size'],
            'hparams': hparams, 'epochs': epochs, 'grace': grace,
            'val_accuracy_avg5': best.last_result[METRIC], 'val_accuracy': best.last_result['accuracy'],
            'best_trial_epochs': best.last_result['training_iteration'],
            'trials': len(trials), 'errored': len(errored), 'trial_epochs': trial_epochs,
            'minutes': minutes, 'seconds_per_trial_epoch': minutes * 60 / max(1, trial_epochs),
        }
        write_json(out, record)
        log(f'split {split} search {cell} done in {minutes:.1f} min, {trial_epochs} trial-epochs '
            f'({record["seconds_per_trial_epoch"]:.2f} s each): val {record["val_accuracy_avg5"]:.4f}, '
            f'lr {record["lr"]:.3g}, decay {record["decay"]:.3g}, batch {record["batch_size"]}, {hparams}')
        slack(f'ciFAIR GEM: split {split} {cell} search done ({minutes:.0f} min), val {record["val_accuracy_avg5"]:.3f}')
        return record

    def search_encoder(self, split, supcon):
        """ The chosen SupCon setting pretrained on train{split}, for the downstream searches. """
        import ray
        path = self.path(split, 'supcon_train.pt')
        if not os.path.exists(path):
            log(f'split {split}: pretraining the chosen SupCon setting on train{split}')
            task = self.remote(train_once)
            result = ray.get(task.remote('contrastive', supcon, {}, self.data_root, f'train{split}',
                                         self.args.epochs, 0, self.args.load_workers, save=path + '.part'))
            os.replace(path + '.part', path)
            write_json(self.path(split, 'supcon_train.json'), result)
        return path

    def finals(self, split, settings):
        """ Final runs on trainval{split}, evaluated on test0; returns {cell: [result per seed]}. """
        import ray
        a = self.args
        data_root, workers, test = self.data_root, a.load_workers, 'test0'
        trainval = f'trainval{split}'

        def supcon_seed(seed, cells, supcon, downstream, encoder):
            # One SupCon encoder per seed, shared by every downstream cell of that seed.
            out = {}
            if not os.path.exists(encoder):
                pre = train_once('contrastive', supcon, {}, data_root, trainval, a.epochs, seed, workers,
                                 save=encoder + '.part')
                os.replace(encoder + '.part', encoder)
                out['pretrain'] = pre
            for cell in cells:
                pipeline = 'linear' if cell == 'ULfz' else 'memorywrap'
                out[cell] = train_once(pipeline, downstream[cell], {'pretrained_encoder': encoder}, data_root,
                                       trainval, a.down_epochs, seed, workers, test_split=test)
            return out

        def scratch_seed(seed, cell, setting):
            pipeline = 'linear' if cell == 'SL' else 'memorywrap'
            return {cell: train_once(pipeline, setting, {}, data_root, trainval, a.epochs, seed, workers,
                                     test_split=test)}

        jobs = {}
        supcon_cells = [c for c in SUPCON_CELLS if c in settings]
        for seed in range(a.final_runs):
            todo = [c for c in supcon_cells if not os.path.exists(self.path(split, f'final_{c}_seed{seed}.json'))]
            if todo:
                encoder = self.path(split, f'supcon_trainval_seed{seed}.pt')
                jobs[self.remote(supcon_seed).remote(seed, todo, settings['supcon'], settings, encoder)] = seed
            for cell in SCRATCH_CELLS:
                if cell in settings and not os.path.exists(self.path(split, f'final_{cell}_seed{seed}.json')):
                    jobs[self.remote(scratch_seed).remote(seed, cell, settings[cell])] = seed
        if jobs:
            log(f'split {split}: {len(jobs)} final jobs')
        pending = list(jobs)
        while pending:
            done, pending = ray.wait(pending, num_returns=1)
            seed = jobs[done[0]]
            for cell, result in ray.get(done[0]).items():
                if cell == 'pretrain':
                    write_json(self.path(split, f'supcon_trainval_seed{seed}.json'), result)
                    continue
                write_json(self.path(split, f'final_{cell}_seed{seed}.json'), result)
                log(f'split {split} final {cell} seed {seed}: test {result["accuracy"]:.4f}')

        results = {}
        for cell in [c for c in SUPCON_CELLS + SCRATCH_CELLS if c in settings]:
            results[cell] = [read_json(self.path(split, f'final_{cell}_seed{seed}.json'))
                             for seed in range(a.final_runs)]
        return results

    def run(self):
        """ Runs the cells in priority order across all splits, writing results.md after each phase. """
        a = self.args
        settings = {split: {} for split in a.splits}
        results = {split: {} for split in a.splits}
        # Most important first: the frozen SupCon cells, then fine-tuning, then the scratch baselines.
        phases = [[c for c in ('ULfz', 'UMfz') if c in a.cells], ['UMft'] if 'UMft' in a.cells else []]
        phases += [[c] for c in SCRATCH_CELLS if c in a.cells]
        for cells in [p for p in phases if p]:
            for split in a.splits:
                phase = {}
                if cells[0] in SUPCON_CELLS:
                    phase['supcon'] = settings[split]['supcon'] = self.search(split, 'supcon')
                    encoder = self.search_encoder(split, phase['supcon'])
                    extra = {'pretrained_encoder': encoder}
                else:
                    extra = None
                for cell in cells:
                    phase[cell] = settings[split][cell] = self.search(split, cell, extra)
                results[split].update(self.finals(split, phase))
            report(a, settings, results)
            slack(f'ciFAIR GEM: {", ".join(cells)} finished on splits {a.splits}: ' + ', '.join(
                f'{c} {100 * np.mean([r["accuracy"] for s in a.splits for r in results[s][c]]):.2f}' for c in cells))


# --- report ------------------------------------------------------------------

def fmt_setting(s):
    text = f'lr {s["lr"]:.3g}, wd {s["decay"]:.2g}, batch {s["batch_size"]}'
    if s['cell'] == 'supcon':
        text += f', temperature {s["hparams"]["temperature"]:.3g}, head {s["hparams"]["projection_head"]}'
    return text


def report(args, all_settings, all_results):
    """ Writes results.md: per-split settings and the summary table over splits and seeds. """
    lines = ['# ciFAIR-10 GEM results', '',
             f'GEM protocol: {args.trials} ASHA trials per search (reduction factor 2), splits {args.splits}, '
             f'{args.final_runs} final runs per split on trainval (500 images), evaluated on test0 (10,000 images). '
             f'MobileNetV2, SGD with momentum 0.9 and a cosine schedule.', '',
             '## Summary', '',
             '| Cell | Pretraining epochs | Head/fine-tuning epochs | Test accuracy | Balanced accuracy | n |',
             '|---|---|---|---|---|---|']
    for cell in SUPCON_CELLS + SCRATCH_CELLS:
        accs = [r['accuracy'] for split in all_results for r in all_results[split].get(cell, [])]
        baccs = [r['balanced_accuracy'] for split in all_results for r in all_results[split].get(cell, [])]
        if not accs:
            continue
        pre = str(args.epochs) if cell in SUPCON_CELLS else '-'
        down = str(args.down_epochs if cell in SUPCON_CELLS else args.epochs)
        lines.append(f'| {CELL_NAMES[cell]} | {pre} | {down} | {100 * np.mean(accs):.2f} ± {100 * np.std(accs, ddof=1):.2f} '
                     f'| {100 * np.mean(baccs):.2f} | {len(accs)} |')
    lines += ['', 'Mean ± sample std over all final runs of all splits.', '', '## Chosen settings per split', '',
              '| Split | Stage | Settings (chosen on val) | Val accuracy (avg of last 5 epochs) | Search time |',
              '|---|---|---|---|---|']
    for split, settings in all_settings.items():
        for name, s in settings.items():
            label = 'SupCon pretraining' if name == 'supcon' else CELL_NAMES[name]
            lines.append(f'| {split} | {label} | {fmt_setting(s)} | {100 * s["val_accuracy_avg5"]:.2f} '
                         f'| {s["minutes"]:.0f} min |')
    lines += ['', '## Per split test accuracy', '', '| Split | ' + ' | '.join(CELL_NAMES[c] for c in SUPCON_CELLS + SCRATCH_CELLS) + ' |',
              '|---' * 6 + '|']
    for split, results in all_results.items():
        cells = [f'{100 * np.mean([r["accuracy"] for r in results[c]]):.2f}' if c in results else '-'
                 for c in SUPCON_CELLS + SCRATCH_CELLS]
        lines.append(f'| {split} | ' + ' | '.join(cells) + ' |')
    path = os.path.join(os.path.abspath(args.out), 'results.md')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    log(f'wrote {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--splits', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--cells', nargs='+', default=SUPCON_CELLS + SCRATCH_CELLS,
                        choices=SUPCON_CELLS + SCRATCH_CELLS)
    parser.add_argument('--trials', type=int, default=250, help='ASHA trials per search (GEM: 250)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='max epochs for SupCon pretraining and scratch cells (GEM: 500)')
    parser.add_argument('--grace', type=int, default=50, help='ASHA grace period for --epochs (GEM: 50)')
    parser.add_argument('--down-epochs', type=int, default=100,
                        help='max epochs for the heads and fine-tuning on a SupCon encoder')
    parser.add_argument('--down-grace', type=int, default=10, help='ASHA grace period for --down-epochs')
    parser.add_argument('--final-runs', type=int, default=10, help='final runs per split (GEM: 10)')
    parser.add_argument('--gpus-per-trial', type=float, default=0.25)
    parser.add_argument('--cpus-per-trial', type=float, default=4)
    parser.add_argument('--load-workers', type=int, default=3, help='data loading processes per trial')
    parser.add_argument('--ray-cpus', type=int, default=None,
                        help='CPUs Ray may use; set it on containers whose CPU quota is below the host core count')
    parser.add_argument('--ray-temp-dir', default=None,
                        help='separate Ray session directory, needed to run several drivers on one machine')
    parser.add_argument('--data-root', default=os.path.join(GEM_ROOT, 'datasets', 'cifair'))
    parser.add_argument('--out', default=os.path.join(GEM_ROOT, 'results', 'cifair_gem'))
    args = parser.parse_args()

    import ray
    from gem.loader import InstanceLoader

    # Download once here, so parallel trials do not race on it.
    for split in ('train0', 'test0'):
        InstanceLoader('datasets').build_instance(DATASET, os.path.abspath(args.data_root), split)

    # gem/ray_classifier.py imports scripts/train.py as `train`.
    paths = [GEM_ROOT, os.path.join(GEM_ROOT, 'scripts'), os.environ.get('PYTHONPATH', '')]
    # No dashboard: its fixed port would clash between drivers on the same machine.
    ray.init(num_cpus=args.ray_cpus, include_dashboard=False, _temp_dir=args.ray_temp_dir,
             runtime_env={'env_vars': {'PYTHONPATH': os.pathsep.join(p for p in paths if p)}})
    os.makedirs(args.out, exist_ok=True)
    write_json(os.path.join(args.out, 'args.json'), vars(args))

    Runner(args).run()
    slack('ciFAIR GEM: all cells finished')


if __name__ == '__main__':
    main()
