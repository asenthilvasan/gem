""" Exact checks of the GEM Memory Wrap, linear and SupCon pipelines against memory-wrap-SENN.

Feeds identical weights and inputs through both codebases in fp32 on the GPU and requires identical results:
forward passes, training steps, frozen encoders, encoder loading, the SupCon loss and augmentations, plus
the ciFAIR-10 splits. Expects memory-wrap-SENN (with the memorywrap package) and its SVHN SupCon encoder
under /workspace, as on the RunPod volume.

Usage (on the pod): python scripts/check_parity.py
"""
import copy
import hashlib
import sys

sys.path[:0] = ['/workspace/gem-SENN/gem', '/workspace/gem-SENN/gem/scripts', '/workspace/memory-wrap-SENN/paper']

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from architectures import mobilenet as repo_mobilenet          # memory-wrap-SENN
import pretrain_supcon as repo_supcon                          # memory-wrap-SENN (defines absl flags only)
from gem.loader import InstanceLoader

ROOT = '/workspace/gem-SENN/gem/datasets/cifair'
REPO_ENCODER = '/workspace/memory-wrap-SENN/paper/models/SVHN/supcon/mobilenet/500_lr0.25_t0.2_ep80_final/seed0.pt'
TOL = 1e-5
failures = []


def check(name, ok, detail=''):
    print(f'{"PASS" if ok else "FAIL"}  {name}  {detail}', flush=True)
    if not ok:
        failures.append(name)


def maxdiff(a, b):
    return (a.float() - b.float()).abs().max().item()


def pipeline(name, **hparams):
    return InstanceLoader('pipelines').build_instance(name, lr=0.05, weight_decay=5e-4, **hparams)


def repo_to_gem(state, head):
    """ memory-wrap-SENN state dict -> GEM wrapper keys (encoder under `encoder.`, the head as `head`). """
    out = {}
    for key, value in state.items():
        if head == 'mw' and key.startswith('mw.'):
            out[key] = value
        else:
            out['encoder.' + key] = value
    return out


def params_close(a, b, prefix_map=lambda k: k):
    sa, sb = a.state_dict(), b.state_dict()
    worst = 0.0
    for key, value in sa.items():
        worst = max(worst, maxdiff(value, sb[prefix_map(key)]))
    return worst


def batches(seed=0, n=16, mem=100):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, 32, 32, generator=g).cuda()
    y = torch.randint(0, 10, (n,), generator=g).cuda()
    m = torch.randn(mem, 3, 32, 32, generator=g).cuda()
    return x, y, m


def repo_mw_step(model, opt, x, y, m):
    """ One step of memory-wrap-SENN's train_memory_model, without autocast (fp32 for an exact comparison). """
    opt.zero_grad()
    loss = torch.nn.CrossEntropyLoss()(model(x, m), y)
    loss.backward()
    opt.step()
    return loss.item()


# --- 1. Memory Wrap model: forward, train step, frozen encoder ------------------------------------------------

torch.manual_seed(0)
repo = repo_mobilenet.EncoderMemoryMobileNetV2(10).cuda()
mw_pipe = pipeline('memorywrap')
gem = mw_pipe.create_model('mobilenet', 10, 3).cuda()
missing, unexpected = gem.load_state_dict(repo_to_gem(repo.state_dict(), 'mw'), strict=False)
check('MW: every repo weight maps onto GEM', not unexpected and all(k.startswith('encoder.linear.') for k in missing),
      f'missing={missing} unexpected={unexpected}')
gem_key = lambda k: k if k.startswith('mw.') else 'encoder.' + k

x, y, m = batches()
repo.eval(); gem.eval()
with torch.no_grad():
    out_r, w_r = repo(x, m, return_weights=True)
    out_g, w_g = gem(x, m, return_weights=True)
check('MW eval forward: logits equal', maxdiff(out_r, out_g) < TOL, f'{maxdiff(out_r, out_g):.2e}')
check('MW eval forward: sparsemax weights equal', maxdiff(w_r, w_g) < TOL, f'{maxdiff(w_r, w_g):.2e}')

repo.train(); gem.train()
opt_r = torch.optim.SGD(repo.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
opt_g, _ = mw_pipe.get_optimizer(gem, max_epochs=1, max_iter=10 ** 9)
for step in range(3):
    x, y, m = batches(step + 1)
    loss_r = repo_mw_step(repo, opt_r, x, y, m)
    gem_metrics = mw_pipe.train_epoch(gem, ([(x, y)], [(m, y)]), opt_g, mw_pipe.get_loss_function(), show_progress=False)
    check(f'MW train step {step}: loss equal', abs(loss_r - gem_metrics.loss) < TOL, f'{loss_r:.6f} vs {gem_metrics.loss:.6f}')
diff = params_close(repo, gem, gem_key)
check('MW after 3 SGD steps: all weights and BN statistics equal', diff < 1e-4, f'{diff:.2e}')

# --- 2. Linear head model -------------------------------------------------------------------------------------

torch.manual_seed(1)
repo_lin = repo_mobilenet.MobileNetV2(10).cuda()
lin_pipe = pipeline('linear')
gem_lin = lin_pipe.create_model('mobilenet', 10, 3).cuda()
missing, unexpected = gem_lin.load_state_dict(repo_to_gem(repo_lin.state_dict(), 'linear'))
check('Linear: weights map one to one', not missing and not unexpected)
lin_key = lambda k: 'encoder.' + k
x, y, _ = batches(5)
repo_lin.eval(); gem_lin.eval()
with torch.no_grad():
    d = maxdiff(repo_lin(x), gem_lin(x))
check('Linear eval forward equal', d < TOL, f'{d:.2e}')
repo_lin.train(); gem_lin.train()
opt_r = torch.optim.SGD(repo_lin.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
opt_g, _ = lin_pipe.get_optimizer(gem_lin, max_epochs=1, max_iter=10 ** 9)
for step in range(3):
    x, y, _ = batches(10 + step)
    opt_r.zero_grad(); torch.nn.CrossEntropyLoss()(repo_lin(x), y).backward(); opt_r.step()
    lin_pipe.train_epoch(gem_lin, [(x, y)], opt_g, lin_pipe.get_loss_function(), show_progress=False)
diff = params_close(repo_lin, gem_lin, lin_key)
check('Linear after 3 SGD steps: all weights and BN statistics equal', diff < 1e-4, f'{diff:.2e}')

# --- 3. Pretrained encoders: repo checkpoint into GEM, GEM SupCon checkpoint into GEM heads ---------------------

ckpt = torch.load(REPO_ENCODER, map_location='cpu')
repo_enc = repo_mobilenet.EncoderMemoryMobileNetV2(10)
repo_enc.load_state_dict({k: v for k, v in ckpt['model_state_dict'].items() if not k.startswith('mw.')}, strict=False)
repo_enc = repo_enc.cuda().eval()
x, _, _ = batches(20)
for name, pipe_name in (('MW', 'memorywrap'), ('Linear', 'linear')):
    model = pipeline(pipe_name, pretrained_encoder=REPO_ENCODER).create_model('mobilenet', 10, 3).cuda().eval()
    with torch.no_grad():
        d = maxdiff(repo_enc.forward_encoder(x), model.encoder.forward_encoder(x))
    check(f'{name}: repo SupCon checkpoint loads with identical encoder features', d < TOL, f'{d:.2e}')

con_pipe = pipeline('contrastive', base_temperature=None)
con = con_pipe.create_model('mobilenet', 10, 3).cuda()
con.train()
with torch.no_grad():  # move BN statistics away from their initial values
    con(torch.randn(64, 3, 32, 32).cuda())
con.eval()
torch.save(con.state_dict(), '/tmp/parity_supcon.pt')
for name, pipe_name in (('MW', 'memorywrap'), ('Linear', 'linear')):
    model = pipeline(pipe_name, pretrained_encoder='/tmp/parity_supcon.pt').create_model('mobilenet', 10, 3).cuda().eval()
    with torch.no_grad():
        d = maxdiff(con.forward_encoder(x), model.encoder.forward_encoder(x))
    check(f'{name}: GEM SupCon checkpoint loads with identical encoder features', d < TOL, f'{d:.2e}')

# Frozen encoders: weights and BN statistics stay fixed; the head trains.
for name, pipe_name in (('MW', 'memorywrap'), ('Linear', 'linear')):
    pipe = pipeline(pipe_name, pretrained_encoder=REPO_ENCODER, freeze_encoder=True)
    model = pipe.create_model('mobilenet', 10, 3).cuda()
    before = copy.deepcopy(model.state_dict())
    opt, _ = pipe.get_optimizer(model, max_epochs=1, max_iter=10 ** 9)
    x, y, m = batches(30)
    loader = ([(x, y)], [(m, y)]) if pipe_name == 'memorywrap' else [(x, y)]
    for _ in range(2):
        pipe.train_epoch(model, loader, opt, pipe.get_loss_function(), show_progress=False)
    after = model.state_dict()
    head = 'mw.' if pipe_name == 'memorywrap' else 'encoder.linear.'
    enc_change = max(maxdiff(before[k], after[k]) for k in before if not k.startswith(head))
    head_change = max(maxdiff(before[k], after[k]) for k in before if k.startswith(head))
    check(f'{name} frozen: encoder weights and BN statistics unchanged', enc_change == 0, f'{enc_change:.2e}')
    check(f'{name} frozen: head trains', head_change > 0, f'{head_change:.2e}')

# Frozen MW step equals the repo's frozen step (set_train_mode: encoder BN in eval, head trains).
torch.manual_seed(2)
repo = repo_mobilenet.EncoderMemoryMobileNetV2(10)
repo.load_state_dict({k: v for k, v in ckpt['model_state_dict'].items() if not k.startswith('mw.')}, strict=False)
repo = repo.cuda()
pipe = pipeline('memorywrap', pretrained_encoder=REPO_ENCODER, freeze_encoder=True)
gem = pipe.create_model('mobilenet', 10, 3).cuda()
gem.load_state_dict({k: v for k, v in repo_to_gem(repo.state_dict(), 'mw').items() if k.startswith('mw.')}, strict=False)
for n, p in repo.named_parameters():
    p.requires_grad_(n.startswith('mw.'))
repo.train()
for n, mod in repo.named_modules():
    if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm) and not n.startswith('mw.'):
        mod.eval()
opt_r = torch.optim.SGD([p for p in repo.parameters() if p.requires_grad], lr=0.05, momentum=0.9, weight_decay=5e-4)
opt_g, _ = pipe.get_optimizer(gem, max_epochs=1, max_iter=10 ** 9)
for step in range(3):
    x, y, m = batches(40 + step)
    repo_mw_step(repo, opt_r, x, y, m)
    pipe.train_epoch(gem, ([(x, y)], [(m, y)]), opt_g, pipe.get_loss_function(), show_progress=False)
diff = params_close(repo, gem, gem_key)
check('MW frozen after 3 steps equals repo frozen training', diff < 1e-4, f'{diff:.2e}')

# --- 4. SupCon loss, probe isolation and one pretraining step ------------------------------------------------

g = torch.Generator().manual_seed(3)
feat = F.normalize(torch.randn(128, 64, generator=g), dim=1).cuda().requires_grad_(True)
labels = torch.randint(0, 10, (64,), generator=g).cuda()
for temp in (0.07, 0.2, 0.5):
    loss_r = repo_supcon.contrastive_loss(feat, labels, temp)
    grad_r, = torch.autograd.grad(loss_r, feat)
    crit = InstanceLoader('pipelines').build_instance('contrastive', temperature=temp, base_temperature=None).get_loss_function()
    loss_g = crit(feat, labels)
    grad_g, = torch.autograd.grad(loss_g, feat)
    check(f'SupCon loss equal (temperature {temp})', abs(loss_r.item() - loss_g.item()) < TOL and maxdiff(grad_r, grad_g) < TOL,
          f'{loss_r.item():.6f} vs {loss_g.item():.6f}')

# One pretraining step: the encoder update must equal a pure SupCon step (the probe must not touch the encoder).
torch.manual_seed(4)
pipe = InstanceLoader('pipelines').build_instance('contrastive', lr=0.25, weight_decay=1e-4, temperature=0.2,
                                                   base_temperature=None, projection_head='identity')
gem = pipe.create_model('mobilenet', 10, 3).cuda().train()
repo = copy.deepcopy(gem.encoder)
opt_g, _ = pipe.get_optimizer(gem, max_epochs=1, max_iter=10 ** 9)
opt_r = torch.optim.SGD(repo.parameters(), lr=0.25, momentum=0.9, weight_decay=1e-4)
for step in range(3):
    gen = torch.Generator().manual_seed(50 + step)
    v1, v2 = torch.randn(32, 3, 32, 32, generator=gen), torch.randn(32, 3, 32, 32, generator=gen)
    y = torch.randint(0, 10, (32,), generator=gen)
    opt_r.zero_grad()
    imgs = torch.cat([v1, v2]).cuda()
    repo_supcon.contrastive_loss(F.normalize(repo.forward_encoder(imgs), dim=1), y.cuda(), 0.2).backward()
    opt_r.step()
    pipe.train_epoch(gem, [([v1, v2], y)], opt_g, pipe.get_loss_function(), show_progress=False)
d = max(maxdiff(v, gem.encoder.state_dict()[k]) for k, v in repo.state_dict().items() if not k.startswith('linear.'))
check('SupCon step: encoder update equals pure SupCon (probe isolated)', d < 1e-4, f'{d:.2e}')
probe_change = maxdiff(gem.classifier.weight, pipe.create_model('mobilenet', 10, 3).classifier.weight.cuda())
check('SupCon step: probe trains', probe_change > 0)

# --- 5. Augmentations ------------------------------------------------------------------------------------------

ds = InstanceLoader('datasets').build_instance('cifair10', ROOT, 'train0')
train_t, test_t = InstanceLoader('pipelines').build_instance('contrastive').get_data_transforms(ds)
repo_t = repo_supcon.contrastive_augmentation([0, 0, 0], [1, 1, 1], hflip=True)
strip = lambda t: [repr(s) for s in t.transforms if not isinstance(s, torchvision.transforms.Normalize)]
check('SupCon views: same augmentations as the repo', strip(train_t) == strip(repo_t),
      f'\n   gem:  {strip(train_t)}\n   repo: {strip(repo_t)}')
print('   test transform:', test_t)
mw_train_t, _ = pipeline('memorywrap', rand_shift=4).get_data_transforms(ds)
print('   MW/linear train transform:', mw_train_t)

# --- 6. ciFAIR-10 splits ------------------------------------------------------------------------------------------

def image_hashes(split):
    d = InstanceLoader('datasets').build_instance('cifair10', ROOT, split)
    return [hashlib.md5(img.tobytes()).hexdigest() for img in d.data], np.bincount(np.asarray(d.targets), minlength=10)

seen = set()
for i in range(3):
    tr, tr_counts = image_hashes(f'train{i}')
    va, va_counts = image_hashes(f'val{i}')
    tv, _ = image_hashes(f'trainval{i}')
    check(f'split {i}: 30 train and 20 val images per class',
          all(tr_counts == 30) and all(va_counts == 20), f'{tr_counts} {va_counts}')
    check(f'split {i}: train and val disjoint, trainval is their union',
          not set(tr) & set(va) and set(tv) == set(tr) | set(va) and len(tv) == 500)
    check(f'split {i}: no images shared with other splits', not set(tv) & seen)
    seen |= set(tv)
test, test_counts = image_hashes('test0')
check('test0: 10,000 images, 1,000 per class, none in any trainval', len(test) == 10000 and all(test_counts == 1000)
      and not set(test) & seen)

# --- 7. Memory Wrap memory sets are not augmented --------------------------------------------------------------

pipe = pipeline('memorywrap', rand_shift=4)
train_ds = InstanceLoader('datasets').build_instance('cifair10', ROOT, 'train0')
pipe.train(train_ds, None, batch_size=50, epochs=1, architecture='mobilenet', show_progress=False, load_workers=0)
a, b = pipe.memory_data[0][0], pipe.memory_data[0][0]
check('MW memory images are deterministic (no augmentation)', torch.equal(a, b))

print()
print('ALL PASSED' if not failures else f'{len(failures)} FAILED: {failures}')
