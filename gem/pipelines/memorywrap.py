import copy
import torch
import torch.utils.data as datautil
import numpy as np
from torch import nn, Tensor
from typing import Callable, Optional, Tuple
from collections import OrderedDict

try:
    from entmax import sparsemax as _entmax_sparsemax
    _HAS_ENTMAX = True
except ImportError:
    _entmax_sparsemax = None
    _HAS_ENTMAX = False

from gem.pipelines.common import BasicAugmentation, ClassificationMetrics
from gem.evaluation import balanced_accuracy_from_predictions
from gem.utils import is_notebook

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


_EPSILON = 1e-6


def _vector_norms(v: Tensor) -> Tensor:
    squared_norms = torch.sum(v * v, dim=1, keepdim=True)
    return torch.sqrt(squared_norms + _EPSILON)


def _distance(x: Tensor, y: Tensor, distance: str = 'cosine') -> Tensor:
    if distance == 'cosine':
        x_norm = x / _vector_norms(x)
        y_norm = y / _vector_norms(y)
        return 1 - torch.mm(x_norm, y_norm.transpose(0, 1))
    if distance == 'l2':
        return (
            x.unsqueeze(1).expand(x.shape[0], y.shape[0], -1) -
            y.unsqueeze(0).expand(x.shape[0], y.shape[0], -1)
        ).pow(2).sum(dim=2)
    if distance == 'dot':
        expanded_x = x.unsqueeze(1).expand(x.shape[0], y.shape[0], -1)
        expanded_y = y.unsqueeze(0).expand(x.shape[0], y.shape[0], -1)
        return -(expanded_x * expanded_y).sum(dim=2)
    raise NameError(f'{distance} not recognized as valid distance.')


def sparsemax(input: Tensor, dim: int = -1) -> Tensor:
    """Sparsemax projection.

    Uses ``entmax.sparsemax`` when available (matches ``memory-wrap-SENN``),
    and falls back to a pure-PyTorch implementation otherwise so this module
    is importable without the optional ``entmax`` dependency.
    """
    if _HAS_ENTMAX:
        return _entmax_sparsemax(input, dim=dim)
    input = input.transpose(dim, -1)
    original_size = input.size()
    input = input.reshape(-1, input.size(-1))
    zs = torch.sort(input, dim=1, descending=True).values
    range = torch.arange(1, input.size(1) + 1, device=input.device, dtype=input.dtype).view(1, -1)
    bound = 1 + range * zs
    cumulative = torch.cumsum(zs, dim=1)
    is_gt = bound > cumulative
    k = is_gt.sum(dim=1, keepdim=True).clamp(min=1)
    tau = (cumulative.gather(1, k - 1) - 1) / k.to(input.dtype)
    output = torch.clamp(input - tau, min=0)
    output = output.reshape(original_size)
    return output.transpose(dim, -1)


def load_pretrained_encoder(model: nn.Module, path: str, skip_prefixes: Tuple[str, ...]):
    """Loads encoder weights into `model.encoder` from a GEM or memory-wrap-SENN checkpoint.

    Parameters starting with one of `skip_prefixes` (the new head, and encoder layers the model never uses)
    are not loaded. Raises if any other parameter is left without pretrained weights.
    """
    loaded = torch.load(path, map_location='cpu')
    state_dict = loaded['model_state_dict'] if isinstance(loaded, dict) and 'model_state_dict' in loaded else loaded
    if not any(key.startswith('encoder.') for key in state_dict.keys()):
        state_dict = {
            f'encoder.{key}': value
            for key, value in state_dict.items()
            if not key.startswith('mw.')
        }
    # Only encoder weights are taken; heads from pretraining (projection, probe) are dropped.
    state_dict = {key: value for key, value in state_dict.items()
                  if key.startswith('encoder.') and not key.startswith(skip_prefixes)}
    missing, _ = model.load_state_dict(state_dict, strict=False)
    missing = [key for key in missing if not key.startswith(skip_prefixes)]
    if missing:
        raise RuntimeError(f'Pretrained encoder is missing {len(missing)} weights, e.g. {missing[:3]}')


def freeze_encoder(model: nn.Module, head_prefix: str):
    """Freezes every parameter outside the head, and BatchNorm statistics via `model.frozen_encoder`."""
    for name, param in model.named_parameters():
        if not name.startswith(head_prefix):
            param.requires_grad_(False)
    model.frozen_encoder = True


class MLP(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class MemoryWrapLayer(nn.Module):

    def __init__(self, encoder_output_dim: int, output_dim: int, classifier: nn.Module = None, distance: str = 'cosine'):
        super(MemoryWrapLayer, self).__init__()
        self.distance_name = distance
        self.classifier = classifier or MLP(encoder_output_dim * 2, encoder_output_dim * 4, output_dim)

    def forward(self, encoder_output: Tensor, memory_set: Tensor, return_weights: bool = False):
        dist = _distance(encoder_output, memory_set, self.distance_name)
        content_weights = sparsemax(-dist, dim=1)
        memory_vector = torch.matmul(content_weights, memory_set)
        final_input = torch.cat([encoder_output, memory_vector], 1)
        output = self.classifier(final_input)
        if return_weights:
            return output, content_weights
        return output


class BaselineMemory(nn.Module):

    def __init__(self, encoder_output_dim: int, output_dim: int, classifier: nn.Module = None, distance: str = 'cosine'):
        super(BaselineMemory, self).__init__()
        self.distance_name = distance
        self.classifier = classifier or MLP(encoder_output_dim, encoder_output_dim * 2, output_dim)

    def forward(self, encoder_output: Tensor, memory_set: Tensor, return_weights: bool = False):
        dist = _distance(encoder_output, memory_set, self.distance_name)
        content_weights = sparsemax(-dist, dim=1)
        memory_vector = torch.matmul(content_weights, memory_set)
        output = self.classifier(memory_vector)
        if return_weights:
            return output, content_weights
        return output


class MemoryWrapModel(nn.Module):

    def __init__(self, encoder: nn.Module, num_classes: int, distance: str, memory_only: bool = False):
        super(MemoryWrapModel, self).__init__()
        self.encoder = encoder
        self.feature_dim = self._prepare_encoder()
        self.mw = BaselineMemory(self.feature_dim, num_classes, distance=distance) if memory_only else MemoryWrapLayer(self.feature_dim, num_classes, distance=distance)
        self.frozen_encoder = False

    def train(self, mode: bool = True):
        super(MemoryWrapModel, self).train(mode)
        # A frozen encoder also keeps its BatchNorm statistics fixed, so the head sees fixed features.
        if self.frozen_encoder:
            self.encoder.eval()
        return self

    def _prepare_encoder(self) -> int:
        if hasattr(self.encoder, 'forward_encoder'):
            if hasattr(self.encoder, 'feature_dim'):
                return self.encoder.feature_dim
            raise ValueError('Encoders with forward_encoder must define feature_dim.')
        if hasattr(self.encoder, 'fc') and isinstance(self.encoder.fc, nn.Linear):
            feature_dim = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
            return feature_dim
        if hasattr(self.encoder, 'linear') and isinstance(self.encoder.linear, nn.Linear):
            feature_dim = self.encoder.linear.in_features
            self.encoder.linear = nn.Identity()
            return feature_dim
        raise ValueError('Unsupported encoder: expected forward_encoder, fc, or linear feature head.')

    def forward_encoder(self, x):
        if hasattr(self.encoder, 'forward_encoder'):
            return self.encoder.forward_encoder(x)
        return self.encoder(x)

    def forward(self, x, memory_input, return_weights: bool = False):
        out = self.forward_encoder(x)
        memory = self.forward_encoder(memory_input)
        return self.mw(out, memory, return_weights=return_weights)


class MemoryWrapClassifier(BasicAugmentation):

    def create_model(self, arch: str, num_classes: int, input_channels: int, config: dict = {}) -> nn.Module:
        encoder = super(MemoryWrapClassifier, self).create_model(arch, num_classes, input_channels, config=config)
        model = MemoryWrapModel(encoder, num_classes, distance=self.hparams['distance'], memory_only=self.hparams['memory_only'])
        if self.hparams['pretrained_encoder']:
            # Memory Wrap never uses the architecture's own classifier (e.g. `encoder.linear`).
            load_pretrained_encoder(model, self.hparams['pretrained_encoder'], ('mw.', 'encoder.linear.', 'encoder.fc.'))
        if self.hparams['freeze_encoder']:
            freeze_encoder(model, 'mw.')
        return model

    def get_loss_function(self) -> Callable:
        return nn.CrossEntropyLoss()

    def train_epoch(self,
                    model,
                    loader,
                    optimizer,
                    criterion,
                    scheduler=None,
                    regularizer=None,
                    show_progress=True,
                    scaler=None):
        train_loader, memory_loader = loader
        model.train()
        total_loss = total_acc = num_samples = 0
        memory_iter = iter(memory_loader)
        use_amp = scaler is not None
        for X, y in tqdm(train_loader, leave=False, disable=not show_progress):
            try:
                memory_input, _ = next(memory_iter)
            except StopIteration:
                memory_iter = iter(memory_loader)
                memory_input, _ = next(memory_iter)
            X, y = X.cuda(), y.cuda()
            memory_input = memory_input.cuda()
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(X, memory_input)
                loss = criterion(output, y)
            if regularizer is not None:
                loss = loss + regularizer(model)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.item() * len(X)
            total_acc += (output.argmax(dim=-1) == y).sum().item()
            num_samples += len(X)
        return ClassificationMetrics(total_loss / num_samples, total_acc / num_samples)

    def evaluate_epoch(self,
                       model,
                       loader,
                       criterion,
                       show_progress=True):
        eval_loader, memory_loader = loader if isinstance(loader, tuple) else (loader, None)
        if memory_loader is None:
            raise ValueError(
                'MemoryWrapClassifier.evaluate_epoch requires a memory loader; '
                'pass loader=(eval_loader, memory_loader).'
            )
        model.eval()
        total_loss = total_acc = num_samples = 0
        memory_iter = iter(memory_loader)
        with torch.no_grad(), torch.cuda.amp.autocast():
            for X, y in tqdm(eval_loader, leave=False, disable=not show_progress):
                try:
                    memory_input, _ = next(memory_iter)
                except StopIteration:
                    memory_iter = iter(memory_loader)
                    memory_input, _ = next(memory_iter)
                X, y = X.cuda(), y.cuda()
                memory_input = memory_input.cuda()
                output = model(X, memory_input)
                loss = criterion(output, y)
                total_loss += loss.item() * len(X)
                total_acc += (output.argmax(dim=-1) == y).sum().item()
                num_samples += len(X)
        return ClassificationMetrics(total_loss / num_samples, total_acc / num_samples)

    def train(self,
              train_data,
              val_data,
              batch_size: int,
              epochs: int,
              architecture: str = 'rn50',
              init_weights: Optional[str] = None,
              show_progress: bool = True,
              show_sub_progress: bool = False,
              eval_interval: int = 1,
              multi_gpu: bool = False,
              load_workers: int = 8,
              keep_transform: bool = False,
              report_tuner: bool = False) -> Tuple[nn.Module, OrderedDict]:
        if not keep_transform:
            train_transform, test_transform = self.get_data_transforms(train_data)
            train_data.transform = train_transform
            if val_data is not None:
                val_data.transform = test_transform
        else:
            test_transform = train_data.transform
        # Memory is drawn from the training images without augmentation, in training as in
        # memory-wrap-SENN's CIFAR-10 loader, and in evaluation.
        memory_eval_data = copy.copy(train_data)
        memory_eval_data.transform = test_transform
        self.memory_data = memory_eval_data
        self.memory_transform = test_transform
        train_loader = datautil.DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=load_workers, persistent_workers=load_workers > 0, pin_memory=True,
        )
        memory_loader = datautil.DataLoader(
            memory_eval_data, batch_size=self.hparams['mem_batch_size'], shuffle=True, num_workers=load_workers, persistent_workers=load_workers > 0, pin_memory=True,
        )
        val_loader = datautil.DataLoader(
            val_data, batch_size=batch_size, shuffle=False, num_workers=load_workers, persistent_workers=load_workers > 0, pin_memory=True,
        ) if val_data is not None else None
        val_memory_loader = datautil.DataLoader(
            memory_eval_data, batch_size=self.hparams['mem_batch_size'], shuffle=True, num_workers=load_workers, persistent_workers=load_workers > 0, pin_memory=True,
        ) if val_data is not None else None
        model = self.create_model(architecture, train_data.num_classes, train_data.num_input_channels).cuda()
        if init_weights is not None:
            model = self.load_weights(model, init_weights)
        par_model = nn.DataParallel(model).cuda() if multi_gpu else model
        iterations = len(train_loader) * epochs
        criterion = self.get_loss_function()
        optimizer, scheduler = self.get_optimizer(par_model, max_epochs=epochs, max_iter=iterations)
        regularizer = self.get_regularizer()
        scaler = torch.cuda.amp.GradScaler()
        metrics = self.train_model(
            par_model, (train_loader, memory_loader), (val_loader, val_memory_loader) if val_loader is not None else None,
            optimizer, criterion, epochs,
            train_args={'scheduler': scheduler, 'regularizer': regularizer, 'show_progress': show_sub_progress, 'scaler': scaler},
            eval_args={'show_progress': show_sub_progress},
            eval_interval=eval_interval, show_progress=show_progress,
            report_tuner=report_tuner,
        )
        return model, metrics

    def evaluate(self, model: nn.Module, test_data, batch_size: int = 10, print_metrics: bool = False):
        test_transform = getattr(self, 'memory_transform', None)
        if test_transform is None:
            _, test_transform = self.get_data_transforms(test_data)
        memory_data = getattr(self, 'memory_data', test_data)
        memory_transform = getattr(self, 'memory_transform', test_transform)
        prev_test_transform = test_data.transform
        prev_memory_transform = memory_data.transform
        test_data.transform = test_transform
        memory_data.transform = memory_transform
        try:
            test_loader = datautil.DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True)
            memory_loader = datautil.DataLoader(memory_data, batch_size=self.hparams['mem_batch_size'], shuffle=True, pin_memory=True)
            criterion = self.get_loss_function()
            model = model.cuda()
            accuracies = []
            balanced_accuracies = []
            for _ in range(self.hparams['eval_memory_runs']):
                acc, acc_b, loss = self._evaluate_once(model, test_loader, memory_loader, criterion)
                accuracies.append(acc)
                balanced_accuracies.append(acc_b)
            acc = float(np.mean(accuracies))
            acc_b = float(np.mean(balanced_accuracies))
            if print_metrics:
                print('Accuracy: {:.2%}'.format(acc))
                print('Balanced accuracy: {:.2%}'.format(acc_b))
            return {'accuracy': acc, 'balanced_accuracy': acc_b}
        finally:
            memory_data.transform = prev_memory_transform
            test_data.transform = prev_test_transform

    def _evaluate_once(self, model: nn.Module, test_loader, memory_loader, criterion):
        gt = []
        pred = []
        total_loss = total_acc = num_samples = 0
        memory_iter = iter(memory_loader)
        model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast():
            for X, y in test_loader:
                try:
                    memory_input, _ = next(memory_iter)
                except StopIteration:
                    memory_iter = iter(memory_loader)
                    memory_input, _ = next(memory_iter)
                output = model(X.cuda(), memory_input.cuda())
                loss = criterion(output, y.cuda())
                total_loss += loss.item() * len(X)
                total_acc += (output.argmax(dim=-1).cpu() == y).sum().item()
                num_samples += len(X)
                gt.append(y.numpy())
                pred.append(output.argmax(dim=-1).cpu().numpy())
        gt = np.concatenate(gt)
        pred = np.concatenate(pred)
        acc = total_acc / num_samples
        acc_b = balanced_accuracy_from_predictions(gt, pred)
        return acc, acc_b, total_loss / num_samples

    @staticmethod
    def get_pipe_name():
        return 'memorywrap'

    @staticmethod
    def default_hparams() -> dict:
        return {
            **super(MemoryWrapClassifier, MemoryWrapClassifier).default_hparams(),
            'distance': 'cosine',
            'mem_batch_size': 100,
            'memory_only': False,
            'eval_memory_runs': 5,
            'pretrained_encoder': None,
            'freeze_encoder': False,
        }
