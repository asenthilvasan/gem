import torch
import torch.utils.data as datautil
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Callable, Optional, Tuple
from collections import OrderedDict

from gem.pipelines.common import BasicAugmentation, ClassificationMetrics
from gem.pipelines.distill_visual_priors.contrastive_augmentation import TwoCropsTransform
from gem.utils import is_notebook

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class ProjectionHead(nn.Module):

    def __init__(self, in_features: int, hidden_dim: int, out_dim: int, head: str):
        super(ProjectionHead, self).__init__()
        if head == 'identity':
            self.net = nn.Identity()
        elif head == 'linear':
            self.net = nn.Linear(in_features, out_dim)
        elif head == 'mlp':
            hidden_dim = in_features if hidden_dim <= 0 else hidden_dim
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            raise ValueError(f'Unknown projection head: {head}')

    def forward(self, x):
        return self.net(x)


class EncoderProjectionModel(nn.Module):

    def __init__(self, encoder: nn.Module, num_classes: int, projection_dim: int, projection_hidden_dim: int,
                 projection_head: str):
        super(EncoderProjectionModel, self).__init__()
        self.encoder = encoder
        self.feature_dim = self._prepare_encoder()
        self.projection = ProjectionHead(self.feature_dim, projection_hidden_dim, projection_dim, projection_head)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

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

    def forward(self, x) -> Tuple[Tensor, Tensor]:
        embeddings = self.forward_encoder(x)
        logits = self.classifier(embeddings)
        projections = F.normalize(self.projection(embeddings), dim=1)
        return logits, projections


class ContrastiveLoss(nn.Module):

    def __init__(self, loss: str = 'supcon', temperature: float = 0.07, hybrid_alpha: float = 0.5,
                 base_temperature: float = 0.07):
        super(ContrastiveLoss, self).__init__()
        if loss not in ('supcon', 'simclr', 'hybrid'):
            raise ValueError(f'Unknown contrastive loss: {loss}')
        self.loss = loss
        self.temperature = temperature
        self.hybrid_alpha = hybrid_alpha
        self.base_temperature = base_temperature

    def _loss(self, features: Tensor, labels: Optional[Tensor]) -> Tensor:
        two_batch = features.size(0)
        batch = two_batch // 2
        device = features.device
        if labels is None:
            mask = torch.eye(two_batch, device=device).roll(batch, dims=1)
        else:
            labels = torch.cat([labels, labels])
            mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
            mask.fill_diagonal_(0)
        logits = features @ features.T / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        not_self = 1 - torch.eye(two_batch, device=device)
        log_prob = logits - torch.log((logits.exp() * not_self).sum(dim=1, keepdim=True) + 1e-12)
        positives = mask.sum(dim=1).clamp(min=1)
        loss = -(mask * log_prob).sum(dim=1).div(positives)
        loss = loss * (self.temperature / self.base_temperature)
        return loss.mean()

    def forward(self, features: Tensor, labels: Tensor) -> Tensor:
        if self.loss == 'supcon':
            return self._loss(features, labels)
        if self.loss == 'simclr':
            return self._loss(features, None)
        loss_sup = self._loss(features, labels)
        loss_sim = self._loss(features, None)
        return self.hybrid_alpha * loss_sup + (1 - self.hybrid_alpha) * loss_sim


class ContrastiveLossClassifier(BasicAugmentation):

    def create_model(self, arch: str, num_classes: int, input_channels: int, config: dict = {}) -> nn.Module:
        encoder = super(ContrastiveLossClassifier, self).create_model(arch, num_classes, input_channels, config=config)
        return EncoderProjectionModel(
            encoder,
            num_classes,
            projection_dim=self.hparams['projection_dim'],
            projection_hidden_dim=self.hparams['projection_hidden_dim'],
            projection_head=self.hparams['projection_head'],
        )

    def get_loss_function(self) -> Callable:
        return ContrastiveLoss(
            loss=self.hparams['loss'],
            temperature=self.hparams['temperature'],
            hybrid_alpha=self.hparams['hybrid_alpha'],
            base_temperature=self.hparams['base_temperature'],
        )

    def train_epoch(self,
                    model,
                    loader,
                    optimizer,
                    criterion,
                    scheduler=None,
                    regularizer=None,
                    show_progress=True):
        model.train()
        total_loss = total_acc = num_samples = 0
        xent = nn.CrossEntropyLoss()
        for X, y in tqdm(loader, leave=False, disable=not show_progress):
            y = y.cuda()
            X[0] = X[0].cuda()
            X[1] = X[1].cuda()
            imgs = torch.cat([X[0], X[1]], dim=0)
            optimizer.zero_grad(set_to_none=True)
            logits, projections = model(imgs)
            loss_contrastive = criterion(projections, y)
            logits_view = logits[:y.size(0)]
            loss = loss_contrastive
            if self.hparams['xent_weight'] > 0:
                loss = loss + self.hparams['xent_weight'] * xent(logits_view, y)
            if regularizer is not None:
                loss = loss + regularizer(model)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.item() * y.size(0)
            total_acc += (logits_view.argmax(dim=-1) == y).sum().item()
            num_samples += y.size(0)
        return ClassificationMetrics(total_loss / num_samples, total_acc / num_samples)

    def evaluate_epoch(self,
                       model,
                       loader,
                       criterion,
                       show_progress=True):
        model.eval()
        total_loss = total_acc = num_samples = 0
        xent = nn.CrossEntropyLoss()
        with torch.no_grad():
            for X, y in tqdm(loader, leave=False, disable=not show_progress):
                X, y = X.cuda(), y.cuda()
                logits, _ = model(X)
                loss = xent(logits, y)
                total_loss += loss.item() * len(X)
                total_acc += (logits.argmax(dim=-1) == y).sum().item()
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
            train_data.transform = TwoCropsTransform(train_transform)
            if val_data is not None:
                val_data.transform = test_transform
        train_loader = datautil.DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=load_workers, pin_memory=True, drop_last=True,
        )
        val_loader = datautil.DataLoader(
            val_data, batch_size=batch_size, shuffle=False, num_workers=load_workers, pin_memory=True
        ) if val_data is not None else None
        model = self.create_model(architecture, train_data.num_classes, train_data.num_input_channels).cuda()
        if init_weights is not None:
            model = self.load_weights(model, init_weights)
        par_model = nn.DataParallel(model).cuda() if multi_gpu else model
        iterations = len(train_loader) * epochs
        criterion = self.get_loss_function()
        optimizer, scheduler = self.get_optimizer(par_model, max_epochs=epochs, max_iter=iterations)
        regularizer = self.get_regularizer()
        metrics = self.train_model(
            par_model, train_loader, val_loader, optimizer, criterion, epochs,
            train_args={'scheduler': scheduler, 'regularizer': regularizer, 'show_progress': show_sub_progress},
            eval_args={'show_progress': show_sub_progress},
            eval_interval=eval_interval, show_progress=show_progress,
            report_tuner=report_tuner,
        )
        return model, metrics

    @staticmethod
    def get_pipe_name():
        return 'contrastive'

    @staticmethod
    def default_hparams() -> dict:
        return {
            **super(ContrastiveLossClassifier, ContrastiveLossClassifier).default_hparams(),
            'loss': 'supcon',
            'temperature': 0.07,
            'base_temperature': 0.07,
            'hybrid_alpha': 0.5,
            'projection_dim': 128,
            'projection_hidden_dim': 0,
            'projection_head': 'identity',
            'xent_weight': 0.0,
        }
