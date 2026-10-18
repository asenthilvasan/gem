import copy
from torch import nn

from gem.pipelines.common import ClassificationMetrics
from gem.pipelines.memorywrap import load_pretrained_encoder, freeze_encoder, make_teacher, distill_loss
from gem.pipelines.xent import CrossEntropyClassifier
from gem.utils import is_notebook

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class LinearHeadModel(nn.Module):
    """ The architecture's own linear classifier on top of a (possibly pretrained) encoder. """

    def __init__(self, encoder: nn.Module, head: str):
        super(LinearHeadModel, self).__init__()
        self.encoder = encoder
        self.head = head
        self.frozen_encoder = False

    def forward(self, x, return_features: bool = False):
        if not return_features:
            return self.encoder(x)
        # The features are the input of the architecture's classifier.
        features = []
        hook = getattr(self.encoder, self.head).register_forward_hook(lambda module, inputs, output: features.append(inputs[0]))
        try:
            output = self.encoder(x)
        finally:
            hook.remove()
        return output, features[0]

    def train(self, mode: bool = True):
        super(LinearHeadModel, self).train(mode)
        # A frozen encoder also keeps its BatchNorm statistics fixed: a true linear probe.
        if self.frozen_encoder:
            self.encoder.eval()
        return self


class LinearHeadClassifier(CrossEntropyClassifier):
    """ Cross-entropy training of a linear head on a pretrained encoder, frozen (linear probe) or fine-tuned.

    Hyper-Parameters
    ----------------
    pretrained_encoder : str
        Path to a checkpoint with encoder weights, e.g. from the `contrastive` pipeline.
    freeze_encoder : bool
        Train only the linear head and keep BatchNorm statistics fixed.
    distill_weight : float
        Weight of the mean cosine distance between the encoder's features and a frozen copy of the pretrained
        encoder's, added to the cross-entropy loss while fine-tuning (DVP-style, as in the memorywrap pipeline).

    See `BasicAugmentation` for the remaining hyper-parameters.
    """

    def create_model(self, arch: str, num_classes: int, input_channels: int, config: dict = {}) -> nn.Module:
        encoder = super(LinearHeadClassifier, self).create_model(arch, num_classes, input_channels, config=config)
        # MobileNetV2 names its classifier `linear`, the WRN `fc`.
        head = next((name for name in ('linear', 'fc') if isinstance(getattr(encoder, name, None), nn.Linear)), None)
        if head is None:
            raise ValueError('The linear pipeline needs an architecture with a `linear` or `fc` classifier.')
        head_prefix = f'encoder.{head}.'
        model = LinearHeadModel(encoder, head)
        if self.hparams['pretrained_encoder']:
            load_pretrained_encoder(model, self.hparams['pretrained_encoder'], (head_prefix,))
        if self.hparams['freeze_encoder']:
            freeze_encoder(model, head_prefix)
        self.teacher = None
        if self.hparams['distill_weight'] > 0:
            # The teacher outputs features: its classifier is replaced by the identity.
            teacher = copy.deepcopy(model.encoder)
            setattr(teacher, head, nn.Identity())
            self.teacher = make_teacher(teacher, self.hparams)
        return model

    def train_epoch(self, model, loader, optimizer, criterion, scheduler=None, regularizer=None, show_progress=True):
        if getattr(self, 'teacher', None) is None:
            return super(LinearHeadClassifier, self).train_epoch(
                model, loader, optimizer, criterion, scheduler=scheduler, regularizer=regularizer, show_progress=show_progress)
        # The base class's loop plus the distillation term (a linear head has no memory images, only queries).
        teacher = self.teacher.cuda()
        model.train()
        total_loss = total_acc = num_samples = 0
        for X, y in tqdm(loader, leave=False, disable=not show_progress):
            X, y = X.cuda(), y.cuda()
            optimizer.zero_grad(set_to_none=True)
            output, features = model(X, return_features=True)
            loss = criterion(output, y) + self.hparams['distill_weight'] * distill_loss(teacher, X, features)
            total_loss += loss.item() * len(X)
            total_acc += (output.argmax(dim=-1) == y).sum().item()
            num_samples += len(X)
            if regularizer is not None:
                loss = loss + regularizer(model)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        return ClassificationMetrics(total_loss / num_samples, total_acc / num_samples)

    @staticmethod
    def get_pipe_name():
        return 'linear'

    @staticmethod
    def default_hparams() -> dict:
        return {
            **super(LinearHeadClassifier, LinearHeadClassifier).default_hparams(),
            'pretrained_encoder': None,
            'freeze_encoder': False,
            'distill_weight': 0.0,
        }
