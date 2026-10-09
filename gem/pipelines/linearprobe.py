from torch import nn

from gem.pipelines.memorywrap import load_pretrained_encoder, freeze_encoder
from gem.pipelines.xent import CrossEntropyClassifier


class LinearHeadModel(nn.Module):
    """ The architecture's own linear classifier on top of a (possibly pretrained) encoder. """

    def __init__(self, encoder: nn.Module):
        super(LinearHeadModel, self).__init__()
        self.encoder = encoder
        self.frozen_encoder = False

    def forward(self, x):
        return self.encoder(x)

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

    See `BasicAugmentation` for the remaining hyper-parameters.
    """

    head_prefix = 'encoder.linear.'

    def create_model(self, arch: str, num_classes: int, input_channels: int, config: dict = {}) -> nn.Module:
        encoder = super(LinearHeadClassifier, self).create_model(arch, num_classes, input_channels, config=config)
        if not (hasattr(encoder, 'linear') and isinstance(encoder.linear, nn.Linear)):
            raise ValueError('The linear pipeline needs an architecture with a `linear` classifier.')
        model = LinearHeadModel(encoder)
        if self.hparams['pretrained_encoder']:
            load_pretrained_encoder(model, self.hparams['pretrained_encoder'], (self.head_prefix,))
        if self.hparams['freeze_encoder']:
            freeze_encoder(model, self.head_prefix)
        return model

    @staticmethod
    def get_pipe_name():
        return 'linear'

    @staticmethod
    def default_hparams() -> dict:
        return {
            **super(LinearHeadClassifier, LinearHeadClassifier).default_hparams(),
            'pretrained_encoder': None,
            'freeze_encoder': False,
        }
