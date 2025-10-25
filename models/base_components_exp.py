from typing import Optional, List
from dataclasses import dataclass

from torch import nn
from torch.nn import functional as F
import torch

from .attention import MaskedMultiHeadAttention


