# ── cluster_cinn.py ──
import torch
import torch.nn as nn
from FrEIA.framework import (
    InputNode, ConditionNode, Node, OutputNode, ReversibleGraphNet
)
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom

class cINN(nn.Module):
    """
    A conditional INN with GLOW‐style affine couplings,
    now conditioned *only* on the image embedding (no scalars).
    """
    def __init__(self,
                 y_dim:int,
                 x_dim:int,
                 hidden_dim:int=128,
                 n_blocks:int=12,
                 clamp:float=2.0):
        super().__init__()
        # 1) Conditioning node
        cond_node = ConditionNode(x_dim, name='cond')
        # 2) Main y‐input node
        nodes = [InputNode(y_dim, name='y_in')]

        # 3) Subnet: small MLP w/ dropout
        def subnet_constructor(ch_in, ch_out):
            return nn.Sequential(
                nn.Linear(ch_in, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(hidden_dim, ch_out)
            )

        # 4) Stack GLOW couplings + random permutations
        for i in range(n_blocks):
            nodes.append(
                Node(
                    nodes[-1],
                    GLOWCouplingBlock,
                    {
                      'subnet_constructor': subnet_constructor,
                      'clamp': clamp
                    },
                    conditions=cond_node,
                    name=f'coupling_{i}'
                )
            )
            nodes.append(
                Node(
                    nodes[-1],
                    PermuteRandom,
                    {'seed': i},
                    name=f'permute_{i}'
                )
            )

        # 5) Output node
        nodes.append(OutputNode(nodes[-1], name='y_out'))
        # 6) Build reversible graph
        self.flow = ReversibleGraphNet(nodes + [cond_node], verbose=False)

    def forward(self, y: torch.Tensor, x: torch.Tensor):
        return self.flow(y, c=x)

    def inverse(self, z: torch.Tensor, x: torch.Tensor):
        return self.flow(z, c=x, rev=True)
