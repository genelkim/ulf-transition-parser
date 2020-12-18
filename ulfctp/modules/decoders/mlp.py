from torch import device
from torch.nn import Module, Sequential, Dropout, Linear, Tanh, ReLU

class MLP(Module):
    """Multi Layer Perceptron.
    """
    def __init__(self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            n_layers: int,
            dropout: float,
        ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout

        layers = []
        cur_input_dim = input_dim
        for i in range(n_layers - 1):
            layers.extend([
                Linear(cur_input_dim, hidden_dim),
                ReLU(),
                Dropout(dropout),
            ])
            cur_input_dim = hidden_dim
        layers.extend([
            Linear(cur_input_dim, output_dim),
            ReLU(),
        ])
        self._mlp = Sequential(*layers)

    def forward(self, embeddings):
        return self._mlp(embeddings)

    # @classmethod
    # def from_params(cls, params):
    #     return cls(
    #         input_dim=params['input_dim'],
    #         hidden_dim=params['hidden_dim'],
    #         n_layers=params.get('n_layers', 2),
    #         dropout=params.get('dropout', 0.3)
    #     )
