import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphPAS_AutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims=(512,),
        latent_dim: int = 256,
        dropout: float = 0.0,
        output_activation: str | None = None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_activation = output_activation

        enc_dims = [input_dim] + list(hidden_dims) + [latent_dim]
        enc_layers = []
        for i in range(len(enc_dims) - 1):
            enc_layers.append(nn.Linear(enc_dims[i], enc_dims[i + 1]))
            if i != len(enc_dims) - 2:
                enc_layers.append(nn.ReLU())
                if dropout > 0:
                    enc_layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*enc_layers)

        dec_dims = [latent_dim] + list(hidden_dims[::-1]) + [input_dim]
        dec_layers = []
        for i in range(len(dec_dims) - 1):
            dec_layers.append(nn.Linear(dec_dims[i], dec_dims[i + 1]))
            if i != len(dec_dims) - 2:
                dec_layers.append(nn.ReLU())
                if dropout > 0:
                    dec_layers.append(nn.Dropout(dropout))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x_hat = self.decoder(z)

        if self.output_activation == "relu":
            x_hat = F.relu(x_hat)
        elif self.output_activation == "sigmoid":
            x_hat = torch.sigmoid(x_hat)
        elif self.output_activation is None:
            pass
        else:
            raise ValueError(f"Unsupported output_activation: {self.output_activation}")

        return x_hat

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    @torch.no_grad()
    def get_embedding(self, x: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
        self.eval()
        outs = []
        for start in range(0, x.shape[0], batch_size):
            batch = x[start:start + batch_size]
            z = self.encode(batch)
            outs.append(z.cpu())
        return torch.cat(outs, dim=0)