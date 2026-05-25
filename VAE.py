import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphPAS_VAE(nn.Module):
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
        self.latent_dim = latent_dim
        self.output_activation = output_activation

        # Encoder (shared trunk)
        enc_dims = [input_dim] + list(hidden_dims)
        enc_layers = []
        for i in range(len(enc_dims) - 1):
            enc_layers.append(nn.Linear(enc_dims[i], enc_dims[i + 1]))
            enc_layers.append(nn.ReLU())
            if dropout > 0:
                enc_layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*enc_layers)

        # Mean and log-variance heads
        self.fc_mu = nn.Linear(enc_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(enc_dims[-1], latent_dim)

        # Decoder
        dec_dims = [latent_dim] + list(hidden_dims[::-1]) + [input_dim]
        dec_layers = []
        for i in range(len(dec_dims) - 1):
            dec_layers.append(nn.Linear(dec_dims[i], dec_dims[i + 1]))
            if i != len(dec_dims) - 2:
                dec_layers.append(nn.ReLU())
                if dropout > 0:
                    dec_layers.append(nn.Dropout(dropout))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

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
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL divergence: KL(q(z|x) || p(z)) where p(z) = N(0, I)."""
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    @torch.no_grad()
    def get_embedding(self, x: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
        self.eval()
        outs = []
        for start in range(0, x.shape[0], batch_size):
            batch = x[start:start + batch_size]
            mu, _ = self.encode(batch)
            outs.append(mu.cpu())
        return torch.cat(outs, dim=0)
