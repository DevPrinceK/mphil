flowchart LR
    A[Input<br/>[batch, seq_len, 2]] --> B[Encoder LSTM<br/>(input_dim=2, hidden_dim=32, num_layers=1)]
    B --> C1[fc_mu<br/>(hidden_dim→latent_dim)]
    B --> C2[fc_logvar<br/>(hidden_dim→latent_dim)]
    C1 --> D[Reparameterization<br/>z = mu + std*eps]
    C2 --> D
    D --> E[fc_decode<br/>(latent_dim→hidden_dim)]
    E --> F[Decoder LSTM<br/>(input_dim=1, hidden_dim=32, num_layers=1)]
    F --> G[fc_out<br/>(hidden_dim→1)]
    G --> H[Output<br/>[batch, seq_len, 1]]