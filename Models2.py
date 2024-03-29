import torch
import torch.nn as nn
import numpy as np
import math


class TimeSiren(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(TimeSiren, self).__init__()
        # just a fully connected NN with sin activations
        self.lin1 = nn.Linear(input_dim, emb_dim, bias=False)
        self.lin2 = nn.Linear(emb_dim, emb_dim)
        self.output_dim = emb_dim
    def forward(self, x):
        x = torch.sin(self.lin1(x))
        x = self.lin2(x)
        return x


class SpeakingTurnDescriptorEmbedder(nn.Module):
    def __init__(self, num_event_types, embedding_dim, output_dim):
        super(SpeakingTurnDescriptorEmbedder, self).__init__()
        # Embedding layer for each category in the descriptor
        self.embedding = nn.Embedding(num_event_types, embedding_dim)
        self.output_dim = output_dim

        # Linear layer to process concatenated embeddings
        self.fc = nn.Linear(embedding_dim * 2, self.output_dim)  # *2 because we concatenate two embeddings

    def forward(self, x):
        # Assuming x is of shape [batch_size, 2], where each row is a descriptor (int1, int2)
        elem1 = x[:, 0].long()
        elem2 = x[:, 1].long()
        embed_1 = self.embedding(elem1)
        embed_2 = self.embedding(elem2)

        # Concatenate the embeddings
        concatenated = torch.cat((embed_1, embed_2), dim=1)

        # Process the concatenated vector through a linear layer
        output = self.fc(concatenated)

        return output

class ObservationEmbedder(nn.Module):
    def __init__(self, num_facial_types, facial_embedding_dim, cnn_output_dim, lstm_hidden_dim, sequence_length):
        super(ObservationEmbedder, self).__init__()
        self.embedding = nn.Embedding(num_facial_types, facial_embedding_dim)

        # Define CNN layers
        self.conv1 = nn.Conv1d(in_channels=facial_embedding_dim, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=cnn_output_dim, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.output_dim = lstm_hidden_dim

        # Calculate the dimensionality after CNN and pooling layers
        cnn_flattened_dim = cnn_output_dim * (sequence_length // 2 // 2)

        # Define LSTM layer
        self.lstm = nn.LSTM(input_size=cnn_flattened_dim, hidden_size=lstm_hidden_dim, batch_first=True)

        # Optional: Additional layer(s) can be added here to further process LSTM output if needed

    def forward(self, x):

        x = x.long()
        # Embedding layer
        x = self.embedding(x)  # Expecting x to be [batch_size, sequence_length]
        x = x.permute(0, 2, 1)  # Change to [batch_size, embedding_dim, sequence_length] for CNN

        # CNN layers
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)

        # Flatten output for LSTM
        x = x.view(x.size(0), 1, -1)  # Flatten CNN output

        # LSTM layer
        lstm_out, (hidden, cell) = self.lstm(x)

        # Here you can use hidden or lstm_out depending on your need
        # For simplicity, we'll consider 'hidden' as the final feature representation

        return hidden[-1]  # Returning the last hidden state of LSTM


def ddpm_schedules(beta1, beta2, T, is_linear=True):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    # beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    # beta_t = (beta2 - beta1) * torch.arange(-1, T + 1, dtype=torch.float32) / T + beta1
    if is_linear:
        beta_t = (beta2 - beta1) * torch.arange(-1, T, dtype=torch.float32) / (T - 1) + beta1
    else:
        beta_t = (beta2 - beta1) * torch.square(torch.arange(-1, T, dtype=torch.float32)) / torch.max(
            torch.square(torch.arange(-1, T, dtype=torch.float32))) + beta1
    beta_t[0] = beta1  # modifying this so that beta_t[1] = beta1, and beta_t[n_T]=beta2, while beta[0] is never used
    # this is as described in Denoising Diffusion Probabilistic Models paper, section 4
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }

class Model_Cond_Diffusion(nn.Module):
    def __init__(self, nn_model, observation_embedder, mi_embedder, betas, n_T, device, x_dim, y_dim, drop_prob=0.1, guide_w=0.0):
        super(Model_Cond_Diffusion, self).__init__()
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.nn_model = nn_model
        self.n_T = n_T
        self.device = device
        self.observation_embedder = observation_embedder
        self.mi_embedder = mi_embedder
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.guide_w = guide_w

    def loss_on_batch(self, x_batch, y_batch, z_batch):

        _ts = torch.randint(1, self.n_T + 1, (y_batch.shape[0], 1)).to(self.device)

        # dropout context with some probability
       #context_mask = torch.bernoulli(torch.zeros(x_batch.shape[0]) + self.drop_prob).to(self.device)

        # randomly sample some noise, noise ~ N(0, 1)
        noise = torch.randn_like(y_batch).to(self.device)
        self.y_dim = noise.shape

        # add noise to clean target actions
        y_t = self.sqrtab[_ts] * y_batch + self.sqrtmab[_ts] * noise
        # use nn model to predict noise
        noise_pred_batch = self.nn_model(y_t, x_batch, z_batch, _ts / self.n_T) #ici possible d'ajouter context_mask en input

        # return mse between predicted and true noise
        return self.loss_mse(noise, noise_pred_batch)

    def sample(self, x_batch, z_batch, return_y_trace=False):
        # also use this as a shortcut to avoid doubling batch when guide_w is zero
        is_zero = False
        if self.guide_w > -1e-3 and self.guide_w < 1e-3:
            is_zero = True

        # how many noisy actions to begin with
        n_sample = x_batch.shape[0]

        y_shape = (n_sample, x_batch.shape[1])


        # sample initial noise, y_0 ~ N(0, 1),
        y_i = torch.randn(y_shape).to(self.device)


        if not is_zero:
            if len(x_batch.shape) > 2:
                # repeat x_batch twice, so can use guided diffusion
                x_batch = x_batch.repeat(2, 1, 1, 1)
                z_batch = z_batch.repeat(2, 1, 1, 1)
            else:
                # repeat x_batch twice, so can use guided diffusion
                x_batch = x_batch.repeat(2, 1)
                z_batch = z_batch.repeat(2, 1)

            # half of context will be zero
            context_mask = torch.zeros(x_batch.shape[0]).to(self.device)
            context_mask[n_sample:] = 1.0  # makes second half of batch context free
        else:
            context_mask = torch.zeros(x_batch.shape[0]).to(self.device)


        #     #x_embed = self.nn_model.embed_context(x_batch)
        #     x = self.event_embedder(x_batch)
        #     x_embed = self.x_sequence_transformer(x)

        # run denoising chain
        y_i_store = []  # if want to trace how y_i evolved
        for i in range(self.n_T, 0, -1):
            t_is = torch.tensor([i / self.n_T]).to(self.device)
            t_is = t_is.repeat(n_sample, 1)

            if not is_zero:
                # double batch
                y_i = y_i.repeat(2, 1)
                t_is = t_is.repeat(2, 1)

            z = torch.randn(y_shape).to(self.device) if i > 1 else 0


            # if extract_embedding:
            #     eps = self.nn_model(y_i, x_batch, t_is) #ici possible d'input le context_mask

            eps = self.nn_model(y_i, x_batch, z_batch, t_is)
            if not is_zero:
                eps1 = eps[:n_sample]
                eps2 = eps[n_sample:]
                eps = (1 + self.guide_w) * eps1 - self.guide_w * eps2
                y_i = y_i[:n_sample]
            y_i = self.oneover_sqrta[i] * (y_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z
            if return_y_trace and (i % 20 == 0 or i == self.n_T or i < 8):
                y_i_store.append(y_i.detach().cpu().numpy())

        if return_y_trace:
            return y_i, y_i_store
        else:
            return y_i  # Ici les y_i representes la suite de y denoisé step par step pr un sample donnée (du bruit z  au y definitif)

    def sample_update(self, x_batch, betas, n_T, return_y_trace=False):
        original_nT = self.n_T

        # set new schedule
        self.n_T = n_T
        for k, v in ddpm_schedules(betas[0], betas[1], self.n_T).items():
            self.register_buffer(k, v.to(self.device))

        # also use this as a shortcut to avoid doubling batch when guide_w is zero
        is_zero = False
        if self.guide_w > -1e-3 and self.guide_w < 1e-3:
            is_zero = True

        # how many noisy actions to begin with
        n_sample = x_batch.shape[0]

        y_shape = (n_sample, self.y_dim)

        # sample initial noise, y_0 ~ N(0, 1),
        y_i = torch.randn(y_shape).to(self.device)

        if not is_zero:
            if len(x_batch.shape) > 2:
                # repeat x_batch twice, so can use guided diffusion
                x_batch = x_batch.repeat(2, 1, 1, 1)
            else:
                # repeat x_batch twice, so can use guided diffusion
                x_batch = x_batch.repeat(2, 1)
            # half of context will be zero
            context_mask = torch.zeros(x_batch.shape[0]).to(self.device)
            context_mask[n_sample:] = 1.0  # makes second half of batch context free
        else:
            context_mask = torch.zeros(x_batch.shape[0]).to(self.device)

        # run denoising chain
        y_i_store = []  # if want to trace how y_i evolved
        for i in range(self.n_T, 0, -1):
            t_is = torch.tensor([i / self.n_T]).to(self.device)
            t_is = t_is.repeat(n_sample, 1)

            if not is_zero:
                # double batch
                y_i = y_i.repeat(2, 1)
                t_is = t_is.repeat(2, 1)

            # I'm a bit confused why we are adding noise during denoising?
            z = torch.randn(y_shape).to(self.device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(y_i, x_batch, t_is, context_mask)
            if not is_zero:
                eps1 = eps[:n_sample]
                eps2 = eps[n_sample:]
                eps = (1 + self.guide_w) * eps1 - self.guide_w * eps2
                y_i = y_i[:n_sample]
            y_i = self.oneover_sqrta[i] * (y_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z
            if return_y_trace and (i % 20 == 0 or i == self.n_T or i < 8):
                y_i_store.append(y_i.detach().cpu().numpy())

        # reset original schedule
        self.n_T = original_nT
        for k, v in ddpm_schedules(betas[0], betas[1], self.n_T).items():
            self.register_buffer(k, v.to(self.device))

        if return_y_trace:
            return y_i, y_i_store
        else:
            return y_i

    def sample_extra(self, x_batch, extra_steps=4, return_y_trace=False):
        # also use this as a shortcut to avoid doubling batch when guide_w is zero
        is_zero = False
        if self.guide_w > -1e-3 and self.guide_w < 1e-3:
            is_zero = True

        # how many noisy actions to begin with
        n_sample = x_batch.shape[0]

        y_shape = (n_sample, self.y_dim)

        # sample initial noise, y_0 ~ N(0, 1),
        y_i = torch.randn(y_shape).to(self.device)

        if not is_zero:
            if len(x_batch.shape) > 2:
                # repeat x_batch twice, so can use guided diffusion
                x_batch = x_batch.repeat(2, 1, 1, 1)
            else:
                # repeat x_batch twice, so can use guided diffusion
                x_batch = x_batch.repeat(2, 1)
            # half of context will be zero
            context_mask = torch.zeros(x_batch.shape[0]).to(self.device)
            context_mask[n_sample:] = 1.0  # makes second half of batch context free
        else:
            # context_mask = torch.zeros_like(x_batch[:,0]).to(self.device)
            context_mask = torch.zeros(x_batch.shape[0]).to(self.device)

        # run denoising chain
        y_i_store = []  # if want to trace how y_i evolved
        # for i_dummy in range(self.n_T, 0, -1):
        for i_dummy in range(self.n_T, -extra_steps, -1):
            i = max(i_dummy, 1)
            t_is = torch.tensor([i / self.n_T]).to(self.device)
            t_is = t_is.repeat(n_sample, 1)

            if not is_zero:
                # double batch
                y_i = y_i.repeat(2, 1)
                t_is = t_is.repeat(2, 1)

            z = torch.randn(y_shape).to(self.device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(y_i, x_batch, t_is, context_mask)
            if not is_zero:
                eps1 = eps[:n_sample]
                eps2 = eps[n_sample:]
                eps = (1 + self.guide_w) * eps1 - self.guide_w * eps2
                y_i = y_i[:n_sample]
            y_i = self.oneover_sqrta[i] * (y_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z
            if return_y_trace and (i % 20 == 0 or i == self.n_T or i < 8):
                y_i_store.append(y_i.detach().cpu().numpy())

        if return_y_trace:
            return y_i, y_i_store
        else:
            return y_i



class Merger(nn.Module):
    def __init__(self, observation_dim, speaking_turn_dim, output_dim):
        super(Merger, self).__init__()
        self.input_dim = observation_dim + speaking_turn_dim
        intermediate_dim = (self.input_dim + output_dim) // 2  # Example intermediate size

        self.fc1 = nn.Linear(self.input_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, output_dim)
        self.relu = nn.ReLU()
        self.residual_connection = nn.Linear(self.input_dim, output_dim)

        self.norm1 = nn.LayerNorm(intermediate_dim)
        self.norm2 = nn.LayerNorm(output_dim)

    def forward(self, observation_embedding, speaking_turn_embedding):
        combined = torch.cat((observation_embedding, speaking_turn_embedding), dim=1)

        # First layer
        x = self.fc1(combined)
        x = self.relu(x)
        x = self.norm1(x)

        # Second layer with residual connection
        x = self.fc2(x)
        residual = self.residual_connection(combined)
        x += residual  # Add residual
        x = self.relu(x)
        x = self.norm2(x)

        return x

class AdvancedTransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, sequence_length, num_encoder_layers=5, num_decoder_layers=5, num_heads=8, norm_layer=nn.LayerNorm):
        super(AdvancedTransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, hidden_dim))
        self.encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
            for _ in range(num_encoder_layers)])
        self.norm = norm_layer(hidden_dim)
        self.decoder = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads)
            for _ in range(num_decoder_layers)])
        self.decoder_norm = norm_layer(hidden_dim)

        self.final_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(True),
            nn.Linear(hidden_dim // 2, sequence_length)
        )

    def forward(self, concatenated_input):
        # Apply input projection and add positional encoding
        x = self.input_projection(concatenated_input) + self.pos_embedding
        # Pass through encoder layers
        for layer in self.encoder:
            x = layer(x)
        x = self.norm(x)

        # Assuming some form of encoded memory is available, would need modification for real use
        memory = x
        for layer in self.decoder:
            x = layer(x, memory)
        x = self.norm(x)

        # Apply final projection to output
        output = self.final_projection(x)
        return output
class Model_mlp_diff(nn.Module):
    def __init__(self, observation_embedder, mi_embedder, sequence_length, concat, concat2, net_type="transformer"):
        super(Model_mlp_diff, self).__init__()
        self.observation_embedder = observation_embedder
        self.mi_embedder = mi_embedder
        self.time_siren = TimeSiren(1, mi_embedder.output_dim)
        self.net_type = net_type
        self.concat = concat
        self.concat2 = concat2
        # Transformer specific initialization
        self.nheads = 8  # Number of heads in multihead attention
        self.trans_emb_dim = 256# Transformer embedding dimension
        self.projection = 128
        self.input_dim = 256 #Merger output
        # Initialize SequenceTransformers for y and x
        self.merger = Merger(observation_embedder.output_dim, mi_embedder.output_dim, self.input_dim)
        self.t_to_input = nn.Linear(mi_embedder.output_dim, self.projection)
        self.y_to_input = nn.Linear(sequence_length, self.projection)
        if self.concat == True :
            self.transformer = AdvancedTransformerModel(self.projection * 3, hidden_dim=self.trans_emb_dim,
                                                        sequence_length=sequence_length, num_encoder_layers=4,
                                                        num_decoder_layers=4, num_heads=self.nheads)
        # elif self.concat2 == True :
        #     self.transformer = AdvancedTransformerModel(self.projection2, hidden_dim=self.trans_emb_dim,
        #                                                 sequence_length=sequence_length, num_encoder_layers=4,
        #                                                 num_decoder_layers=4, num_heads=self.nheads)

        else:
            self.transformer = AdvancedTransformerModel(self.input_dim + mi_embedder.output_dim + sequence_length, hidden_dim=self.trans_emb_dim,sequence_length=sequence_length,num_encoder_layers=5, num_decoder_layers=5, num_heads=self.nheads)

    def forward(self, y, x, z, t):

        #Embedd time steps
        embedded_t = self.time_siren(t)

        #Embedd observation sequence and Mi behaviors
        x = self.observation_embedder(x)
        z = self.mi_embedder(z)
        x_input = self.merger(x,z)

        if self.concat == True :

            # Project before concat
            t_input = self.t_to_input(embedded_t)
            y_input = self.y_to_input(y)

            concat = torch.cat([y_input,x_input,t_input], dim=-1)

        # elif self.concat2 == True : // embedd following the sequence length dimension (but need to embedd noisy also)
        #     # Project before concat
        #     t_input = self.t_to_input(embedded_t)
        #     y_input = self.y_to_input(y)
        #
        #     concat = torch.cat([y_input, x_input, t_input], dim=2)

        else:
            concat = torch.cat([y, x_input, embedded_t], dim=-1)

        out = self.transformer(concat)
        return out
