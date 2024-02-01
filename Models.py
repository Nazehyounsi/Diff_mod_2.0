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

class EventEmbedder(nn.Module):
    def __init__(self, num_event_types, event_embedding_dim, continuous_embedding_dim,output_dim,event_weights=None):
        super(EventEmbedder, self).__init__()
        self.event_embedding = nn.Embedding(num_event_types, event_embedding_dim)
        self.duration_embedding = nn.Linear(1, continuous_embedding_dim)
        self.start_time_embedding = nn.Linear(1, continuous_embedding_dim)
        self.output_dim = output_dim
        input_dim = event_embedding_dim + 2 * continuous_embedding_dim
        self.interaction_lstm = nn.LSTM(input_dim,output_dim, batch_first=True)

        # Initialize event type weights
        if event_weights is None:
            # If no weights are provided, initialize as ones (i.e., no change in importance).
            self.event_weights = nn.Parameter(torch.ones(num_event_types), requires_grad=False)
        else:
            self.event_weights = nn.Parameter(torch.tensor(event_weights), requires_grad=False)

    def forward(self, x):

        event_type_indices = x[:, :, 0].long()
        event_type_embed = self.event_embedding(event_type_indices)
        duration_embed = self.duration_embedding(x[:, :, 1].unsqueeze(-1))
        start_time_embed = self.start_time_embedding(x[:, :, 2].unsqueeze(-1))

        # if add weights on events
        event_type_weights = self.event_weights[event_type_indices]
        event_type_embed = event_type_embed * event_type_weights.unsqueeze(-1)

        concatenated_embeddings = torch.cat([event_type_embed, duration_embed, start_time_embed], dim=-1)
        output, _ = self.interaction_lstm(concatenated_embeddings)

        last_element_weight = 1.5  # This is an example weight, adjust as needed
        weighted_last_element = output[:, -1, :] * last_element_weight

        # Replace the last element with the weighted version
        output = torch.cat((output[:, :-1, :], weighted_last_element.unsqueeze(1)), dim=1)

        return output


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
    def __init__(self, nn_model, event_embedder, betas, n_T, device, x_dim, y_dim, drop_prob=0.1, guide_w=0.0):
        super(Model_Cond_Diffusion, self).__init__()
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.nn_model = nn_model
        self.n_T = n_T
        self.device = device
        self.event_embedder = event_embedder
        self.x_sequence_transformer = SequenceTransformer(16, 16, 8)
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.guide_w = guide_w

    def loss_on_batch(self, x_batch, y_batch):

        _ts = torch.randint(1, self.n_T + 1, (y_batch.shape[0], 1)).to(self.device)

        # dropout context with some probability
       #context_mask = torch.bernoulli(torch.zeros(x_batch.shape[0]) + self.drop_prob).to(self.device)

        # randomly sample some noise, noise ~ N(0, 1)
        noise = torch.randn_like(y_batch).to(self.device)
        self.y_dim = noise.shape

        # add noise to clean target actions
        y_t = self.sqrtab[_ts] * y_batch + self.sqrtmab[_ts] * noise
        # use nn model to predict noise
        noise_pred_batch = self.nn_model(y_t, x_batch, _ts / self.n_T) #ici possible d'ajouter context_mask en input

        # return mse between predicted and true noise
        return self.loss_mse(noise, noise_pred_batch)

    def sample(self, x_batch, return_y_trace=False):
        # also use this as a shortcut to avoid doubling batch when guide_w is zero
        is_zero = False
        if self.guide_w > -1e-3 and self.guide_w < 1e-3:
            is_zero = True

        # how many noisy actions to begin with
        n_sample = x_batch.shape[0]

        y_shape = (n_sample, 3)


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

            eps = self.nn_model(y_i, x_batch, t_is)
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

#
class FCBlock(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        # one layer of non-linearities (just a useful building block to use below)
        self.model = nn.Sequential(
            nn.Linear(in_feats, out_feats),
            nn.BatchNorm1d(num_features=out_feats),
            nn.GELU(),
        )

    def forward(self, x):
        return self.model(x)


class SequenceTransformer(nn.Module):
    def __init__(self, input_dim, trans_emb_dim, nheads):
        super(SequenceTransformer, self).__init__()
        self.trans_emb_dim = trans_emb_dim
        self.input_projection = nn.Linear(input_dim, trans_emb_dim)
        self.flag = False
        if input_dim != trans_emb_dim:
            self.flag = True
        else:
            pass



        # Positional embedding for transformer
        self.pos_embed = TimeSiren(1, trans_emb_dim)

        # Transformer encoder layer
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=trans_emb_dim, nhead=nheads, dim_feedforward=trans_emb_dim),
            num_layers=6
        )

        # Output linear layer to get a single vector representation
        self.output_linear = nn.Linear(trans_emb_dim, trans_emb_dim)

    def forward(self, x):

        if self.flag == True:
            x = self.input_projection(x)
        else:
            pass

        # Transpose x to match PyTorch's transformer input shape requirements
        x = x.transpose(0, 1)  # Shape: [seq_length, batch_size, features_dim]

        # Apply positional encoding
        seq_length, batch_size, _ = x.shape
        positions = torch.arange(seq_length, dtype=torch.float, device=x.device).unsqueeze(-1)
        pos_encoding = self.pos_embed(positions)


        x = x + pos_encoding.unsqueeze(1).expand(-1, batch_size, -1)

        # Pass through the transformer encoder
        x = self.transformer_encoder(x)

        # Aggregate sequence information into a single vector (e.g., by averaging)
        x = x.mean(dim=0)

        # Apply final linear layer
        x = self.output_linear(x)

        return x
class TransformerEncoderBlock(nn.Module):
    def __init__(self, trans_emb_dim, transformer_dim, nheads):
        super(TransformerEncoderBlock, self).__init__()
        # mainly going off of https://jalammar.github.io/illustrated-transformer/

        self.trans_emb_dim = trans_emb_dim
        self.transformer_dim = transformer_dim
        self.nheads = nheads

        self.input_to_qkv1 = nn.Linear(self.trans_emb_dim, self.transformer_dim * 3)
        self.multihead_attn1 = nn.MultiheadAttention(self.transformer_dim, num_heads=self.nheads)
        self.attn1_to_fcn = nn.Linear(self.transformer_dim, self.trans_emb_dim)
        self.attn1_fcn = nn.Sequential(
            nn.Linear(self.trans_emb_dim, self.trans_emb_dim * 4),
            nn.GELU(),
            nn.Linear(self.trans_emb_dim * 4, self.trans_emb_dim),
        )
        self.norm1a = nn.BatchNorm1d(self.trans_emb_dim)
        self.norm1b = nn.BatchNorm1d(self.trans_emb_dim)

    def split_qkv(self, qkv):
        assert qkv.shape[-1] == self.transformer_dim * 3
        q = qkv[:, :, :self.transformer_dim]
        k = qkv[:, :, self.transformer_dim: 2 * self.transformer_dim]
        v = qkv[:, :, 2 * self.transformer_dim:]
        return (q, k, v)



    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        qkvs1 = self.input_to_qkv1(inputs)
        # shape out = [3, batchsize, transformer_dim*3]

        qs1, ks1, vs1 = self.split_qkv(qkvs1)
        # shape out = [3, batchsize, transformer_dim]

        attn1_a = self.multihead_attn1(qs1, ks1, vs1, need_weights=False)
        attn1_a = attn1_a[0]
        # shape out = [3, batchsize, transformer_dim = trans_emb_dim x nheads]

        attn1_b = self.attn1_to_fcn(attn1_a)
        attn1_b = attn1_b / 1.414 + inputs / 1.414  # add residual
        # shape out = [3, batchsize, trans_emb_dim]

        # normalise
        attn1_b = self.norm1a(attn1_b.transpose(0, 2).transpose(0, 1))
        attn1_b = attn1_b.transpose(0, 1).transpose(0, 2)
        # batchnorm likes shape = [batchsize, trans_emb_dim, 3]
        # so have to shape like this, then return

        # fully connected layer
        attn1_c = self.attn1_fcn(attn1_b) / 1.414 + attn1_b / 1.414
        # shape out = [3, batchsize, trans_emb_dim]

        # normalise
        # attn1_c = self.norm1b(attn1_c)
        attn1_c = self.norm1b(attn1_c.transpose(0, 2).transpose(0, 1))
        attn1_c = attn1_c.transpose(0, 1).transpose(0, 2)
        return attn1_c

class Model_mlp_diff(nn.Module):
    def __init__(self, event_embedder, y_dim, net_type="transformer"):
        super(Model_mlp_diff, self).__init__()
        self.event_embedder = event_embedder
        self.time_siren = TimeSiren(1, event_embedder.output_dim)
        self.net_type = net_type

        # Transformer specific initialization
        self.nheads = 16  # Number of heads in multihead attention
        self.trans_emb_dim = 64  # Transformer embedding dimension
        self.transformer_dim = self.trans_emb_dim * self.nheads

        # Initialize SequenceTransformers for y and x
        self.x_sequence_transformer = SequenceTransformer(event_embedder.output_dim, 16, 8)


        # Linear layers to project embeddings to transformer dimension
        self.t_to_input = nn.Linear(event_embedder.output_dim, self.trans_emb_dim)
        self.y_to_input = nn.Linear(3, self.trans_emb_dim)
        self.x_to_input = nn.Linear(16, self.trans_emb_dim)

        # Positional embedding for transformer
        self.pos_embed = TimeSiren(1, self.trans_emb_dim)

        # Transformer blocks
        self.transformer_block1 = TransformerEncoderBlock(self.trans_emb_dim, self.transformer_dim, self.nheads)

        # Final layer to project transformer output to desired output dimension
        self.final = nn.Linear(self.trans_emb_dim * 3, 3)  # Adjust the output dimension as needed

    def forward(self, y, x, t):

        embedded_t = self.time_siren(t)

        # CHUNK DESCRIPTOR CASE !
        # in the case we need to process separatly observation and past action through different pipelines (not only with weighted event_embedder)
        # observations_past_act = x[:, :-1, :]  # All elements except the last
        # chunk_descriptor = x[:, -1, :] # the last element of x
        # embed_chunk_descriptor = embedding_class_special_for_chunk_descriptor(chunk_descriptor) (introduce embedding_class in init of model_mlp_diff)
        # x = self.event_embedder(observation_past_act)
        # x = x + embed_chunk_descriptor

        #comment this if chunk descriptor case
        x = self.event_embedder(x)

        # Transform sequences
        x = self.x_sequence_transformer(x)
        #transformed_y = self.y_sequence_transformer(y)

        # Project embeddings to transformer dimension
        t_input = self.t_to_input(embedded_t)
        y_input = self.y_to_input(y)
        x_input = self.x_to_input(x)


        #t_input = t_input.unsqueeze(1).repeat(1, x.shape[1], 1)

        # Add positional encoding
        t_input += self.pos_embed(torch.zeros(x.shape[0], 1).to(x.device) + 1.0)
        y_input += self.pos_embed(torch.zeros(y.shape[0], 1).to(x.device) + 2.0)
        x_input += self.pos_embed(torch.zeros(x.shape[0], 1).to(x.device) + 3.0)

        # Concatenate inputs for transformer
        inputs = torch.cat((t_input[None, :, :], y_input[None, :, :], x_input[None, :, :]), 0)

        # Pass through transformer blocks
        block_output = self.transformer_block1(inputs)

        # Flatten and add final linear layer
        transformer_out = block_output.transpose(0, 1)  # Roll batch to first dim

        flat = torch.flatten(transformer_out, start_dim=1, end_dim=2)
        out = self.final(flat)
        return out
