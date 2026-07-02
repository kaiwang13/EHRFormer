"""Unified EHRFormer model architecture for pretraining and finetuning."""
import torch.nn as nn
import torch
from einops import rearrange
from transformers import BertConfig
from transformers import GPT2Model, BertModel
from transformers.models.bert.modeling_bert import BertEncoder
# from flash_attn.models.bert import BertEncoder, BertModel

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_
        return output, None
        
class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
        
def masked_mean(x, mask):
    """Mean of x [B, L, D] over L, weighted by mask [B, L]."""
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)


class CohortDiscriminator(nn.Module):
    """Predicts the originating cohort from the pooled patient representation.

    Placed behind a gradient-reversal layer so that minimizing its loss drives
    the encoder toward cohort-invariant (batch-effect-free) representations
    (paper: cohort-agnostic adversarial training)."""
    def __init__(self, hidden, proj, n_cohorts):
        super().__init__()
        self.grl = GradientReversalLayer()
        self.fc = nn.Sequential(nn.Linear(hidden, proj), nn.ReLU(), nn.Linear(proj, n_cohorts))

    def forward(self, x, lambda_=1.0):
        self.grl.lambda_ = lambda_
        return self.fc(self.grl(x))


class MissingnessDiscriminator(nn.Module):
    """Predicts the per-visit, per-feature missingness pattern from the
    representation, behind a gradient-reversal layer, to encourage
    missing-invariant representations (paper: missing-data discrimination)."""
    def __init__(self, hidden, proj, n_feats):
        super().__init__()
        self.grl = GradientReversalLayer()
        self.fc = nn.Sequential(nn.Linear(hidden, proj), nn.ReLU(), nn.Linear(proj, n_feats))

    def forward(self, x, lambda_=1.0):   # x: [B, L, D] -> [B, L, n_feats]
        self.grl.lambda_ = lambda_
        return self.fc(self.grl(x))

class EHREmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.register_buffer(
            "token_type_ids", torch.arange(config['n_category_feats']+config['n_float_feats']+1), persistent=False
        )
        self.register_buffer(
            "one", torch.ones(1, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "zero", torch.zeros(1, dtype=torch.long), persistent=False
        )

        dec_hidden = config['transformer'].hidden_size
        enc_hidden = config.get('encoder_hidden', dec_hidden)      # paper: 1024
        n_enc_layers = config.get('n_encoder_layers', 1)           # paper: 24
        self.bert = BertModel(BertConfig(
            vocab_size=config['n_category_values']+config['n_float_values']+2,  # 1 for padding, 0 for CLS
            hidden_size=enc_hidden,
            num_hidden_layers=n_enc_layers,
            num_attention_heads=config.get('encoder_heads', 12),
            intermediate_size=enc_hidden * 4,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=8192,
            type_vocab_size=config['n_category_feats']+config['n_float_feats']+1,  # 0 for CLS
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=1,
            position_embedding_type="none",
            use_cache=True,
            classifier_dropout=None,
            attn_implementation="eager",
        ))
        # project the examination-encoder output to the temporal-decoder width if they differ
        self.proj = nn.Linear(enc_hidden, dec_hidden) if enc_hidden != dec_hidden else nn.Identity()

    def forward(self, cat_feats, float_feats):  # cat_feats: (b, nc, l), float_feats: (b, nf, l)
        B = cat_feats.shape[0]
        L = cat_feats.shape[2]
        cat_feats_mask = cat_feats == -1
        float_feats_mask = float_feats == -1
        attention_mask = torch.cat([self.one.unsqueeze(1).unsqueeze(0).expand(B, -1, L), ~cat_feats_mask, ~float_feats_mask], dim=1)
        attention_mask = rearrange(attention_mask, 'b n l -> (b l) n')

        cat_feats = cat_feats + 2
        cat_feats[cat_feats_mask] = 1
        float_feats = float_feats + 2 + self.config['n_category_values']
        float_feats[float_feats_mask] = 1
        input_ids = torch.cat([self.zero.unsqueeze(1).unsqueeze(0).expand(B, -1, L), cat_feats, float_feats], dim=1)
        input_ids = rearrange(input_ids, 'b n l -> (b l) n')

        BL = input_ids.shape[0]
        token_type_ids = self.token_type_ids.unsqueeze(0).expand(BL, -1)
        ft_emb = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        ).pooler_output
        ft_emb = self.proj(ft_emb)
        ft_emb = rearrange(ft_emb, '(b l) d -> b l d', b=B)
        return ft_emb  # time_index: (b, l, d)

class MultiTaskHead2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_dim = config['transformer'].hidden_size
        self.n_cls = self.config['n_cls']

        self.cls_fcs = nn.ModuleList(nn.Sequential(
            nn.Linear(self.output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n)
        ) for n in self.n_cls)

    def forward(self, x, B):
        results = []
        for i, cls_fc in enumerate(self.cls_fcs):
            y_cls = cls_fc(x)
            y_cls = rearrange(y_cls, '(b l) d -> b l d', b=B)
            results.append(y_cls)
        return results

class MultiTaskHeadPretrain(nn.Module):
    """Pretraining head: reconstructs masked features in the CURRENT visit and
    predicts the NEXT visit's values (paper: masked prediction over current and
    next examination), plus a per-visit age-regression head (biological clock)."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_dim = config['transformer'].hidden_size
        self.n_cls = len(config.get('cls_label_names', []))
        self.n_reg = len(config.get('reg_label_names', []))
        self.predict_next = config.get('predict_next_visit', True)

        def cls_head():
            return nn.Sequential(nn.Linear(self.output_dim, 256), nn.ReLU(), nn.Linear(256, 2))

        def reg_head():
            return nn.Sequential(nn.Linear(self.output_dim, 256), nn.ReLU(), nn.Linear(256, 1))

        # current-visit reconstruction heads
        self.cls_heads = nn.ModuleList([cls_head() for _ in range(self.n_cls)])
        self.reg_heads = nn.ModuleList([reg_head() for _ in range(self.n_reg)])
        # next-visit prediction heads (autoregressive)
        if self.predict_next:
            self.next_cls_heads = nn.ModuleList([cls_head() for _ in range(self.n_cls)])
            self.next_reg_heads = nn.ModuleList([reg_head() for _ in range(self.n_reg)])
        # biological-clock age regression
        self.age_head = reg_head()

    def forward(self, hidden_states, cat_feats, float_feats, valid_mask):
        out = {
            'cls': [head(hidden_states) for head in self.cls_heads],              # each [B, L, 2]
            'reg': [head(hidden_states).squeeze(-1) for head in self.reg_heads],  # each [B, L]
            'age': self.age_head(hidden_states).squeeze(-1),                      # [B, L]
        }
        if self.predict_next:
            out['cls_next'] = [head(hidden_states) for head in self.next_cls_heads]
            out['reg_next'] = [head(hidden_states).squeeze(-1) for head in self.next_reg_heads]
        return out

class EHRFormer(nn.Module):
    """Unified EHRFormer model for both pretraining and finetuning."""
    def __init__(self, config):
        super().__init__()
        self.config = config 
        self.mode = config.get('mode', 'finetune')
        
        # VAE components
        self.n_cls = len(config.get('cls_label_names', []))
        self.n_reg = len(config.get('reg_label_names', []))
        
        # BERT config for VAE encoders/decoder
        self.bert_config = BertConfig(
            vocab_size=config['n_category_values']+config['n_float_values']+2,
            hidden_size=config['transformer'].hidden_size,
            num_hidden_layers=2,
            num_attention_heads=12,
            intermediate_size=config['transformer'].hidden_size * 4,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=8192,
            type_vocab_size=config['n_category_feats']+config['n_float_feats']+1,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=1,
            position_embedding_type="none",
            use_cache=True,
            classifier_dropout=None,
            attn_implementation="eager",
        )

        self.ehr_embed = EHREmbedding(config)
        self.transformer = GPT2Model(config['transformer'])
        
        # VAE components
        self.ehr_mu = BertEncoder(self.bert_config)
        self.ehr_std = BertEncoder(self.bert_config)
        self.decoder = BertEncoder(self.bert_config)
        
        if self.mode == 'finetune':
            self.head = MultiTaskHead2(config)
        else:
            self.pretrain_head = MultiTaskHeadPretrain(config)
            # domain-adversarial discriminators (gradient reversal) for
            # cohort-invariant and missing-invariant representations
            hidden = config['transformer'].hidden_size
            proj = config.get('proj_dim', 256)
            n_feats = config['n_category_feats'] + config['n_float_feats']
            self.cohort_disc = CohortDiscriminator(hidden, proj, config.get('n_cohorts', 1))
            self.missing_disc = MissingnessDiscriminator(hidden, proj, n_feats)

    def reparameterize(self, mu, logvar):
        """VAE reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, cat_feats, float_feats, valid_mask, time_index, **kwargs):
        B = cat_feats.shape[0]
        ft_emb = self.ehr_embed(cat_feats, float_feats)
        y = self.transformer(
            inputs_embeds=ft_emb,
            position_ids=time_index,
            attention_mask=valid_mask
        ).last_hidden_state
        
        # VAE encoding - get mu and std
        mu_z = self.ehr_mu(y).last_hidden_state
        std_z = self.ehr_std(y).last_hidden_state

        # Reparameterize
        z = self.reparameterize(mu_z, std_z)

        # Decode
        h = self.decoder(z).last_hidden_state
        
        if self.mode == 'finetune':
            h = rearrange(h, 'b l d -> (b l) d')
            y_cls = self.head(h, B)
            return y_cls, mu_z, std_z
        else:  # pretrain mode
            # reconstruct current + predict next examination, + age (biological clock)
            pretrain_output = self.pretrain_head(h, cat_feats, float_feats, valid_mask)
            # domain-adversarial heads (gradient reversal happens inside the discriminators)
            grl_lambda = self.config.get('grl_lambda', 1.0)
            pretrain_output['cohort'] = self.cohort_disc(masked_mean(h, valid_mask), grl_lambda)
            pretrain_output['missing'] = self.missing_disc(h, grl_lambda)
            return pretrain_output, mu_z, std_z