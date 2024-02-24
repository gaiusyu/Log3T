from pytorch_pretrained_bert import BertTokenizer
import torch
import math
import torch.nn as nn
import numpy as np


torch.manual_seed(0)
torch.cuda.manual_seed(0)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained('Vocab/Vocab.txt')
vocab_size=len(tokenizer.vocab)
maxlen = 128
word_maxlen=16
batch_size =24
n_layers = 1  # number of encoder layer
n_heads = 1  # number of heads
d_model = 64  # model dimension
d_ff = 64 * 4  #  FeedForward dimension
d_k = d_v = 32 # QKV dimension


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, seq_len = seq_q.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_q.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]
    return pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]


def gelu(x):
    """
      Implementation of the gelu activation function.
      For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
      0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
      Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
        self.pos_embed = nn.Embedding(maxlen * word_maxlen, d_model)  # position embedding
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x).to(device)  # [seq_len] -> [batch_size, seq_len]
        embedding = self.tok_embed(x.to(device)) + self.pos_embed(pos.to(device))
        return self.norm(embedding)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask,stage):
        scores = torch.matmul(Q.to(device), K.to(device).transpose(-1, -2)) / np.sqrt(
            d_k)  # scores : [batch_size, n_heads, seq_len, seq_len]
        scores.masked_fill_(attn_mask.to(device), -1e9).to(
            device)  # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores.to(device)).to(device)
        at = attn.squeeze(dim=0).squeeze(dim=0)
        if stage == "parse":
            return at.detach().cpu().numpy().to(device)
        context = torch.matmul(attn.to(device), V.to(device))
        return context.to(device)


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.fc = nn.Linear(n_heads * d_v, d_model)

    def forward(self, Q, K, V, attn_mask,stage):
        # q: [batch_size, seq_len, d_model], k: [batch_size, seq_len, d_model], v: [batch_size, seq_len, d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q.to(device)).view(batch_size, -1, n_heads, d_k).transpose(1, 2).to(
            device)  # q_s: [batch_size, n_heads, seq_len, d_k]
        k_s = self.W_K(K.to(device)).view(batch_size, -1, n_heads, d_k).transpose(1, 2).to(
            device)  # k_s: [batch_size, n_heads, seq_len, d_k]
        v_s = self.W_V(V.to(device)).view(batch_size, -1, n_heads, d_v).transpose(1, 2).to(
            device)  # v_s: [batch_size, n_heads, seq_len, d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1,
                                                  1).to(
            device)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, seq_len, d_v], attn: [batch_size, n_heads, seq_len, seq_len]
        context = ScaledDotProductAttention()(q_s.to(device), k_s.to(device), v_s.to(device), attn_mask.to(device),stage)
        if stage == "parse":
            return context
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                            n_heads * d_v).to(
            device)  # context: [batch_size, seq_len, n_heads, d_v]
        output = self.fc(context).to(device)
        # return nn.LayerNorm(d_model).to(device)(output.to(device)).to(device)
        if stage == 'check':
            return context
        output_o = nn.LayerNorm(d_model).to(device)(output.to(device) +  residual.to(device)).to(
            device)
        return output_o  # output: [batch_size, seq_len, d_model]


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.fc2(gelu(self.fc1(x.to(device)).to(device)).to(device))


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask,stage):
        enc_outputs = self.enc_self_attn(enc_inputs.to(device), enc_inputs.to(device), enc_inputs.to(device),
                                         enc_self_attn_mask.to(device),stage)  # enc_inputs to same Q,K,V
        if stage == "check":
            return enc_outputs.to(device)
        enc_outputs = self.pos_ffn(enc_outputs).to(device)  # enc_outputs: [batch_size, seq_len, d_model]
        return enc_outputs.to(device)


class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(0.5),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(d_model, 2)
        self.linear = nn.Linear(d_model, d_model)
        # self.linear2=nn.Linear(d_model, d_model)
        self.activ2 = gelu
        # fc2 is shared with embedding layer

        self.fc2 = nn.Linear(d_model, 1, bias=False)

    def forward(self, input_ids, stage):
        output = self.embedding(input_ids.to(device)).to(device)  # [bach_size, seq_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(input_ids.to(device), input_ids.to(device)).to(
            device)  # [batch_size, maxlen, maxlen]
        for layer in self.layers:
            # output: [batch_size, max_len, d_model]
            output = layer(output.to(device), enc_self_attn_mask.to(device),stage)
            if stage == "check":
                return output

        h_masked = self.linear(output.to(device))  # [batch_size, max_pred, d_model]
        logits_lm = torch.sigmoid(self.fc2(h_masked).to(device))  # [batch_size, max_pred, vocab_size]     #
        if stage == 'standard':
            h_masked1 = self.linear2(output.to(device))  # [batch_size, max_pred, d_model]
            logits_lm1 = torch.sigmoid(self.fc2(h_masked1).to(device))  # [batch_size, max_pred, vocab_size]     #
            return logits_lm, logits_lm1
        return logits_lm


