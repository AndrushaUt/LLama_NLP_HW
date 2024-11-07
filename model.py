import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import huggingface_hub as hh

from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset

from tokenizers import normalizers
from tokenizers.normalizers import NFD, Lowercase, Strip

from tokenizers.normalizers import BertNormalizer

from tqdm import tqdm

import time

class RoPEMaskedAttentionHead(nn.Module):
    def __init__(self, d_model, context_window):
        super().__init__()
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        self.R = self.get_rotary_matrix(context_window, d_model)

    def get_rotary_matrix(self, context_window, embedding_dim):
        R = torch.zeros((context_window, embedding_dim, embedding_dim), requires_grad=False)
        for position in range(context_window):
            for i in range(embedding_dim//2):
                theta = 10000. ** (-2.*(i - 1) / embedding_dim)
                m_theta = position * theta
                R[position, 2*i,2*i] = np.cos(m_theta)
                R[position, 2*i,2*i+1] = - np.sin(m_theta)
                R[position, 2*i+1,2*i] = np.sin(m_theta)
                R[position, 2*i+1,2*i+1] = np.cos(m_theta)
        return R
    
    def forward(self, x, return_attn_weights=False):
        b,m,d = x.shape
        
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q_rotated = (torch.bmm(q.transpose(0,1), self.R[:m])).transpose(0,1)
        k_rotated = (torch.bmm(k.transpose(0,1), self.R[:m])).transpose(0,1)

        activations = F.scaled_dot_product_attention(
            q_rotated,k_rotated,v,dropout_p =.1, is_causal=True
        )

        if return_attn_weights:
            attn_mask = torch.tril(torch.ones((m,m)), diagonal=0)
            attn_weights = torch.bmm(q_rotated, k_rotated.transpose(1,2)) / np.sqrt(d) + attn_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            return activations, attn_weights
        return activations


class RoPEMaskedMultiheadAttention(nn.Module):
    def __init__(self, n_heads, d_model, context_window):
        super().__init__()
        self.heads = nn.ModuleList([
            RoPEMaskedAttentionHead(d_model, context_window) for _ in range(n_heads)
        ])
        self.linear = nn.Linear(n_heads * d_model, d_model)
        self.dropout = nn.Dropout(.1)

    def forward(self, x):
        heads = [h(x) for h in self.heads]
        x = torch.cat(heads, dim=-1)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit
    https://arxiv.org/pdf/2002.05202v1.pdf
    """
    def __init__(self, size):
        super().__init__()
        self.linear_gate = nn.Linear(size, size)
        self.linear = nn.Linear(size, size)
        self.beta = torch.randn(1, requires_grad=True)

        self.beta = nn.Parameter(torch.ones(1))
        self.register_parameter("beta", self.beta)

    def forward(self, x): 
        swish_gate = self.linear_gate(x) * torch.sigmoid(self.beta * self.linear_gate(x))
        out = swish_gate * self.linear(x)
        return out

class RMSNorm(nn.Module):
    def __init__(self, layer_shape, eps=1e-8, bias=False):
        super(RMSNorm, self).__init__()
        self.register_parameter("scale", nn.Parameter(torch.ones(layer_shape)))

    def forward(self, x):
        """
        assumes shape is (batch, seq_len, d_model)
        """
        ff_rms = torch.linalg.norm(x, dim=(1,2)) * x[0].numel() ** -.5
        raw = x / ff_rms.unsqueeze(-1).unsqueeze(-1)
        return self.scale[:x.shape[1], :].unsqueeze(0) * raw


class LlamaBlock(nn.Module):
    def __init__(self, context_window, d_model, n_heads):
        super().__init__()
        self.rms = RMSNorm((context_window, d_model))
        
        self.attention = RoPEMaskedMultiheadAttention(n_heads, d_model, context_window)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_model),
            SwiGLU(d_model),
        )

    def forward(self, x):
        x = self.rms(x) # rms pre-normalization
        x = x + self.attention(x)

        x = self.rms(x) # rms pre-normalization
        x = x + self.feedforward(x)
        return x

class Llama(nn.Module):
    def __init__(self, context_window, vocab_size, d_model, n_layers, n_heads):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.llama_blocks = nn.Sequential(
            *[LlamaBlock(context_window, d_model, n_heads) for i in range(n_layers)]
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            SwiGLU(d_model),
            nn.Linear(d_model, vocab_size),
        )

        # print("model params:", sum([m.numel() for m in self.parameters()]))

    def forward(self, idx, targets=None):
        x = self.embeddings(idx)
        x = self.llama_blocks(x)
        logits = self.ffn(x)

        if targets is None:
            return logits
        
        else:
            loss = F.cross_entropy(logits.view(-1, 32000), targets.view(-1))
            return logits, loss


    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info


hh.login("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")

dataset = load_dataset("ashaba1in/small_openwebtext")
auto_tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')
auto_tokenizer.pad_token = auto_tokenizer.eos_token
auto_tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), Strip(), BertNormalizer()])
def tokenize_function(examples):
    return auto_tokenizer(examples['text'], truncation=True, padding=True)

tokenized_datasets = dataset['train'].select(range(1))

ids = [auto_tokenizer.encode(d['text']) for d in tokenized_datasets]

dataset = Dataset.from_dict({"input_ids": ids})


data_collator = DataCollatorWithPadding(tokenizer=auto_tokenizer)

def get_batches(data, batch_size, context_window):
    loader = DataLoader(data, batch_size=batch_size, collate_fn=data_collator)

    for batch in loader:
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask', None)  # Используется, если attention_mask доступен

        # Генерация случайных индексов, которые гарантируют наличие данных, отличных от паддинга
        idx_options = []

        for i in range(input_ids.size(0)):  # Для каждого предложения в батче
            valid_positions = (input_ids[i] != auto_tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
            valid_positions = valid_positions[valid_positions < (input_ids.size(1) - context_window)]
            
            if len(valid_positions) > 0:
                # Доступные начальные индексы, которые не выходят за пределы
                valid_idx = torch.multinomial(torch.ones(len(valid_positions)), 1).item()
                idx_options.append(valid_positions[valid_idx])
            else:
                # Если по какой-то причине не нашлось допустимой позиции
                idx_options.append(0)  # заполнитель, берем с начала

        x = torch.stack([input_ids[i, idx:idx + context_window] for i, idx in enumerate(idx_options)])
        y = torch.stack([input_ids[i, idx + 1:idx + context_window + 1] for i, idx in enumerate(idx_options)])

        yield x, y

batch_size = 4
context_window = 16

# Используйте весь датасет для обучения
data_iterator = get_batches(dataset, batch_size=batch_size, context_window=context_window)

def train(model, optimizer, scheduler=None, print_logs=False):
    for epoch in range(10):
        model.train()
        optimizer.zero_grad()
        train_losses = []
        val_losses = []
        for x, y in tqdm(get_batches(dataset, batch_size=batch_size, context_window=context_window), desc=f"Epoch={epoch + 1}, Training"):
            logits, loss = model(x, targets=y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        model.eval()
        for x, y in tqdm(get_batches(dataset, batch_size=batch_size, context_window=context_window), desc=f"Epoch={epoch+1}, Validating"):
            _, loss = model(x, targets=y)
            val_losses.append(loss.item())

        if scheduler:
            scheduler.step()

        print(f"Epoch={epoch+1}, train_loss: {np.mean(train_losses)}, val_loss: {np.mean(val_losses)}")

learning_rate = 3e-4

llama_model = Llama(
    context_window=context_window,
    vocab_size=32000,
    d_model=512,
    n_layers=8,
    n_heads=8,
)

optimizer = torch.optim.AdamW(llama_model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1)

print(llama_model)

train(llama_model, optimizer)