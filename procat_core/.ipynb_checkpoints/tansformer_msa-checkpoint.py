import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class TiedRowAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x shape: [seq_len, batch_size, d_model]
        batch_size,seq_len, _ = x.size()

        # Apply QKV projection
        qkv = self.qkv_proj(x)  # qkv shape: [seq_len, batch_size, d_model * 3]
        qkv = qkv.view(batch_size, seq_len, 3, self.nhead, self.d_model // self.nhead)  # qkv shape: [3, batch_size, seq_len, nhead, d_model // nhead]
        qkv = qkv.permute(2, 0, 1, 3, 4)  # qkv shape: [3, batch_size, seq_len, nhead, d_model // nhead]

        # qkv shape: [batch_size,seq_len, d_model * 3]
        # qkv = qkv.view(batch_size,seq_len, 3, self.nhead, self.d_model // self.nhead).transpose(2,0) 
        # print("qkv",qkv.shape)
        q, k, v = qkv[0], qkv[1], qkv[2]  # q, k, v shapes: [batch_size, seq_len, nhead, d_model // nhead]

        # Apply attention
        # print(q.shape,k.shape,v.shape)
        q = q.transpose(1, 2)  # q shape: [batch_size, nhead, seq_len, d_model // nhead]
        k = k.transpose(1, 2)  # k shape: [batch_size, nhead, seq_len, d_model // nhead]
        v = v.transpose(1, 2)  # v shape: [batch_size, nhead, seq_len, d_model // nhead]

        attn_output = []
        scale = self.d_model ** -0.5
        for head in range(self.nhead):
            q_head = q[:, head, :, :]
            k_head = k[:, head, :, :]
            v_head = v[:, head, :, :]
            attn_weights = (q_head @ k_head.transpose(-2, -1)) * scale
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_output_head = attn_weights @ v_head * (self.d_model // self.nhead)
            attn_output.append(attn_output_head)

        # Stack all heads together
        attn_output = torch.stack(attn_output, dim=1)
        
        # print("attn_output",attn_output.shape)
        attn_output = attn_output.transpose(2, 1).contiguous().view(batch_size,seq_len, -1)  # attn_output shape: [seq_len, batch_size, d_model]
        # print(attn_output.shape)
        attn_output = self.o_proj(attn_output)  # attn_output shape: [seq_len, batch_size, d_model]

        return attn_output

class AxialAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(AxialAttention, self).__init__()
        self.row_attention = nn.MultiheadAttention(d_model, nhead)
        self.col_attention = nn.MultiheadAttention(d_model, nhead)

    def forward(self, x):
        # x shape: [seq_len, batch_size, d_model]
        x = x.transpose(0, 1)  # x shape: [batch_size, seq_len, d_model]

        # Split x into rows and columns
        rows = x
        cols = x.transpose(1, 2)  # cols shape: [batch_size, d_model, seq_len]

        # Apply row and column attention
        rows, _ = self.row_attention(rows, rows, rows)  # rows shape: [batch_size, seq_len, d_model]
        cols, _ = self.col_attention(cols, cols, cols)  # cols shape: [batch_size, d_model, seq_len]

        # Combine rows and columns
        x = rows + cols.transpose(1, 2)  # x shape: [batch_size, seq_len, d_model]

        return x.transpose(0, 1)  # return shape: [seq_len, batch_size, d_model]

class MSA_Transformer(nn.Module):
    def __init__(self, d_model,protein_dim ,hid_dim,nhead, num_layers, num_classes):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        # self.o_proj = nn.Linear(protein_dim, d_model)

        self.embedding = nn.Embedding(26, protein_dim, padding_idx=0)
        
        self.row_pos_enc = PositionalEncoding(d_model)
        self.col_pos_enc = PositionalEncoding(d_model)

        self.row_attention = TiedRowAttention(d_model, nhead)
        self.col_attention = nn.MultiheadAttention(d_model, nhead)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
        )

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead), 
            num_layers
        )

        self.output_layer = nn.Linear(d_model, hid_dim)  # 输出层

    def forward(self, x):
        # x is of shape [batch_size, seq_len, d_model]
        # print("self.d_model",self.d_model)
        # Apply row and column positional encodings
        
        # print(x.shape)
        # print(x.shape)
        x = self.embedding(x.long())
        # print(x.shape)
        x = x.transpose(0, 1)  # x shape: [seq_len, batch_size, d_model]
        # x = self.o_proj(x)
        x = self.row_pos_enc(x)
        x = self.col_pos_enc(x.transpose(0, 1)).transpose(1, 0)

        # Apply row attention
        # print("x: ",x.shape)
        x = self.row_attention(x)
        # print("row_attention x: ",x.shape)
    
        # Apply feedforward layer
        x = self.feed_forward(x)
        # print("feed_forward x: ",x.shape)

        # Apply column attention

        # x = x.transpose(0, 1)  # x shape: [batch_size, seq_len, d_model]
        # x, _ = self.col_attention(x, x, x)  # x shape: [batch_size, seq_len, d_model]
        # x = x.transpose(1, 0)  # x shape: [seq_len, batch_size, d_model]
        # print("col_attention x: ",x.shape)


        # Apply feedforward layer
        # x = self.feed_forward(x)
        
        x = self.transformer_encoder(x)
        # print("output layer",x.shape)

        # Apply output layer
        x = self.output_layer(x)  # average over the sequence length
        # print(x.shape)
        return x

class BiDirectionalTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(BiDirectionalTransformer, self).__init__()

        self.embedding = nn.Linear(sequence_length, d_model)  # assuming sequence_length is your third dimension
        self.row_transformer = nn.Transformer(d_model, nhead, num_layers)
        self.col_transformer = nn.Transformer(d_model, nhead, num_layers)
        self.fc = nn.Linear(2 * d_model, d_model)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, sequence_length)

        x = self.embedding(x)  # shape: (batch_size, sequence_length, d_model)

        row_output = self.row_transformer(x.transpose(0, 1))  # shape: (sequence_length, batch_size, d_model)
        col_output = self.col_transformer(x.transpose(0, 2))  # shape: (sequence_length, batch_size, d_model)

        # Combine row and column outputs and transpose back
        combined = torch.cat([row_output, col_output], dim=-1)  # shape: (sequence_length, batch_size, 2*d_model)
        combined = combined.transpose(0, 1)  # shape: (batch_size, sequence_length, 2*d_model)

        output = self.fc(combined)  # shape: (batch_size, sequence_length, d_model)

        return output
