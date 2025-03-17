# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from torch.optim import lr_scheduler
from tqdm import trange
import tqdm
from tqdm import tqdm
from sklearn.metrics import roc_auc_score,log_loss,mean_absolute_error,mean_squared_error,r2_score,accuracy_score,f1_score,recall_score,precision_score

class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]

        # query = key = value [batch size, sent len, hid dim]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # Q, K, V = [batch size, sent len, hid dim]

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        # K, V = [batch size, n heads, sent len_K, hid dim // n heads]
        # Q = [batch size, n heads, sent len_q, hid dim // n heads]
        
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, sent len_Q, sent len_K]
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))

        # attention = [batch size, n heads, sent len_Q, sent len_K]

        x = torch.matmul(attention, V)

        # x = [batch size, n heads, sent len_Q, hid dim // n heads]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, sent len_Q, n heads, hid dim // n heads]

        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))

        # x = [batch size, src sent len_Q, hid dim]

        x = self.fc(x)

        # x = [batch size, sent len_Q, hid dim]

        return x


class Encoder(nn.Module):
    """protein feature extraction."""
    def __init__(self, protein_dim, hid_dim, n_layers,kernel_size , dropout, device):
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd (for now)"

        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.device = device
        #self.pos_embedding = nn.Embedding(1000, hid_dim)
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.convs = nn.ModuleList([nn.Conv1d(hid_dim, 2*hid_dim, kernel_size, padding=(kernel_size-1)//2) for _ in range(self.n_layers)])   # convolutional layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.input_dim,self.hid_dim)

    def forward(self, protein):
        #pos = torch.arange(0, protein.shape[1]).unsqueeze(0).repeat(protein.shape[0], 1).to(self.device)
        #protein = protein + self.pos_embedding(pos)
        #protein = [batch size, protein len,protein_dim]
        # print("protein:",protein.shape)
        conv_input = self.fc(protein)
        # conv_input=[batch size,protein len,hid dim]
        #permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1)
        #conv_input = [batch size, hid dim, protein len]
        for i, conv in enumerate(self.convs):
            #pass through convolutional layer
            conved = conv(self.dropout(conv_input))
            #conved = [batch size, 2*hid dim, protein len]

            #pass through GLU activation function
            conved = F.glu(conved, dim=1)
            #conved = [batch size, hid dim, protein len]

            #apply residual connection / high way
            conved = (conved + conv_input) * self.scale
            #conved = [batch size, hid dim, protein len]

            #set conv_input to conved for next loop iteration
            conv_input = conved

        conved = conved.permute(0,2,1)
        # conved = [batch size,protein len,hid dim]
        return conved


class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.pf_dim = pf_dim

        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)  # convolution neural units
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)  # convolution neural units

        self.do = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, sent len, hid dim]

        x = x.permute(0, 2, 1)

        # x = [batch size, hid dim, sent len]

        x = self.do(F.relu(self.fc_1(x)))

        # x = [batch size, pf dim, sent len]

        x = self.fc_2(x)

        # x = [batch size, hid dim, sent len]

        x = x.permute(0, 2, 1)

        # x = [batch size, sent len, hid dim]

        return x

class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()

        self.ln = nn.LayerNorm(hid_dim)
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.ea = self_attention(hid_dim, n_heads, dropout, device)
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        # trg = [batch_size, compound len, atom_dim]
        # src = [batch_size, protein len, hid_dim] # encoder output
        # trg_mask = [batch size, compound sent len]
        # src_mask = [batch size, protein len]

        trg = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))

        trg = self.ln(trg + self.do(self.ea(trg, src, src, src_mask)))
        
        # trg = self.ln(trg + self.do(self.ea(trg, src, trg, src_mask)))

        trg = self.ln(trg + self.do(self.pf(trg)))

        return trg


class Decoder(nn.Module):
    """ compound feature extraction."""
    def __init__(self, atom_dim, hid_dim, n_layers, n_heads, pf_dim, decoder_layer, self_attention,
                 positionwise_feedforward, dropout, device):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.output_dim = atom_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.decoder_layer = decoder_layer
        self.self_attention = self_attention
        self.positionwise_feedforward = positionwise_feedforward
        self.dropout = dropout
        self.device = device
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.layers = nn.ModuleList(
            [decoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device)
             for _ in range(n_layers)])
        self.ft = nn.Linear(atom_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.fc_1 = nn.Linear(hid_dim, 256)
        self.fc_2 = nn.Linear(256, 2)

    def forward(self, trg, src, trg_mask=None,src_mask=None):
        # trg = [batch_size, compound len, atom_dim]
        # src = [batch_size, protein len, hid_dim] # encoder output
        # print(trg.shape)
        trg = self.ft(trg)
        # trg = [batch size, compound len, hid dim]

        for layer in self.layers:
            trg = layer(trg, src)

        # trg = [batch size, compound len, hid dim]
        """Use norm to determine which atom is significant. """
        norm = torch.norm(trg,dim=2)
        # norm = [batch size,compound len]
        norm = F.softmax(norm,dim=1)
        # norm = [batch size,compound len]
        norm = norm.unsqueeze(-1)
        # norm = [batch size, compound len, 1]
        weighted_trg = trg * norm
    # weighted_trg = [batch size, compound len, hid dim]

        v = weighted_trg.sum(dim=1)  

#         sum_ = torch.zeros((self.hid_dim)).to(self.device)
#         # sum = [hid_dim]
        
#         for i in range(trg.shape[0]):  # Loop over the batch dimension
#             v = trg[i,]
#             v = v * norm[i]
#             v = torch.sum(v, dim=0)
#             # v = [hid_dim]
#             sum_ += v

        # sum = [hid_dim]
        # print("sum_:",sum_.shape)
        label = F.relu(self.fc_1(v))
        # print("label:",label.shape)
        # label = self.fc_2(label)
        # print("label:",label.shape)

        return label


class Predictor(nn.Module):
    def __init__(self, encoder, decoder, device, atom_dim=34):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.weight = nn.Parameter(torch.FloatTensor(atom_dim, atom_dim))
        self.init_weight()

    def init_weight(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        
    def gcn(self, input, adj):
        output = []
        for inp, ad in zip(input, adj):
            # inp =[num_node, atom_dim]
            # ad = [num_node, num_node]
            support = torch.mm(inp, self.weight)
            # support =[num_node,atom_dim]
            out = torch.mm(ad, support)
            output.append(out)
        return torch.stack(output)

    def forward(self, compound, adj, protein):
        # compound = [atom_num, atom_dim]
        # adj = [atom_num, atom_num]
        # protein = [protein len, 100]
        
        compound = self.gcn(compound, adj)
        # print("compound:",compound.shape,adj.shape)
        # compound = torch.unsqueeze(compound, dim=0)
        # print("compound:",compound.shape,adj.shape)
        # compound = [batch size=1 ,atom_num, atom_dim]

        # protein = torch.unsqueeze(protein, dim=0)
        # protein =[ batch size=1,protein len, protein_dim]
        # print("protein",protein.shape)
        # print("protein:",protein.shape)
        enc_src = self.encoder(protein)
        # print("protein:",enc_src.shape)
        # print("enc_src",enc_src.shape)
        # enc_src = [batch size, protein len, hid dim]
        # print("compound",compound.shape)

        out = self.decoder(compound, enc_src)
        # out = [batch size, 2]
        #out = torch.squeeze(out, dim=0)
        return out

    def __call__(self, inputs, correct_interaction, method="train"):

        if method=="train":
            correct_interaction = correct_interaction.squeeze(dim=1)
            compound, adj, protein = inputs
            Loss = nn.CrossEntropyLoss()
            predicted_interaction = self.forward(compound, adj, protein)
            loss = Loss(predicted_interaction, correct_interaction)
            # print("loss:",loss)
            return loss

        elif method=="evaluation":
            with torch.no_grad():
                correct_interaction = correct_interaction.squeeze(dim=1)
                compound, adj, protein = inputs
                predicted_interaction = self.forward(compound, adj, protein)
                correct_labels = correct_interaction.squeeze().to('cpu').data.numpy()
                
                predicted_labels = F.softmax(predicted_interaction,1).to('cpu').data.numpy()
      
            return correct_labels, predicted_labels
        
        elif method=="predict":
            with torch.no_grad():
                compound, adj, protein = inputs
                predicted_interaction = self.forward(compound, adj, protein)
                predicted_labels = F.softmax(predicted_interaction,1).to('cpu').data.numpy()
         
            return predicted_labels
            

class Trainer(object):
    def __init__(self, model, lr, weight_decay, batch):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr, weight_decay=weight_decay)
        self.batch = batch

    def train(self, dataset, device):
        self.model.train()
        N = len(dataset)
        loss_total = 0
        i = 0
        scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        self.optimizer.zero_grad()
        for compounds, adjacencies, proteins, correct_interaction in tqdm(dataset, desc="Training progress"):
            i = i+1
            # compounds = compounds.to(device)
            # adjacencies = adjacencies.to(device)
            # proteins = proteins.to(device)
            correct_interaction = correct_interaction.to(device)
            
            inputs = (compounds,adjacencies,proteins)
            loss = self.model(inputs, correct_interaction)
            
            loss = loss / self.batch
            loss.backward()
            if i % self.batch  == 0 or  i == N:
                self.optimizer.step()
                self.optimizer.zero_grad()
            loss_total += loss.item()
            # torch.cuda.empty_cache()

        # torch.cuda.empty_cache()
        return loss_total

class Tester(object):
    def __init__(self, model,method):
        self.model = model
        self.method = method

    def test(self, dataset,device="cpu"):
        self.model.eval()
        N = len(dataset)
        train = True
        if self.method=="train":
            y_true,y_pred, S = [], [], []
            with torch.no_grad():
                for compounds, adjacencies, proteins, correct_interaction in tqdm(dataset, desc="Test progress"):
                    # compounds = compounds.to(device)
                    # adjacencies = adjacencies.to(device)
                    # proteins = proteins.to(device)
                    # correct_interaction = correct_interaction.to(device)
                    inputs = (compounds,adjacencies,proteins)
                    correct_labels, predicted_labels = self.model(inputs, correct_interaction,method="evaluation")
                    y_true.extend(correct_labels)
                    y_pred.extend(predicted_labels)

            f1_type = "binary"
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            acc = accuracy_score(y_true, y_pred.argmax(1))
            f1 = f1_score(y_true, y_pred.argmax(1), average=f1_type)
            p = precision_score(y_true, y_pred.argmax(1), average=f1_type)
            r = recall_score(y_true, y_pred.argmax(1), average=f1_type)
            # auc = roc_auc_score(y_true, y_pred[:, 1]) if args.output_size == 2 else f1

            AUC=acc
            precision=p
            recall = r
            return AUC,f1, precision, recall
        else:
            y_pred = []
            with torch.no_grad():
                for compounds, adjacencies, proteins, correct_interaction in tqdm(dataset, desc="Test progress"):
                    inputs = (compounds,adjacencies,proteins)
                    predicted_labels = self.model(inputs, correct_interaction,self.method)
                    y_pred.extend(predicted_labels)
            
            return y_pred

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)
        





