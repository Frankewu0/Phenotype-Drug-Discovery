import os
import pandas as pd
import torch.utils.data as data
import torch
import numpy as np
from functools import partial
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from torch.utils.data import DataLoader
from .utils import integer_label_protein,graph_collate_func

class DTIDataset(data.Dataset):
    def __init__(self, list_IDs, df, max_drug_nodes=290):
        self.list_IDs = list_IDs
        self.df = df
        self.max_drug_nodes = max_drug_nodes

        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        index = self.list_IDs[index]
        v_d = self.df.iloc[index]['Smiles']
        # print(v_d)
        v_d = self.fc(smiles=v_d, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
        actual_node_feats = v_d.ndata.pop('h')
        num_actual_nodes = actual_node_feats.shape[0]
        num_virtual_nodes = self.max_drug_nodes - num_actual_nodes
        virtual_node_bit = torch.zeros([num_actual_nodes, 1])
        actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
        v_d.ndata['h'] = actual_node_feats
        virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 74), torch.ones(num_virtual_nodes, 1)), 1)
        v_d.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
        v_d = v_d.add_self_loop()

        v_p = self.df.iloc[index]['sequence']
        v_p = integer_label_protein(v_p)
        y = self.df.iloc[index]["label"]
        # y = torch.Tensor([y])
        return v_d, v_p, y


class MultiDataLoader(object):
    def __init__(self, dataloaders, n_batches):
        if n_batches <= 0:
            raise ValueError("n_batches should be > 0")
        self._dataloaders = dataloaders
        self._n_batches = np.maximum(1, n_batches)
        self._init_iterators()

    def _init_iterators(self):
        self._iterators = [iter(dl) for dl in self._dataloaders]

    def _get_nexts(self):
        def _get_next_dl_batch(di, dl):
            try:
                batch = next(dl)
            except StopIteration:
                new_dl = iter(self._dataloaders[di])
                self._iterators[di] = new_dl
                batch = next(new_dl)
            return batch

        return [_get_next_dl_batch(di, dl) for di, dl in enumerate(self._iterators)]

    def __iter__(self):
        for _ in range(self._n_batches):
            yield self._get_nexts()
        self._init_iterators()

    def __len__(self):
        return self._n_batches
    
def load_datasets(data_dir: str, test_file: str):
    train_df = pd.read_csv(os.path.join(data_dir, 'val.csv'))
    val_df = pd.read_csv(os.path.join(data_dir, 'val.csv'))
    test_df = pd.read_csv(test_file)
    return (
        DTIDataset(train_df.index.values, train_df),
        DTIDataset(val_df.index.values, val_df),
        DTIDataset(test_df.index.values, test_df),
    )

def create_dataloaders(cfg, train_set, val_set, test_set):
    params = {
        'batch_size': cfg.MODEL.BATCH_SIZE,
        'num_workers': cfg.MODEL.NUM_WORKERS,
        'collate_fn': graph_collate_func
    }
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **params)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **params)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=False, **params)
    return train_loader, val_loader, test_loader
