import os
import os.path as osp
import numpy as np
import json

import torch
import torch_geometric.data as tgd

from ase.db import connect
from tqdm.notebook import tqdm
from kaggle.api.kaggle_api_extended import KaggleApi

from processing import get_data

class MOSESDataset(tgd.InMemoryDataset):
    def __init__(self, competition_name, root, mode, val_size=0.2, transform=None, pre_transform=None, pre_filter=None, use_kaggle_api=True):
        self.use_kaggle_api = use_kaggle_api
        self.competition_name = competition_name
        
        self.val_size = val_size
        if val_size > 0:
            self.load_idx = {'train':0, 'val':1, 'test':2}
        else:
            self.load_idx = {'train':0, 'test':1}

        self.modes = list(self.load_idx.keys())

        if mode not in self.modes:
            raise ValueError(f'{mode} mode not in {self.modes}')
        self.mode = mode
        
        if self.use_kaggle_api:
            self.kaggle_api = KaggleApi()
            self.kaggle_api.authenticate()
        
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.load_idx[self.mode]])
        

    @property
    def raw_file_names(self):
        if self.use_kaggle_api:
            return [str(f) for f in self.kaggle_api.competition_list_files(self.competition_name)]
        else:
            return ['ase_database_example.py', 'sample_submission.csv', 'train.db', 'test.db']

    @property
    def processed_file_names(self):
        modes_data = [f'{mode}_data.pt' for mode in self.modes]
        return modes_data + ['unique_symbols.json']

    def download(self):
        self.kaggle_api.competition_download_files(self.competition_name, path=self.root)

        archive_path = osp.join(self.root, self.competition_name + '.zip')
        tgd.extract_zip(archive_path, self.raw_dir, log=False)
        os.remove(archive_path)

    def process(self):
        train_val_database_path = osp.join(self.raw_dir, f'train.db')
        test_database_path = osp.join(self.raw_dir, f'test.db')
        unique_symbols_name = 'unique_symbols.json'
       
        train_val_database = connect(train_val_database_path)
        train_val_size = train_val_database.count()
        
        test_database = connect(test_database_path)
        test_size = test_database.count()

        unique_symbols_path = osp.join(self.processed_dir, unique_symbols_name)
        if osp.exists(unique_symbols_path):
            with open(unique_symbols_path, 'r') as f:
                unique_symbols = json.load(f)
        else:
            unique_symbols = set()
            for row in tqdm(train_val_database.select(), total=train_val_database.count()):
                unique_symbols.update(np.unique(row.symbols))
            unique_symbols = list(unique_symbols)

            with open(unique_symbols_path, 'w') as f:
                json.dump(unique_symbols, f)

        # train/val split
        val_size = self.val_size
        val_idxs = np.random.choice(np.arange(train_val_size), size=int(val_size*train_val_size))
        
        mode_list = {
            'train': [],
            'val': [],
            'test': []
        }

        for idx, row in tqdm(enumerate(train_val_database.select()), total=train_val_size):
            data = get_data(row, unique_symbols)
            
            if idx in val_idxs:
                mode_list['val'].append(data)
            else:
                mode_list['train'].append(data)
                
        for idx, row in tqdm(enumerate(test_database.select()), total=test_size):
            data = get_data(row, unique_symbols)
            mode_list['test'].append(data)
           
        
        for mode in self.modes:
            if self.pre_filter is not None:
                mode_list[mode] = [data for data in mode_list[mode] if self.pre_filter(data)]

            if self.pre_transform is not None:
                mode_list[mode] = [self.pre_transform(data) for data in mode_list[mode]]
                
            data, slices = self.collate(mode_list[mode])
            torch.save((data, slices), self.processed_paths[self.load_idx[mode]])
            