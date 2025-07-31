import logging
from typing import Dict, Tuple, List
from functools import partial
import numpy as np
import pandas as pd
import torch
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------------------------
#  CellpaintingDataset
# -------------------------------------------------------------------------------------------------------------------
class CellpaintingDataset(torch.utils.data.IterableDataset ):

    def __init__(self,
                 type : str = None,
                 training_path : str = None,
                 validation_path : str = None,
                 test_path : str = None,
                 batch_size: int = 1,
                 sample_size : int = 3,
                 rows_per_batch : int  = None,
                 conversions : Dict = None,
                 train_start : int = None,
                 train_end : int = None,
                 val_start : int = None,
                 val_end : int = None,
                 test_start : int = None,
                 test_end : int = None,
                 names : List = None,
                 usecols : List = None,
                 iterator : bool = True,
                 verbose : bool = False,
                 compounds_per_batch : int = 1,
                 tpsa_threshold : int = 100,
                 **misc ):
        # print("Cellpainting __init__ routine", flush=True)
        # Store the filename in object's memory
        type = type.lower()
        assert (type in ['train', 'val', 'test']), f" type parm must be {{'train', 'val', or 'test'}}"
        self.type = type

        self.names = names
        self.dtype = conversions
        self.usecols = usecols
        self.compounds_per_batch = compounds_per_batch
        self.iterator = iterator
        # chunksize should be a mulitple of sample_size
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.tpsa_threshold = tpsa_threshold
        if rows_per_batch is None:
            self.rows_per_batch = self.sample_size  * self.compounds_per_batch

        if self.type == 'train': 
            self.filename = training_path
            self.start = train_start
            self.end = train_end
        elif self.type == 'val':
            self.filename = validation_path
            self.start = val_start
            self.end = val_end
        else: 
            self.filename = test_path
            self.start = test_start
            self.end = test_end
        self.numrows = self.end-self.start

        file_sz = self.end - self.start
        smp_sz = self.sample_size
        cpb_sz = self.compounds_per_batch
        bth_sz = self.batch_size
        recs_per_batch = smp_sz * bth_sz * cpb_sz
        bth_per_epoch = file_sz // recs_per_batch

        logger.info(f" Building CellPantingDataset for {self.type}")
        logger.info(f" filename:  {self.filename}")
        logger.info(f" type    :  {self.type}")
        logger.info(f" start   :  {self.start}")
        logger.info(f" end     :  {self.end}")
        logger.info(f" numrows :  {self.numrows}")
        logger.info(f" names   :  {self.names}     usecols :  {self.usecols}")
        logger.info(f" batch_size  :  {self.batch_size}")
        logger.info(f" sample_size :  {self.sample_size}")
        logger.info(f" compounds_per_batch :  {self.compounds_per_batch}")
        logger.info(f" rows per batch (chunksize) :  {self.rows_per_batch}")
        logger.info(f" TPSA threshold :  {self.tpsa_threshold}")
        logger.info(f" Each mini-batch contains {recs_per_batch/smp_sz} compounds with {smp_sz} samples per compound : total {recs_per_batch} rows")
        logger.info(f" Number of {recs_per_batch} row full size batches per epoch: {bth_per_epoch}")
        logger.info(f" Rows covered by {bth_per_epoch} full size batches ({recs_per_batch} rows) per epoch:  {(file_sz // recs_per_batch) * recs_per_batch}")
        logger.info(f" Last partial batch contains : {file_sz % recs_per_batch} rows")
        logger.info(f" ")

    # self.group_labels = np.arange(self.batch_size * self.sample_size, dtype = np.int64) // self.sample_size
        # print(f" Dataset batch_size: {self.batch_size}" )
        # print(self.group_labels)
        # And that's it, we no longer need to store the contents in the memory

    def preprocess(self,
                   text):
        ### Do something with data here
        print(f" Running preprocess data \n")
        text_pp = text.lower().strip()
        print(len(text_pp))
        ###
        return text_pp

    def line_mapper(self,
                    line):
        # print(f" Running line_mapper \n")
        # Splits the line into text and label and applies preprocessing to the text
        # data  = line.to_numpy()
        # text = self.preprocess(text)
        # print(f" compound: {compound.shape}  -  {data}")
        return line

    def __iter__(self):
        # Create an iterator
        # print("Cellpainting __iter__ routine", flush=True)
        self.file_iterator = pd.read_csv(self.filename,
                                         names = self.names,
                                         header = 0,
                                         skiprows = self.start,
                                         nrows = self.numrows,
                                         dtype = self.dtype,
                                         usecols = self.usecols,
                                         iterator = self.iterator,
                                         chunksize = self.rows_per_batch,)
        # print(type(self.file_iterator), self.numrows, self.chunksize)
        # df_ps = dd.read_csv(profile_file, header=header, names = names, usecols = usecols, dtype = dtype) 
        # Map each element using the line_mapper
        # self.mapped_itr = map(self.line_mapper, self.file_iterator)
        return self.file_iterator

    # def __next__(self):
        # print("Cellpainting __next__ funtion called")
        # super().__next__()


    def __len__(self):
        # print("Cellpainting __len__ funtion called")
        return self.numrows

# -------------------------------------------------------------------------------------------------------------------
#  InfiniteDataLoader
# -------------------------------------------------------------------------------------------------------------------

class InfiniteDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f" Dataset size: {len(self.dataset)}   rows per batch: {self.dataset.rows_per_batch} ")
        # Initialize an iterator over the dataset.

    def __iter__(self):
        self.dataset_iterator = super().__iter__()
        # print(f"InfiniteDataLoader __iter__  routine   {type(self.dataset_iterator)}")
        return self

    def __next__(self):
        try:
            # print(f"InfiniteDataLoader __next__  routine ")
            batch = next(self.dataset_iterator)
            # print(type(batch), len(batch), type(batch[0]), batch[0].shape)
        except StopIteration:
            # print("InfiniteDataLoader __next__  routine  -- End of dataset encountered!!!")
            # Dataset exhausted, use a new fresh iterator.
            # self.dataset_iterator = super().__iter__()
            # batch = next(self.dataset_iterator)
            raise StopIteration
        else:
            return batch


def custom_collate_fn(batch):
    batch_numpy = np.concatenate(batch)
    plates  = batch_numpy[:,:4]
    compounds = batch_numpy[:,4]
    cmphash = batch_numpy[:,5:7].astype(np.int64)
    tpsa = batch_numpy[:,7:10]
    labels = torch.from_numpy(batch_numpy[:,10:11].astype(np.float32))
    data = torch.from_numpy(batch_numpy[:, 11:].astype(np.float32))

    return data, labels, plates, compounds, cmphash, tpsa

def dynamic_collate_fn(batch, tpsa_threshold : int ):
    batch_numpy = np.concatenate(batch)
    plates  = batch_numpy[:,:4]
    compounds = batch_numpy[:,4]
    cmphash = batch_numpy[:,5:7].astype(np.int64)
    tpsa = batch_numpy[:,7:10]
    orig_labels = torch.from_numpy(batch_numpy[:,10:11].astype(np.float32))
    labels = np.where(batch_numpy[:,7:8] > tpsa_threshold, 1.0, 0.0)
    labels = torch.from_numpy(labels.astype(np.float32))
    data = torch.from_numpy(batch_numpy[:, 11:].astype(np.float32))

    return data, labels, plates, compounds, cmphash, tpsa, orig_labels

 
def pharmacophore_collate_fn(batch, return_bits: list):
    batch_numpy = np.concatenate(batch)
    print(f" batch shape: {len(batch)}   batch_numpy: {batch_numpy.shape}")
    for i in batch:
        print(f" type: {type(i)} shape: {i.shape}")
    compounds = batch_numpy[:,0]
    cmphash = batch_numpy[:,1:3].astype(np.int64)
    print(return_bits)
    print(batch_numpy[:10, 3:40])
    bits = batch_numpy[:,return_bits+3]
    print(bits[:10])
    
    return compounds, cmphash, bits

# -------------------------------------------------------------------------------------------------------------------
#  Pharmacophore Dataset
# -------------------------------------------------------------------------------------------------------------------
class PharmacophoreDataset(torch.utils.data.IterableDataset ):

    def __init__(self,
                 type : str = None,
                 training_path : str = None,
                 validation_path : str = None,
                 test_path : str = None,
                 batch_size: int = 1,
                 sample_size : int = 3,
                 rows_per_batch : int  = None,
                 conversions : Dict = None,
                 train_start : int = None,
                 train_end : int = None,
                 val_start : int = None,
                 val_end : int = None,
                 test_start : int = None,
                 test_end : int = None,
                 names : List = None,
                 usecols : List = None,
                 iterator : bool = True,
                 verbose : bool = False,
                 compounds_per_batch : int = 1,
                 return_bits = None,
                 **misc ):
        # print("Cellpainting __init__ routine", flush=True)
        # Store the filename in object's memory
        type = type.lower()
        assert (type in ['train', 'val', 'test']), f" type parm must be {{'train', 'val', or 'test'}}"
        self.type = type

        self.names = names
        self.dtype = conversions
        self.usecols = usecols
        self.compounds_per_batch = compounds_per_batch
        self.iterator = iterator
        # chunksize should be a mulitple of sample_size
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.return_bits = np.array(sorted(return_bits))
        if rows_per_batch is None:
            self.rows_per_batch = self.sample_size  * self.compounds_per_batch

        if self.type == 'train': 
            self.filename = training_path
            self.start = train_start
            self.end = train_end
        elif self.type == 'val':
            self.filename = validation_path
            self.start = val_start
            self.end = val_end
        else: 
            self.filename = test_path
            self.start = test_start
            self.end = test_end
        self.numrows = self.end-self.start

        file_sz = self.end - self.start
        smp_sz = self.sample_size
        cpb_sz = self.compounds_per_batch
        bth_sz = self.batch_size
        recs_per_batch = smp_sz * bth_sz * cpb_sz
        bth_per_epoch = file_sz // recs_per_batch

        logger.info(f" Building Pharmacophore Dataset for {self.type}")
        logger.info(f" filename:  {self.filename}")
        logger.info(f" type    :  {self.type}")
        logger.info(f" start   :  {self.start}")
        logger.info(f" end     :  {self.end}")
        logger.info(f" numrows :  {self.numrows}")
        logger.info(f" names   :  {self.names}     usecols :  {self.usecols}")
        logger.info(f" batch_size  :  {self.batch_size}")
        logger.info(f" sample_size :  {self.sample_size}")
        logger.info(f" compounds_per_batch :  {self.compounds_per_batch}")
        logger.info(f" rows per batch (chunksize) :  {self.rows_per_batch}")
        logger.info(f" Return Bits :  {self.return_bits}")
        logger.info(f" Each mini-batch contains {recs_per_batch/smp_sz} compounds with {smp_sz} samples per compound : total {recs_per_batch} rows")
        logger.info(f" Number of {recs_per_batch} row full size batches per epoch: {bth_per_epoch}")
        logger.info(f" Rows covered by {bth_per_epoch} full size batches ({recs_per_batch} rows) per epoch:  {(file_sz // recs_per_batch) * recs_per_batch}")
        logger.info(f" Last partial batch contains : {file_sz % recs_per_batch} rows")
        logger.info(f" ")

    # self.group_labels = np.arange(self.batch_size * self.sample_size, dtype = np.int64) // self.sample_size
        # print(f" Dataset batch_size: {self.batch_size}" )
        # print(self.group_labels)
        # And that's it, we no longer need to store the contents in the memory

    def preprocess(self,
                   text):
        ### Do something with data here
        print(f" Running preprocess data \n")
        text_pp = text.lower().strip()
        print(len(text_pp))
        ###
        return text_pp

    def line_mapper(self,
                    line):
        # print(f" Running line_mapper \n")
        # Splits the line into text and label and applies preprocessing to the text
        # data  = line.to_numpy()
        # text = self.preprocess(text)
        # print(f" compound: {compound.shape}  -  {data}")
        return line

    def __iter__(self):
        # Create an iterator
        # print("Cellpainting __iter__ routine", flush=True)
        self.file_iterator = pd.read_csv(self.filename,
                                         names = self.names,
                                         header = 0,
                                         skiprows = self.start,
                                         nrows = self.numrows,
                                         dtype = self.dtype,
                                         usecols = self.usecols,
                                         iterator = self.iterator,
                                         chunksize = self.rows_per_batch,)
        return self.file_iterator


    def __len__(self):
        # print("Cellpainting __len__ funtion called")
        return self.numrows


