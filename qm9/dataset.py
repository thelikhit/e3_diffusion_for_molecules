from torch.utils.data import DataLoader, Subset, ConcatDataset
from qm9.data.args import init_argparse
from qm9.data.collate import PreprocessQM9
from qm9.data.utils import initialize_datasets
import os
import copy
import torch
import numpy as np
import random

# deterministic behavior
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def retrieve_dataloaders(cfg):
    if 'qm9' in cfg.dataset:
        batch_size = cfg.batch_size
        num_workers = cfg.num_workers
        filter_n_atoms = cfg.filter_n_atoms
        # Initialize dataloader
        args = init_argparse('qm9')
        # data_dir = cfg.data_root_dir
        args, datasets, num_species, charge_scale = initialize_datasets(args, cfg.datadir, cfg.dataset,
                                                                        subtract_thermo=args.subtract_thermo,
                                                                        force_download=args.force_download,
                                                                        remove_h=cfg.remove_h)
        qm9_to_eV = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114, 'zpve': 27211.4, 'gap': 27.2114, 'homo': 27.2114,
                     'lumo': 27.2114}

        for dataset in datasets.values():
            dataset.convert_units(qm9_to_eV)

        combine_filtered_n_atoms=False
        if combine_filtered_n_atoms:
            datasets_12 = filter_atoms_(datasets, 12)
            datasets_25 = filter_atoms_(datasets, 25)

            datasets_12['train'] = reduce_dataset_(datasets_12, 'train', 500)
            datasets_12['valid'] = reduce_dataset_(datasets_12, 'valid', 50)
            datasets_12['test'] = reduce_dataset_(datasets_12, 'test', 50)

            datasets_25['train'] = reduce_dataset_(datasets_25, 'train', 500)
            datasets_25['valid'] = reduce_dataset_(datasets_25, 'valid', 50)
            datasets_25['test'] = reduce_dataset_(datasets_25, 'test', 50)
            datasets = {}
            for split in ['train', 'valid', 'test']:
                datasets[split] = ConcatDataset([
                    datasets_12[split],
                    datasets_25[split]
                ])
        else:
            if filter_n_atoms is not None:
                print("Retrieving molecules with only %d atoms" % filter_n_atoms)
                datasets = filter_atoms(datasets, filter_n_atoms)
            reduce_dataset(datasets, 'train', 1000)
            reduce_dataset(datasets, 'valid', 100)
            reduce_dataset(datasets, 'test', 100)

        
        sample_single_molecule=False    
        if sample_single_molecule:
            if 'train' in datasets:
                train = datasets['train']
                idx = torch.randint(0, len(train), (1,)).repeat(batch_size)
                subset = Subset(train, idx)
                for split in ['train', 'valid', 'test']:
                    datasets[split] = subset

        dump_dataset_to_file=True
        if dump_dataset_to_file:
            output_file = "dataset_dump.txt"

            with open(output_file, "w") as f:
                for split in ['train', 'valid', 'test']:
                    f.write(f"=== {split.upper()} DATASET ===\n")
                    f.write(f"Total samples: {len(datasets[split])}\n\n")

                    for i, sample in enumerate(datasets[split]):
                        f.write(f"--- {split.upper()} Sample {i} ---\n")
                        f.write(str(sample))
                        f.write("\n\n")
                        f.write("\n" + "=" * 60 + "\n\n")
        
        # Construct PyTorch dataloaders from datasets
        preprocess = PreprocessQM9(load_charges=cfg.include_charges)
        dataloaders = {split: DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=args.shuffle if (split == 'train') else False,
                                         num_workers=num_workers,
                                         collate_fn=preprocess.collate_fn)
                             for split, dataset in datasets.items()}
    elif 'geom' in cfg.dataset:
        import build_geom_dataset
        from configs.datasets_config import get_dataset_info
        data_file = './data/geom/geom_drugs_30.npy'
        dataset_info = get_dataset_info(cfg.dataset, cfg.remove_h)

        # Retrieve QM9 dataloaders
        split_data = build_geom_dataset.load_split_data(data_file,
                                                        val_proportion=0.1,
                                                        test_proportion=0.1,
                                                        filter_size=cfg.filter_molecule_size)
        transform = build_geom_dataset.GeomDrugsTransform(dataset_info,
                                                          cfg.include_charges,
                                                          cfg.device,
                                                          cfg.sequential)
        dataloaders = {}
        for key, data_list in zip(['train', 'val', 'test'], split_data):
            dataset = build_geom_dataset.GeomDrugsDataset(data_list,
                                                          transform=transform)
            shuffle = (key == 'train') and not cfg.sequential

            # Sequential dataloading disabled for now.
            dataloaders[key] = build_geom_dataset.GeomDrugsDataLoader(
                sequential=cfg.sequential, dataset=dataset,
                batch_size=cfg.batch_size,
                shuffle=shuffle)
        del split_data
        charge_scale = None
    else:
        raise ValueError(f'Unknown dataset {cfg.dataset}')

    return dataloaders, charge_scale


def combine_and_retrieve_dataloaders(cfg_qm9, cfg_drugs):
    raise NotImplementedError
    


def filter_atoms(datasets, n_nodes):
    for key in datasets:
        dataset = datasets[key]
        idxs = dataset.data['num_atoms'] == n_nodes
        for key2 in dataset.data:
            dataset.data[key2] = dataset.data[key2][idxs]

        datasets[key].num_pts = dataset.data['one_hot'].size(0)
        datasets[key].perm = None
    return datasets

def filter_atoms_(datasets, n_nodes):
    filtered_datasets = copy.deepcopy(datasets)    
    for key in filtered_datasets:
        filtered_dataset = filtered_datasets[key]
        idxs = filtered_dataset.data['num_atoms'] == n_nodes
        for key2 in filtered_dataset.data:
            filtered_dataset.data[key2] = filtered_dataset.data[key2][idxs]
        filtered_datasets[key].num_pts = filtered_dataset.data['one_hot'].size(0)
        filtered_datasets[key].perm = None
    return filtered_datasets

def reduce_dataset(datasets, name, n_samples):
    if name not in datasets:
        print(f"Warning: '{name}' dataset not found.")
        return
    dataset = datasets[name]
    n = min(len(dataset), n_samples)
    indices = torch.randperm(len(dataset))[:n]
    datasets[name] = Subset(dataset, indices)
    return datasets[name]


def reduce_dataset_(input_dataset, name, n_samples):
    input_dataset_split = input_dataset[name]
    n = min(len(input_dataset_split), n_samples)
    gen = torch.Generator().manual_seed(int(seed))
    indices = torch.randperm(len(input_dataset_split), generator=gen)[:n]
    reduced_dataset = Subset(copy.deepcopy(input_dataset_split), indices)
    return reduced_dataset
