from torch.utils.data import DataLoader, Subset
from qm9.data.args import init_argparse
from qm9.data.collate import PreprocessQM9
from qm9.data.utils import initialize_datasets
import os
import torch


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

        if filter_n_atoms is not None:
            print("Retrieving molecules with only %d atoms" % filter_n_atoms)
            datasets = filter_atoms(datasets, filter_n_atoms)
            
            
# --- MODIFICATION START ---
        # Define the target number of samples for the datasets
        target_train_samples = 400
        target_val_samples = 50
        target_test_samples = 50

        # Function to reduce dataset size
        def reduce_dataset_size(dataset_name, target_samples):
            if dataset_name in datasets:
                current_dataset = datasets[dataset_name]
                if len(current_dataset) > target_samples:
                    indices = torch.randperm(len(current_dataset))[:target_samples]
                    datasets[dataset_name] = Subset(current_dataset, indices)
                    print(f"Reduced '{dataset_name}' dataset to {len(datasets[dataset_name])} samples.")
                else:
                    print(f"'{dataset_name}' dataset already has {len(current_dataset)} samples, no reduction needed.")
            else:
                print(f"Warning: '{dataset_name}' dataset not found in the initialized datasets.")

        reduce_dataset_size('train', target_train_samples)
        reduce_dataset_size('valid', target_val_samples)
        reduce_dataset_size('test', target_test_samples)

        # Datapoint Repetition ---
        print(f"\nChecking for datapoint repetition for datasets smaller than batch size ({batch_size})...")
        for split, dataset in datasets.items():
            current_len = len(dataset)
            if current_len > 0 and current_len < batch_size:
                # Calculate how many times the current dataset needs to be repeated
                # to reach at least 'batch_size' elements (using ceiling division)
                num_repeats = (batch_size + current_len - 1) // current_len

                # Create a list of indices that repeats the original dataset's indices
                repeated_indices = []
                for _ in range(num_repeats):
                    repeated_indices.extend(list(range(current_len)))

                # Create a new Subset using the repeated indices, truncated to exactly 'batch_size'
                # This ensures the DataLoader always gets 'batch_size' elements, even if repeated.
                datasets[split] = Subset(dataset, repeated_indices[:batch_size])
                print(f"[{split}] Dataset size ({current_len}) was less than batch size ({batch_size}). "
                      f"Repeated datapoints to reach {len(datasets[split])} for full batch processing.")
            elif current_len == 0 and batch_size > 0:
                # Warn if the dataset is empty but a positive batch size is expected
                print(f"Warning: [{split}] Dataset is empty, but batch_size is {batch_size}. "
                      "Cannot populate datapoints by repetition.")
# --- MODIFICATION END ---


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


def filter_atoms(datasets, n_nodes):
    for key in datasets:
        dataset = datasets[key]
        idxs = dataset.data['num_atoms'] == n_nodes
        for key2 in dataset.data:
            dataset.data[key2] = dataset.data[key2][idxs]

        datasets[key].num_pts = dataset.data['one_hot'].size(0)
        datasets[key].perm = None
    return datasets