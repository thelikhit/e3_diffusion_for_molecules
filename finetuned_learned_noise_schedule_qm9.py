# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import copy
import utils
import wandb
from configs.datasets_config import get_dataset_info
from os.path import join
from qm9 import dataset
from qm9.models import get_optim, get_model
from equivariant_diffusion import en_diffusion
from equivariant_diffusion.utils import assert_correctly_masked
from equivariant_diffusion import utils as flow_utils
import torch
import time
import pickle
from qm9.utils import prepare_context, compute_mean_mad
from train_test import train_epoch, test, analyze_and_save
import numpy as np

# Note: Change args from within the file.

npy_file = 'outputs/edm_qm9/generative_model_ema.npy'
pickle_file = 'outputs/edm_qm9/args.pickle'

# load args from pickle file
with open(pickle_file, 'rb') as f:
    args = pickle.load(f)

# set hyper parameters from default values
# DO NOT CHANGE ARCHITECTURE 
args.exp_name='finetuned_learned_noise_schedule'
args.n_epochs=10
args.test_epochs=2
args.batch_size=64

# learned noise schedule
args.diffusion_noise_schedule = 'learned'
args.diffusion_loss_type = 'vlb'

dataset_info = get_dataset_info(args.dataset, args.remove_h)
atom_encoder = dataset_info['atom_encoder']
atom_decoder = dataset_info['atom_decoder']

# device args
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32

# CAREFUL with this -->
# not sure what this does
if not hasattr(args, 'normalization_factor'):
    args.normalization_factor = 1
if not hasattr(args, 'aggregation_method'):
    args.aggregation_method = 'sum'

# wandb args and config
args.wandb_usr = utils.get_wandb_username(args.wandb_usr)
if args.no_wandb:
    mode = 'disabled'
else:
    mode = 'online' if args.online else 'offline'
kwargs = {'entity': args.wandb_usr, 'name': args.exp_name, 'project': 'e3_diffusion', 'config': args,
          'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
wandb.init(**kwargs)
wandb.save('*.txt')

# Retrieve QM9 dataloaders
dataloaders, charge_scale = dataset.retrieve_dataloaders(args)

# no idea what these are
context_node_nf = 0
property_norms = None
args.context_node_nf = context_node_nf

print(args)

# Load model
loaded_model, nodes_dist, prop_dist = get_model(args, device, dataset_info, dataloaders['train'])
if prop_dist is not None:
    property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
    prop_dist.set_normalizer(property_norms)

# load and apply weights
loaded_model_state_dict = torch.load(npy_file, map_location=device)
loaded_model.load_state_dict(loaded_model_state_dict, strict=False)

# optimiser
optim = get_optim(args, loaded_model)
loaded_model.to(device)

gradnorm_queue = utils.Queue()
gradnorm_queue.add(3000)  # Add large value that will be flushed.

def check_mask_correct(variables, node_mask):
    for variable in variables:
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)

# Initialize dataparallel if enabled and possible.
if args.dp and torch.cuda.device_count() > 1:
    print(f'Training using {torch.cuda.device_count()} GPUs')
    model_dp = torch.nn.DataParallel(loaded_model.cpu())
    model_dp = model_dp.cuda()
else:
    model_dp = loaded_model

# Initialize model copy for exponential moving average of params.
if args.ema_decay > 0:
    model_ema = copy.deepcopy(loaded_model)
    ema = flow_utils.EMA(args.ema_decay)
    if args.dp and torch.cuda.device_count() > 1:
        model_ema_dp = torch.nn.DataParallel(model_ema)
    else:
        model_ema_dp = model_ema
else:
    ema = None
    model_ema = loaded_model
    model_ema_dp = model_dp

# TODO: change these values? 
best_nll_val = 1e8
best_nll_test = 1e8

# begin finetuning
for epoch in range(0, args.n_epochs):
    start_epoch = time.time()
    train_epoch(args=args, loader=dataloaders['train'], epoch=epoch, model=loaded_model, model_dp=model_dp,
                model_ema=model_ema, ema=ema, device=device, dtype=dtype, property_norms=property_norms,
                nodes_dist=nodes_dist, dataset_info=dataset_info,
                gradnorm_queue=gradnorm_queue, optim=optim, prop_dist=prop_dist)
    print(f"Epoch took {time.time() - start_epoch:.1f} seconds.")
    if epoch % args.test_epochs == 0:
        if isinstance(loaded_model, en_diffusion.EnVariationalDiffusion):
            wandb.log(loaded_model.log_info(), commit=True)
        if not args.break_train_epoch:
            analyze_and_save(args=args, epoch=epoch, model_sample=model_ema, nodes_dist=nodes_dist,
                             dataset_info=dataset_info, device=device,
                             prop_dist=prop_dist, n_samples=args.n_stability_samples)
        nll_val = test(args=args, loader=dataloaders['valid'], epoch=epoch, eval_model=model_ema_dp,
                       partition='Val', device=device, dtype=dtype, nodes_dist=nodes_dist,
                       property_norms=property_norms)
        nll_test = test(args=args, loader=dataloaders['test'], epoch=epoch, eval_model=model_ema_dp,
                        partition='Test', device=device, dtype=dtype,
                        nodes_dist=nodes_dist, property_norms=property_norms)
        if nll_val < best_nll_val:
            best_nll_val = nll_val
            best_nll_test = nll_test
            if args.save_model:
                args.current_epoch = epoch + 1
                utils.save_model(optim, 'outputs/%s/finetuned_optim.npy' % args.exp_name)
                utils.save_model(loaded_model, 'outputs/%s/finetuned_generative_model.npy' % args.exp_name)
                if args.ema_decay > 0:
                    utils.save_model(model_ema, 'outputs/%s/finetuned_generative_model_ema.npy' % args.exp_name)
                with open('outputs/%s/finetuned_args.pickle' % args.exp_name, 'wb') as f:
                    pickle.dump(args, f)
            if args.save_model:
                utils.save_model(optim, 'outputs/%s/finetuned_optim_%d.npy' % (args.exp_name, epoch))
                utils.save_model(loaded_model, 'outputs/%s/finetuned_generative_model_%d.npy' % (args.exp_name, epoch))
                if args.ema_decay > 0:
                    utils.save_model(model_ema, 'outputs/%s/finetuned_generative_model_ema_%d.npy' % (args.exp_name, epoch))
                with open('outputs/%s/finetuned_args_%d.pickle' % (args.exp_name, epoch), 'wb') as f:
                    pickle.dump(args, f)
        print('Val loss: %.4f \t Test loss:  %.4f' % (nll_val, nll_test))
        print('Best val loss: %.4f \t Best test loss:  %.4f' % (best_nll_val, best_nll_test))
        wandb.log({"Val loss ": nll_val}, commit=True)
        wandb.log({"Test loss ": nll_test}, commit=True)
        wandb.log({"Best cross-validated test loss ": best_nll_test}, commit=True)

