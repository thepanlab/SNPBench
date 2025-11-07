"""
Training

Usage examples:

# simple:
$ python3 train.py --model_dir ./path_to_dir_with/config.yml

# use nohup to run in background:
$ python3 train.py --model_dir ./path_to_dir_with_config > train.log 2>&1 &

# specify GPU device and run in background:
$ CUDA_VISIBLE_DEVICES=0 nohup python3 -u train.py --model_dir ./path_to_dir_with_config > train.log 2>&1 &
"""

import argparse, logging, json, os, warnings
from contextlib import nullcontext
import numpy as np
import pandas as pd
import torch

from ukb_data import UKBData
from model.datasets import DaskIterableDataset
import model.nnet as net
import model.metrics as metrics
from evaluate import evaluate
import model.transforms as transforms
import utils

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

device=("cuda" if torch.cuda.is_available() else "cpu")

# argparser
def create_parser():
    parser = argparse.ArgumentParser(description='UKB DNN Training', 
                                     fromfile_prefix_chars='@')
    parser.add_argument(
        '--model_dir', type=str, required=True, default=None, 
        help="(str) Path to folder containing config.yaml and the " + \
             "location where results will be saved."
    )
    return parser

def build_dataloader(X, y, batch_size, shuffle=False, drop_last=False, imputers=None,
                     transform_x=None, transform_y=None, use_decoys=False, 
                     decoy_seed=42, num_workers=0):
    ds = DaskIterableDataset(
        X=X, y=y, batch_size=batch_size, shuffle=shuffle,
        drop_last=drop_last, imputers=imputers, 
        transform_x=transform_x, transform_y=transform_y,
        use_decoys=use_decoys, decoy_seed=decoy_seed, 
    )
    return torch.utils.data.DataLoader(
        ds, 
        batch_size=None, # iterable dataset already yields (Xb, yb)
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
        # collate_fn=lambda x: x[0] # already batched in dataset
    )

def show_epoch_metrics(metrics, split_name='', print_fn=None, precision=4):
    if print_fn is None:
        print_fn = logging.info  # consistent logging with rest of training pipeline

    if not metrics:
        line = f"{split_name} -- (no metrics)"
    else:
        keys = ['loss'] + sorted(k for k in metrics.keys() if k != 'loss')
        metrics_str = ", ".join([f"{k}: {metrics[k]:.{precision}f}" for k in keys if k in metrics])
        line = f"{split_name} -- {metrics_str}"
    print_fn(line)

def _is_improvement(current: float, best: float, min_delta: float, mode: str) -> bool:
    """Return True if current is an improvement over best by at least min_delta."""
    if mode == 'max':
        return (current - best) > min_delta
    elif mode == 'min':
        return (best - current) > min_delta
    else:
        raise ValueError(f"early_stopping.mode must be 'max' or 'min', got {mode}")

def train(
    model, optimizer, dataloader, loss_fn, metrics_dict,
    l1_lambda=0.0, save_summary_steps=20, grad_clip=None,
    grad_scaler=None, pred_transform=None,
    amp_enabled=False, amp_dtype=None):
    """
    Train model for one epoch over dataloader.

    grad_clip (float|None): If set, applies norm-based gradient clipping.
    grad_scaler (torch.cuda.amp.GradScaler|None): If provided, enables
        loss scaling (typical for FP16). For BF16 & FP32, keep as None.
    pred_transform (callable|None): Applied to outputs only for metrics
        (e.g., sigmoid/softmax); the loss is always computed on raw outputs.
    amp_enabled (bool): If True, forward pass is wrapped in autocast for 
        mixed precision.
    amp_dtype (torch.dtype|None): AMP compute dtype (e.g., torch.float16 or 
         torch.bfloat16).
    """
    model.train() # train mode
    dev = next(model.parameters()).device
    
    # Precision/AMP mode flags
    # use_fp16: true mixed-precision (FP16) with GradScaler
    # use_bf16: autocast in BF16; no GradScaler needed
    use_fp16 = bool(amp_enabled and amp_dtype is torch.float16 and grad_scaler is not None)
    # use_bf16 = bool(amp_enabled and amp_dtype is torch.bfloat16 and grad_scaler is None)

    if amp_enabled and amp_dtype is None:
        # if amp is requested but dtype not set, default to bf16
        amp_dtype = torch.bfloat16

    loss_avg = utils.RunningAverage() # track mean loss over epoch
    summary = [] # track metrics
    # iterate minibatches, one optimizer step per minibatch
    for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
        x_batch = x_batch.to(dev, non_blocking=True)
        y_batch = y_batch.to(dev, non_blocking=True)
        # clear gradients
        optimizer.zero_grad(set_to_none=True)
        
        # autocast context (no-op/nullcontext if amp_enabled=False)
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled)
            if torch.cuda.is_available() else nullcontext()
        )
        with autocast_ctx:
            outputs = model(x_batch) # forward pass in AMP
            loss = loss_fn(outputs, y_batch) # compute loss
        
        # L1 regularization (optional)
        if l1_lambda > 0.0:
            # keep L1 penalty in fp32 (for stability)
            with torch.autocast(device_type="cuda", enabled=False):
                l1_penalty = sum(
                    p.float().abs().sum() 
                    for name, p in model.named_parameters()
                    if p.requires_grad and "bias" not in name
                )
            loss = loss + l1_lambda * l1_penalty
        
        # backprop and optimizer step (optional grad clipping)
        if use_fp16:
            # fp16 mixed precision with GradScaler
            grad_scaler.scale(loss).backward()
            if grad_clip is not None and grad_clip > 0:
                # unscale before clipping so the threshold applies 
                # to the true gradient magnitudes
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            # step the optimizer and update the scaler
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            # bf16 AMP (no scaler) or pure FP32
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
        
        # update running loss average
        loss_avg.update(float(loss.item()))
        # store metrics every so often
        if batch_idx % save_summary_steps == 0:
            # detach preds
            y_pred = outputs.detach()
            # transform preds (e.g., sigmoid/softmax) only for metrics
            if pred_transform is not None:
                y_pred = pred_transform(y_pred)
            batch_summary = metrics.compute_metrics(
                metrics_dict,
                y_pred.to(torch.float32).cpu().numpy(),
                y_batch.detach().to(torch.float32).cpu().numpy(),
            )
            summary.append(batch_summary)
        # Average metrics across recorded minibatches
        if summary:
            metrics_avg = {m: float(np.mean([x[m] for x in summary])) for m in summary[0]}
        else:
            metrics_avg = {}
        # include mean loss over minibatches
        metrics_avg["loss"] = float(loss_avg())
        return metrics_avg


# Main entry point
if __name__ == "__main__":
    # Parse args
    parser = create_parser()
    args=parser.parse_args()
    model_dir = os.path.abspath(args.model_dir)
    assert os.path.isdir(model_dir), f"Model directory not found: {model_dir}"
    cfg_path = os.path.join(model_dir, 'config.yaml')
    assert os.path.isfile(cfg_path), f"Config file not found: {cfg_path}"
    
    # Start a new log file
    utils.set_logger(os.path.join(model_dir, 'train.log'))
    logging.info('='*80)
    logging.info(f"UKB DNN Training")
    logging.info('='*80)
    logging.info(f"Conda env:   {os.environ.get('CONDA_DEFAULT_ENV', 'unknown')}")
    logging.info(f"Model dir:   {model_dir}")
    logging.info(f"Device:      {device}")
    logging.info('-'*80)
    
    # Load config
    cfg = utils.load_yaml(cfg_path)
    # define variables provided in the config file
    task_type = str(cfg.get('task_type', 'regression')).lower() # regression | binary_classification
    seed = int(cfg.get('seed', 42))
    missing_policy = cfg.get('missing_policy', None) # 'mean' | 'mode' | None
    optimizer_name = str(cfg.get('optimizer', 'adam')).lower()
    loss_func = str(cfg.get('loss_function', 'mse')).lower()
    lr = float(cfg.get('learning_rate', 1e-4))
    l2_lambda = float(cfg.get('l2_lambda', 0.0))
    l1_lambda = float(cfg.get('l1_lambda', 0.0))
    hidden_layers = cfg.get('hidden_layers', [1000, 200, 50])
    dropout_rate = cfg.get('dropout_rate', None)
    activation_fn = str(cfg.get('activation_function', 'relu')).lower()
    grad_clip = cfg.get('grad_clip', None) # e.g., 1.0
    batch_size = int(cfg.get('batch_size', 128))
    num_epochs = int(cfg.get('num_epochs', 100))
    save_summary_steps = int(cfg.get('save_summary_steps', 20))
    use_decoys = bool(cfg.get('use_decoys', False))
    decoy_seed = cfg.get('decoy_seed', None)
    num_workers = int(cfg.get('num_workers', 0))
    restore_file = cfg.get('restore_file', None) # e.g., 'best' or 'last'
    # monitor_metric = cfg.get('model', {}).get('monitor_metric')
    monitor_metric = str(cfg.get('monitor_metric', 'r2')).lower()
    # metrics_config_file = str(cfg.get('metrics_config_file', None)) # IRRELEVANT NOW (metrics config dict now built in metrics.py)
    # lr_scheduler_name = cfg.get('lr_scheduler', None) # reduce_on_plateau | cosine_annealing | None
    use_lr_scheduler = bool(cfg.get('lr_scheduler', False))
    model_name = cfg.get('model_name', 'vanillanet')# .lower()
    use_x_transforms = bool(cfg.get('use_x_transforms', True))  # only relevant for VanillaNet
    use_y_transforms = bool(cfg.get('use_y_transforms', False))
    include_dom = bool(cfg.get('include_dom', True))
    include_rec = bool(cfg.get('include_rec', True))
    # early stopping enabled or not
    es_enabled = bool(cfg.get('es_enabled', False))
    # n_epochs w/out improvement after which training will be stopped
    es_patience = int(cfg.get('es_patience', 10))
    # minimum change to qualify as an improvement
    es_min_delta = float(cfg.get('es_min_delta', 0.0))
    # start checking after these many epochs
    es_min_epochs = int(cfg.get('es_min_epochs', 0))
    es_mode = str(cfg.get('es_mode', 'max')).lower()
    
    # automatic mixed precision (AMP) training
    model_precision = str(cfg.get('precision', 'fp32')).lower()

    # torch compile params
    torch_compile_enabled = bool(cfg.get('torch_compile', False))
    torch_compile_mode = str(cfg.get('torch_compile_mode', 'default')).lower()
    
    # determine how to handle missing values
    if missing_policy is not None:
        missing_policy = missing_policy.lower()
    if dropout_rate is not None:
        dropout_rate = float(dropout_rate)
    if grad_clip is not None:
        grad_clip = float(grad_clip)
    if es_mode not in ('max', 'min'):
        raise ValueError(f"Invalid early_stopping.mode: {es_mode}")
    
    # set decoy seeds for train and val
    train_decoy_seed = None
    val_decoy_seed = None
    if use_decoys and decoy_seed is not None:
        decoy_seed = int(decoy_seed)
        train_decoy_seed = decoy_seed
        val_decoy_seed = decoy_seed + 1
    elif use_decoys and decoy_seed is None:
        raise ValueError("decoy_seed must be provided if use_decoys is True.")

    # set global seed
    utils.set_global_seed(seed)
    
    # display/log config items
    for k,v in cfg.items():
        logging.info(f"  {k}: {v}")
    logging.info('-'*80)

    # UKBData object (loads genotype and phenotype data)
    data = UKBData(cfg, verbose=True, print_func=logging.info)
    
    # NOTE: make sure optimize_chunks=True when training with decoys,
    # and make sure not to change rows_per_chunk in config file    
    train_data = data.get_partition(split='train', 
                                    optimize_chunks=True,
                                    verbose=True)
    val_data = data.get_partition(split='val', 
                                  optimize_chunks=True,
                                  verbose=True)
    
    X_train, y_train = train_data.X, train_data.y
    X_val, y_val = val_data.X, val_data.y
    
    logging.info('='*80)
    
    n_snps = X_train.shape[1]
    outs_units = 1 # if task_type == 'regression' else n_classes 
    
    # Missing policy (impute NaNs with mean, mode, or zeros)
    imputers = None
    if missing_policy is not None:
        if not missing_policy.lower() in ['mean', 'means', 'mode', 'modes']:
            raise ValueError(f"Unknown missing_policy: {missing_policy}")
        imputers = data.get_impute_array(split='train', policy=missing_policy.lower())
        logging.info(f"Missing genotype values will be imputed with {missing_policy.lower()}.")
    else:
        logging.info("Missing genotype values will be imputed with zeros.")

    # X transformations 
    x_transform = None
    # optional for VanillaNet; required for IndicatorGenoNet
    if (model_name.lower() in ['vanillanet', 'vanilla_net'] 
        or model_name.lower() in ['linearridge', 'linear_ridge']):
        ins_units = int(n_snps)
        # X data transformation (e.g., standardization)
        if use_x_transforms:
            minvar = 1e-6
            # compute mean & scale (from training set only)
            variant_mean, variant_scale = data.genotype_transforms_stats(
                split='train', min_var=minvar
            )
            # get a callable transformation object for X data 
            x_transform = transforms.SNPStandardizeTransform(
                variant_mean=variant_mean, 
                variant_scale=variant_scale, 
                eps=minvar, 
                do_unit_zscore=False
            )
            logging.info("VanillaNet: applying SNPStandardizeTransform to dosage features.")
        else:
            logging.info("Not applying X data transformation(s).")
    elif model_name.lower() in ['indicatorgenonet', 'indicator_geno_net']:
        logging.info("IndicatorGenoNet: computing indicator-channel stats from TRAIN only.")
        means, stds, M, C = transforms.compute_indicator_channel_stats_from_dask(
            X_train, include_dom=include_dom, include_rec=include_rec
        )
        x_transform = transforms.IndicatorChannelTransform(
            means, stds, n_snps=M, 
            include_dom=include_dom, 
            include_rec=include_rec
        )
        # set ins_units equal to n_snps * n_channels to account for indicator encoding
        ins_units = int(M * C)
        # save means, stds, and metadata for use during inference
        np.save(os.path.join(model_dir, 'indicator_means.npy'), means)
        np.save(os.path.join(model_dir, 'indicator_stds.npy'), stds)
        with open(os.path.join(model_dir, 'indicator_meta.txt'), 'w') as f:
            f.write(f"M={M}\nC={C}\ninclude_dom={include_dom}\ninclude_rec={include_rec}\n")
        
        logging.info(f"IndicatorGenoNet: using channels C={C} -> input_units={ins_units}.")
        logging.info(f"Saved indicator channel stats:")
        logging.info(f"  - Means: {os.path.join(model_dir, 'indicator_means.npy')}")
        logging.info(f"  - Stds:  {os.path.join(model_dir, 'indicator_stds.npy')}")
        logging.info(f"  - Meta:  {os.path.join(model_dir, 'indicator_meta.txt')}")
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    
    # get dataloaders
    dl_train = build_dataloader(
        X_train, y_train, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True, 
        imputers=imputers, 
        transform_x=x_transform, 
        transform_y=None, 
        use_decoys=use_decoys, 
        decoy_seed=train_decoy_seed, 
        num_workers=num_workers
    )
    dl_val = build_dataloader(
        X_val, y_val, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=False, 
        imputers=imputers, 
        transform_x=x_transform, 
        transform_y=None,
        use_decoys=use_decoys, 
        decoy_seed=val_decoy_seed, 
        num_workers=num_workers
    )
    
    # build model
    if model_name.lower() in ['vanillanet', 'vanilla_net']:
        ins_units = int(n_snps)
        logging.info(f"Building VanillaNet: {ins_units} -> "
                     f"{hidden_layers} -> {outs_units}")
        model = net.VanillaNet(
            ins_units,
            outs_units,
            deep_layers_units=hidden_layers,
            dropout_rate=dropout_rate,
            activation=activation_fn,
        ).to(device)
    elif model_name.lower() in ['indicatorgenonet', 'indicator_geno_net']:
        logging.info(f"Building IndicatorGenoNet: {ins_units} -> "
                     f"{hidden_layers} -> {outs_units}")
        model = net.IndicatorGenoNet(
            input_units=ins_units, # n_snps * n_channels
            output_units=outs_units,
            deep_layers_units=hidden_layers,
            dropout_rate=dropout_rate,
            n_snps=n_snps,
            include_dom=include_dom,
            include_rec=include_rec,
            use_wide=True,
            activation='relu',
            wide_init_scale=0.01
        ).to(device)
    elif model_name.lower() in ['linearridge', 'linear_ridge']:
        logging.info(f"Building LinearRidge: {ins_units} -> {outs_units}")
        model = net.LinearRidge(ins_units, output_units=outs_units).to(device)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    
    if torch_compile_enabled and hasattr(torch, 'compile'):
        model = torch.compile(model, mode=torch_compile_mode)
        logging.info(f"Compiled model (mode={torch_compile_mode})")
    
    # optimizer
    if optimizer_name.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, 
                                     weight_decay=l2_lambda)
    elif optimizer_name.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, 
                                      weight_decay=l2_lambda)
    elif optimizer_name.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, 
                                    weight_decay=l2_lambda, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # loss function
    pred_transform = None
    if task_type == 'binary_classification':
        # loss function is BCE with batch-wise weighting
        # to address class imbalance
        control_count = int(np.sum(y_train == 0))
        case_count = int(np.sum(y_train == 1))
        if case_count == 0 or control_count == 0:
            logging.warning("Detected empty class in training labels; "
                            f"using pos_weight=1.0.")
            pos_weight_val = 1.0
        else:
            pos_weight_val = max(1.0e-6, control_count / max(1, case_count))
        pos_weight_tensor = torch.tensor(
            [pos_weight_val], dtype=torch.float32, device=device
        )
        # The loss for pos examples is multiplied by the pos_weight, 
        # effectively increasing the penalty for misclassifying pos examples. 
        # (helps the model pay more attention to pos class during training)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor, 
                                             reduction='mean')
        pred_transform = torch.sigmoid
        logging.info(f"Positive class weight for BCEWithLogitsLoss: "
                     f"{pos_weight_val:.6f}")
    elif task_type == 'regression':
        if loss_func == 'mse':
            loss_fn = torch.nn.MSELoss(reduction='mean')
        elif loss_func == 'mae':
            loss_fn = torch.nn.L1Loss(reduction='mean')
        else:
            raise ValueError(f"Unknown loss function: {loss_func}")
    else:
        raise ValueError(f"Unknown task type (needed for loss function): "
                         f"{task_type}")

    # metrics
    task_metrics = metrics.get_task_metrics(task_type)

    # check that monitor_metric is a valid metric in task_metrics;
    # allow 'loss' monitoring if it's not in task_metrics
    if monitor_metric != 'loss' and monitor_metric not in task_metrics:
        raise ValueError(f"Unknown monitor metric: {monitor_metric}")
    
    # Mixed precision training setup with GradScaler
    if model_precision is None or model_precision == 'fp32':
        amp_enabled = False
        amp_dtype_tensor = None
        scaler = None
        logging.info("Precision: FP32 (no AMP).")
    elif model_precision == 'amp_fp16':
        amp_enabled = True
        amp_dtype_tensor = torch.float16
        scaler = torch.cuda.amp.GradScaler()
        logging.info("Precision: AMP FP16 with GradScaler.")
    elif model_precision == 'amp_bf16':
        amp_enabled = True
        amp_dtype_tensor = torch.bfloat16
        scaler = None  # BF16 typically does not use GradScaler
        logging.info("Precision: AMP BF16 (no GradScaler).")
    else:
        raise ValueError("precision must be 'fp32', 'amp_fp16', or 'amp_bf16'.")
    
    # Learning rate scheduler (constructed before checkpoint, so 
    # it gets restored correctly if resuming from checkpoint)
    scheduler = None
    if use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs, 
            last_epoch=0
        )

    # check if a checkpoint exists
    if restore_file is not None:
        # Restore from checkpoint (assume .pth.tar extension)
        checkpoint_path = os.path.join(model_dir, f"{restore_file}.pth.tar")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"File does not exist: {checkpoint_path}")
        checkpoint = utils.load_checkpoint(
            checkpoint_path, 
            model, 
            optimizer=optimizer, 
            scheduler=scheduler, 
            scaler=scaler, # device=device
        )
        model.to(device)
        # load the json configuration file
        with open(os.path.join(model_dir, 'metrics_val_best.json'), 'r') as f:
            prev_best_metrics = json.load(f)
        with open(os.path.join(model_dir, 'metrics_val_last.json'), 'r') as f:
            prev_last_metrics = json.load(f)
        
        start_epoch = checkpoint['epoch']
        best_val_score = prev_best_metrics[monitor_metric]
        last_val_score = prev_last_metrics[monitor_metric]

        # if metrics_val_best.json contains epoch, then restore best_epoch
        best_epoch = int(prev_best_metrics.get('epoch', start_epoch))

        # early-stopping state
        es_state = checkpoint.get('early_stopping', None)
        if es_enabled and es_state is not None:
            # keep JSON best as the source of truth if mismatch
            best_val_score = float(es_state.get('best_score', best_val_score))
            epochs_no_improve = int(es_state.get('epochs_no_improve', 0))
            stopped_epoch = es_state.get('stopped_epoch', None)
        else:
            epochs_no_improve = 0
            stopped_epoch = None
        
        logging.info('')
        logging.info(f"Loaded checkpoint from {checkpoint_path}")
        logging.info(f"Starting training from epoch {start_epoch}")
        logging.info(f"Best epoch's validation {monitor_metric} was {best_val_score:.4f}")
        logging.info(f"Last epoch's validation {monitor_metric} was {last_val_score:.4f}")
    else:
        start_epoch = 0
        best_epoch = 0
        best_val_score = (-np.inf if es_mode == 'max' else np.inf)
        epochs_no_improve = 0
        stopped_epoch = None
        logging.info("No restore file. Starting training from scratch.")


    num_model_params = sum(p.numel() for p in model.parameters())
    num_trainable_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info('')
    logging.info(f"Model:\n{model}")
    logging.info(f"Optimizer: {optimizer}")
    logging.info(f"Model parameters: total={num_model_params:,}, "
                 f"trainable={num_trainable_model_params:,}")
    logging.info(f"Model compiled: {torch_compile_enabled} "
                 f"(mode={torch_compile_mode})")
    logging.info(f"Model device location: {next(model.parameters()).device}")
    logging.info(f"Monitor metric: {monitor_metric}")
    logging.info(f"Learning rate scheduler: {scheduler}")
    logging.info('')

    # Training
    logging.info('========== STARTING TRAINING ==========')
    history = []
    # main training loop
    for epoch in range(start_epoch, num_epochs):
        lr_start = optimizer.param_groups[0]['lr']
        logging.info(f"Epoch {epoch+1}/{num_epochs} "
                     f"(lr={lr_start:.3e})")
        
        # train for one epoch
        train_metrics_avg = train(
            model, optimizer, dl_train, loss_fn, task_metrics, 
            l1_lambda=l1_lambda, 
            save_summary_steps=save_summary_steps, 
            grad_clip=grad_clip, 
            grad_scaler=scaler, 
            pred_transform=pred_transform,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype_tensor,
        )
        show_epoch_metrics(train_metrics_avg, split_name="train", 
                           print_fn=logging.info, precision=4)
        
        # evaluate val set
        val_metrics_avg = evaluate(
            model, dl_val, loss_fn, task_metrics, device,
            pred_transform=pred_transform, 
            amp_enabled=amp_enabled, amp_dtype=amp_dtype_tensor,
        )
        show_epoch_metrics(val_metrics_avg, split_name="val", 
                           print_fn=logging.info, precision=4)
        
        # record to history
        current_history = {'epoch': epoch+1}
        current_history.update(
            {f"train_{k}": v for k, v in train_metrics_avg.items()}
        )
        current_history.update(
            {f"val_{k}":v for k,v in val_metrics_avg.items()}
        )
        history.append(current_history)

        # step the scheduler (if any)
        if scheduler is not None:
            scheduler.step()

        # early stopping + checkpointing
        current_val_score = float(val_metrics_avg[monitor_metric])
        if not np.isfinite(current_val_score):
            # nans treated as non-improvements for early stopping
            logging.warning(f"val_{monitor_metric} is non-finite "
                            f"at epoch {epoch+1}; skipping improvement check.")
            improved = False
        else:
            improved = _is_improvement(current_val_score, best_val_score, 
                                       es_min_delta, es_mode)
        # governs saving best.pth.tar
        is_best = improved
        # add the epoch number to the val metrics dict
        val_metrics_avg['epoch'] = int(epoch+1)
        if improved:
            logging.info(f"*** new best val_{monitor_metric} ***")
            best_val_score = current_val_score
            best_epoch = int(epoch + 1)
            epochs_no_improve = 0
            # save best validation metrics to json file (when best is achieved)
            utils.save_dict_to_json(
                val_metrics_avg, 
                os.path.join(model_dir, "metrics_val_best.json")
            )
        else:
            epochs_no_improve += 1
        
        # always save 'last' metrics
        utils.save_dict_to_json(
            val_metrics_avg, os.path.join(model_dir, "metrics_val_last.json")
        )
        # assemble checkpoint state dict
        current_state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(), 
            'optimizer': optimizer.state_dict(),
        }
        if scheduler is not None:
            current_state['scheduler'] = scheduler.state_dict()
        if scaler is not None:
            current_state['scaler'] = scaler.state_dict()
        if es_enabled:
            # persist the early-stopping state
            current_state['early_stopping'] = {
                'best_score': float(best_val_score),
                'epochs_no_improve': int(epochs_no_improve),
                'stopped_epoch': (
                    int(stopped_epoch) if stopped_epoch is not None else None
                ),
            }
        # Save checkpoints (takes care of overwriting last.pth.tar and, 
        # if is_best is True, also best.pth.tar)
        utils.save_checkpoint(current_state, is_best=is_best, 
                              checkpoint=model_dir)
        # Early stopping decision
        if (es_enabled 
            and (epoch + 1) >= es_min_epochs 
            and epochs_no_improve >= es_patience):
            stopped_epoch = int(epoch + 1)
            logging.info('========== EARLY STOPPING TRIGGERED ==========')
            logging.info(f"Stopped at epoch {stopped_epoch}. "
                         f"Best epoch was {best_epoch} with "
                         f"val_{monitor_metric}={best_val_score:.6f}.")
            break # exit epoch iter loop when early stopping is triggered
        logging.info('')

    logging.info('========== DONE TRAINING ==========')
    logging.info(f"Best epoch was {best_epoch} with a validation "
                 f"{monitor_metric} of {best_val_score:.5f}.")
    
    # Create a dataframe from the training/validation history
    history_df = pd.DataFrame(history)
    # Save the history dataframe to a csv (history.csv in model_dir)
    history_path = os.path.join(model_dir, 'history.csv')
    history_df.to_csv(history_path, index=False)

    logging.info(f"Results directory content: {model_dir}")
    for fn in os.listdir(model_dir):
        logging.info(fn)
