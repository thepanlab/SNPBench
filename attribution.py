# attribution.py
"""
Generate SNP interpretation scores for a trained model.

For each SNP j, compute the mean absolute interpretation score across 
all test samples i as: InterpScore_j = Mean(|Attribution_ij|). 

Usage:
$ python attribution.py --model_dir ./experiments/dnn_decoy_minibatcher4 \
    --baselines_format modes \
    --mult-by-inputs

$ CUDA_VISIBLE_DEVICES=1 nohup python3 -u attribution.py \
    --model_dir ./path_to_dir_with_saved_model_and_config \
    --baselines_format ['zeros'|'mean'|'mode'|'random_sample'] \
    [--mult-by-inputs | --no-mult-by-inputs] \
    > nohup.interp.out 2>&1 &

Arparse args:
--model_dir (str): Path to folder containing config.yaml and best.pth.tar.
--baselines_format (str): Baseline strategy ('zeros', 'means', 'modes', 
                          'random_sample').
--mult-by-inputs (bool): If True: global attributions (multiply_by_inputs). 
                         If False: local attributions.
--n_baseline_samples (int): For 'random_sample' baseline, this represents
                            the number of random training samples used.
"""

import argparse, logging, os, sys, time, warnings
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch

from ukb_data import UKBData
from model.datasets import DaskIterableDataset
import model.interpretation as interp
import model.nnet as net
import utils

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

N_BASELINE_SAMPLES_DEFAULT = 1000  # only used if baselines_format=random_sample

def create_parser():
    p = argparse.ArgumentParser(description='UKB DNN Attributions', fromfile_prefix_chars='@')
    p.add_argument('--model_dir', 
                   type=str, 
                   required=True,
                   help="Path to folder containing config.yaml and best.pth.tar.")
    p.add_argument('--baselines_format', 
                   type=str, 
                   default='modes',
                   choices=['zero','zeros','two','twos','mean','means','mode','modes','random_sample'],
                   help="Baseline strategy.")
    p.add_argument('--mult-by-inputs', 
                   action=argparse.BooleanOptionalAction,
                   default=True, 
                   dest='mult_by_inputs',
                   help="If True: global attributions (multiply_by_inputs). If False: local attributions.")
    p.add_argument('--n_baseline_samples', 
                   type=int, 
                   default=None,
                   help=f"For 'random_sample': number of training samples to draw (default: {N_BASELINE_SAMPLES_DEFAULT}).")
    return p


def load_trained_model(ckpt_path, ins_units, outs_units, deep_layers_units, dropout_rate, device):
    model = net.VanillaNet(ins_units, outs_units,
                           deep_layers_units=deep_layers_units,
                           dropout_rate=dropout_rate)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model.to(device)

def build_dataloader(X, y, batch_size, shuffle=False, drop_last=False, imputers=None,
                     transform_x=None, transform_y=None, use_decoys=False, decoy_seed=42, num_workers=0):
    ds = DaskIterableDataset(
        X=X, y=y, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        drop_last=drop_last,
        imputers=imputers, 
        transform_x=transform_x, 
        transform_y=transform_y,
        use_decoys=use_decoys, 
        decoy_seed=decoy_seed,
    )
    return torch.utils.data.DataLoader(
        ds, batch_size=None, pin_memory=True,
        num_workers=num_workers, persistent_workers=num_workers > 0
    )

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    model_dir = os.path.abspath(args.model_dir)
    assert os.path.isdir(model_dir), f"Model directory not found: {model_dir}"

    cfg_path = os.path.join(model_dir, 'config.yaml')
    assert os.path.isfile(cfg_path), f"Config file not found: {cfg_path}"

    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    attr_type = 'global' if args.mult_by_inputs else 'local'
    mult_flag = args.mult_by_inputs # global vs local

    # logging
    log_path = os.path.join(model_dir, f'attributions_{attr_type}_baselines_{args.baselines_format.lower()}.log')
    utils.set_logger(log_path)

    logging.info("> Interpretation")
    cfg = utils.load_yaml(cfg_path)
    cfg['attribution_type'] = attr_type
    cfg['mult_by_inputs'] = args.mult_by_inputs
    cfg['baselines_format'] = args.baselines_format
    logging.info('-'*50)
    for k, v in cfg.items():
        logging.info(f"  {k}: {v}")
    logging.info('-'*50)

    # Data
    data = UKBData(cfg, verbose=True)
    test_data = data.get_partition(split='test', optimize_chunks=True, verbose=True)
    X_test, y_test = test_data.X, test_data.y

    if cfg.get('use_decoys', False):
        assert cfg.get('decoy_seed') is not None, \
            "cfg['decoy_seed'] must be set if using decoys."
        decoy_seed = cfg.get('decoy_seed')
        test_decoy_seed = decoy_seed + 2

    if not cfg.get('use_decoys', False):
        # non-decoy ins_units
        ins_units = X_test.shape[1]
        test_decoy_seed = None
    else:
        # decoy ins_units
        ins_units = int(2 * X_test.shape[1])
        assert cfg.get('decoy_seed') is not None, \
            "cfg['decoy_seed'] must be set if using decoys."
        test_decoy_seed = cfg.get('decoy_seed') + 2

    dl_test = build_dataloader(
        X_test, y_test,
        cfg.get('batch_size', 128),
        shuffle=False, drop_last=False, imputers=None,
        transform_x=None, transform_y=None,
        use_decoys=cfg.get('use_decoys', False), 
        decoy_seed=test_decoy_seed,
        num_workers=cfg.get('num_workers', 0)
    )
    
    output_units = 1

    # Model
    logging.info('-'*50)
    logging.info("> Loading trained model:")
    ckpt_path = os.path.join(model_dir, 'best.pth.tar')
    model = load_trained_model(
        ckpt_path, ins_units, output_units, cfg['hidden_layers'], cfg['dropout_rate'], device
    )
    logging.info(f'Model loaded from: {ckpt_path}')
    logging.info(f"{model}")
    logging.info(f"Model on GPU: {next(model.parameters()).is_cuda}")
    logging.info(f"Model in eval mode: {not model.training}")
    logging.info('-'*50)

    # Baselines
    logging.info("> Interpretation Baselines")
    fmt = args.baselines_format.lower()
    if fmt in ['zero', 'zeros']:
        X_baseline = torch.zeros(X_test.shape[1], dtype=torch.float32).view(1, -1)
        logging.info("Using 'zeros' baseline.")
    elif fmt in ['two', 'twos']:
        X_baseline = 2.0 * torch.ones(X_test.shape[1], dtype=torch.float32).view(1, -1)
        logging.info("Using 'twos' baseline.")
    elif fmt in ['mean', 'means']:
        logging.info("Computing TRAIN per-SNP means baseline...")
        means = data.variant_means(split='train').astype(np.float32)
        assert means.shape[0] == X_test.shape[1], "Mean baseline length mismatch."
        X_baseline = torch.tensor(means, dtype=torch.float32).view(1, -1)
        logging.info("Using 'means' baseline (from training set).")
    elif fmt in ['mode', 'modes']:
        logging.info("Computing TRAIN per-SNP modes baseline...")
        modes = data.variant_modes(split='train', snap_to_int=True).astype(np.float32)
        assert modes.shape[0] == X_test.shape[1], "Mode baseline length mismatch."
        X_baseline = torch.tensor(modes, dtype=torch.float32).view(1, -1)
        logging.info("Using 'modes' baseline (from training set).")
    elif fmt == 'random_sample':
        train_data = data.get_partition(split='train', optimize_chunks=True)
        X_train = train_data.X
        nK = args.n_baseline_samples or N_BASELINE_SAMPLES_DEFAULT
        nK = min(int(nK), int(X_train.shape[0]))
        rng = np.random.default_rng(seed=cfg.get('decoy_seed', 42))
        idx = rng.choice(np.arange(X_train.shape[0]), size=nK, replace=False).astype(np.int64)
        X_baseline = torch.tensor(X_train[idx, :].compute(), dtype=torch.float32)
        X_baseline = torch.nan_to_num(X_baseline, nan=0.0)
        logging.info(f"Using {nK} train set random samples as baselines.")
    else:
        logging.error(f"Unknown --baselines_format {args.baselines_format}")
        sys.exit(1)

    if cfg.get('use_decoys', False):
        X_baseline = torch.cat([X_baseline, X_baseline], dim=1)
    logging.info('-'*50)
    
    assert X_baseline.shape[1] == ins_units, \
        f"Baseline width {X_baseline.shape[1]} != input units {ins_units}"
    
    # use single-array baseline for methods that require it
    bl = X_baseline.mean(dim=0, keepdim=True) \
        if X_baseline.ndim == 2 else X_baseline

    # Algorithms
    logging.info("> Starting interpretation...")
    algo_functions = {
        'Saliency': interp.captum_saliency,
        'SaliencySG': interp.captum_saliency,
        #'Shap': interp.shap_shap,
        'GradShap': interp.captum_gradientshap, 
        'GradShapSG': interp.captum_gradientshap,
        'DeepLift': interp.captum_deeplift,
        'DeepLiftSG': interp.captum_deeplift,
        'IntegratedGradients': interp.captum_integratedgradients,
        'IntegratedGradientsSG': interp.captum_integratedgradients,
    }
    # number of attributed samples attributed per slice
    slice_size = 1
    # algorithm parameters
    algo_params = {
        # no baseline methods
        'Saliency': {'abs': False, 'target': 0, 
                     'slice_size': slice_size, 
                     'smoothgrad': False},
        'SaliencySG': {'abs': False, 'target': 0, 
                       'slice_size': slice_size, 
                       'smoothgrad': True},
        
        # SHAP (uses baseline(s) directly; no mult_by_inputs concept doesn't apply)
        #'Shap': {'baselines': X_baseline},

        # methods that support multiply_by_inputs & baselines
        # 'advanced gradient methods'
        'GradShap': {'mult_by_inputs': mult_flag, 
                     'baselines': bl, 'target': 0,
                     'slice_size': slice_size, 'smoothgrad': False}, 
        'GradShapSG': {'mult_by_inputs': mult_flag, 
                       'baselines': bl, 'target': 0, 
                       'slice_size': slice_size, 'smoothgrad': True}, 
        'DeepLift': {'mult_by_inputs': mult_flag, 
                     'baselines': bl, 'target': 0, 
                     'slice_size': slice_size, 'smoothgrad': False}, 
        'DeepLiftSG': {'mult_by_inputs': mult_flag, 
                       'baselines': bl, 'target': 0, 
                       'slice_size': slice_size, 'smoothgrad': True}, 
        'IntegratedGradients': {'mult_by_inputs': mult_flag, 
                                'baselines': bl, 'target': 0, 
                                'slice_size': slice_size, 'smoothgrad': False}, 
        'IntegratedGradientsSG': {'mult_by_inputs': mult_flag, 
                                  'baselines': bl, 'target': 0, 
                                  'slice_size': slice_size, 'smoothgrad': True}, 
    }
    
    algo_names = list(algo_functions.keys())
    logging.info(f"Interpretation algorithms: {algo_names}")

    sample_count = 0
    attrs_running_sum = {algo: np.zeros(ins_units, dtype=np.float32) for algo in algo_names}
    timer_log = {algo: 0.0 for algo in algo_names}

    for batch_idx, batch in enumerate(tqdm(dl_test)):
        X_batch = batch[0].to(torch.float32).to(device)
        sample_count += X_batch.shape[0]

        for algo, algo_func in algo_functions.items():
            t0 = time.time()
            batch_attrs = algo_func(model, X_batch, device, **algo_params[algo])
            timer_log[algo] += (time.time() - t0)

            batch_attrs = batch_attrs.detach().cpu().numpy().squeeze()
            attrs_running_sum[algo] += np.sum(np.abs(batch_attrs), axis=0)

    # SNP labels
    if cfg.get('use_decoys', False):
        decoy_snp_labels = [f'dec{i+1:06d}' for i in range(len(test_data.variant_df))]
        snp_labels = np.concatenate([test_data.variant_df['SNP'].values, decoy_snp_labels])
    else:
        snp_labels = test_data.variant_df['SNP'].values

    first_len = len(attrs_running_sum[algo_names[0]])
    assert len(snp_labels) == first_len, (
        f"Attributed SNPs ({first_len}) != SNP labels ({len(snp_labels)})."
    )

    # Mean |attr| across samples
    scores_df = pd.DataFrame(
        {algo: attrs_running_sum[algo] / sample_count for algo in algo_names},
        index=snp_labels,
    )

    # Timing: samples/sec
    samples_per_second = pd.Series(
        {algo: (sample_count / timer_log[algo]) if timer_log[algo] > 0 else float('inf') for algo in algo_names},
        index=algo_names, name="samples_per_second"
    )

    # Save
    scores_filename = f"attributions_{attr_type}_baselines_{args.baselines_format}.csv"
    times_filename = f"attributions_{attr_type}_baselines_{args.baselines_format}.time_obs_per_sec.csv"
    scores_path = os.path.join(model_dir, scores_filename)
    times_path = os.path.join(model_dir, times_filename)

    scores_df.to_csv(scores_path, index_label='snp')
    samples_per_second.to_csv(times_path, index_label="algorithm")

    logging.info(f"Saved file: {scores_path}")
    logging.info(f"Saved file: {times_path}")
    logging.info('-'*50)
    logging.info(f"Samples per second:\n{samples_per_second}")
    logging.info(f"Scores (first 5 rows):\n{scores_df.head()}")
    logging.info('-'*50)
