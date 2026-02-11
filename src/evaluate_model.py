#!/usr/bin/env python
# coding: utf-8
"""
Evaluate a trained TrAISformer model on the test set.
Outputs standardized metrics for comparison.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import pickle
from tqdm import tqdm
import math
import logging
import argparse
import json
import time

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import models, trainers, datasets, utils
from config_trAISformer import Config


def evaluate_model(model_path, config, test_dataloader, device, init_seqlen=18):
    """
    Evaluate model on test set and return metrics.

    Args:
        model_path: Path to model checkpoint
        config: Configuration object
        test_dataloader: DataLoader for test set
        device: torch device
        init_seqlen: Initial sequence length for prediction

    Returns:
        dict: Metrics including prediction errors at multiple horizons
    """
    print(f"Loading model from {model_path}...")

    # Create model
    model = models.TrAISformer(config, partition_model=None)

    # Load state dict (handle torch.compile() wrapper if present)
    state_dict = torch.load(model_path)
    # Remove '_orig_mod.' prefix if present (from torch.compile)
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Prepare for evaluation
    v_ranges = torch.tensor([2, 3, 0, 0]).to(device)
    v_roi_min = torch.tensor([model.lat_min, -7, 0, 0]).to(device)
    max_seqlen = init_seqlen + 6 * 4  # 18 + 24 = 42

    # Storage for predictions
    l_min_errors, l_mean_errors, l_masks = [], [], []
    inference_times = []

    print("Evaluating on test set...")
    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))

    with torch.no_grad():
        for it, (seqs, masks, seqlens, mmsis, time_starts) in pbar:
            seqs_init = seqs[:, :init_seqlen, :].to(device)
            masks = masks[:, :max_seqlen].to(device)
            batchsize = seqs.shape[0]
            error_ens = torch.zeros((batchsize, max_seqlen - init_seqlen, config.n_samples)).to(device)

            # Measure inference time
            batch_start = time.time()

            for i_sample in range(config.n_samples):
                preds = trainers.sample(model,
                                        seqs_init,
                                        max_seqlen - init_seqlen,
                                        temperature=1.0,
                                        sample=True,
                                        sample_mode=config.sample_mode,
                                        r_vicinity=config.r_vicinity,
                                        top_k=config.top_k)
                inputs = seqs[:, :max_seqlen, :].to(device)
                input_coords = (inputs * v_ranges + v_roi_min) * torch.pi / 180
                pred_coords = (preds * v_ranges + v_roi_min) * torch.pi / 180
                d = utils.haversine(input_coords, pred_coords) * masks
                error_ens[:, :, i_sample] = d[:, init_seqlen:]

            batch_time = (time.time() - batch_start) * 1000  # Convert to ms
            inference_times.append(batch_time / batchsize)  # Per sample

            # Accumulation through batches
            l_min_errors.append(error_ens.min(dim=-1))
            l_mean_errors.append(error_ens.mean(dim=-1))
            l_masks.append(masks[:, init_seqlen:])

    # Compute aggregate metrics
    l_min = [x.values for x in l_min_errors]
    m_masks = torch.cat(l_masks, dim=0)
    min_errors = torch.cat(l_min, dim=0) * m_masks
    pred_errors = min_errors.sum(dim=0) / m_masks.sum(dim=0)
    pred_errors = pred_errors.detach().cpu().numpy()

    # Extract errors at specific time horizons
    # Assuming 10-minute intervals: 1 step = 10min, 6 steps = 1h, 12 steps = 2h, 18 steps = 3h
    metrics = {
        'prediction_errors_km': {
            '10min': float(pred_errors[0]) if len(pred_errors) > 0 else None,
            '1hour': float(pred_errors[5]) if len(pred_errors) > 5 else None,
            '2hour': float(pred_errors[11]) if len(pred_errors) > 11 else None,
            '3hour': float(pred_errors[17]) if len(pred_errors) > 17 else None,
        },
        'prediction_errors_full': pred_errors.tolist(),
        'inference_time_ms': float(np.mean(inference_times)),
        'inference_time_std_ms': float(np.std(inference_times)),
        'total_test_samples': len(test_dataloader.dataset),
        'model_path': model_path,
        'config': {
            'n_samples': config.n_samples,
            'sample_mode': config.sample_mode,
            'r_vicinity': config.r_vicinity,
            'top_k': config.top_k,
            'init_seqlen': init_seqlen,
        }
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate TrAISformer model')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output-json', type=str, default='evaluation_results.json',
                        help='Output JSON file for results')
    parser.add_argument('--config-module', type=str, default='config_trAISformer',
                        help='Config module to use')
    args = parser.parse_args()

    # Load configuration
    print("Loading configuration...")
    cf = Config()
    device = cf.device
    init_seqlen = cf.init_seqlen

    # Load test dataset
    print("Loading test dataset...")
    datapath = os.path.join(cf.datadir, cf.testset_name)
    with open(datapath, "rb") as f:
        l_pred_errors = pickle.load(f)

    # Preprocess test data (same as training)
    moving_threshold = 0.05
    for V in l_pred_errors:
        try:
            moving_idx = np.where(V["traj"][:, 2] > moving_threshold)[0][0]
        except:
            moving_idx = len(V["traj"]) - 1
        V["traj"] = V["traj"][moving_idx:, :]

    test_data = [x for x in l_pred_errors
                 if not np.isnan(x["traj"]).any() and len(x["traj"]) > cf.min_seqlen]

    print(f"Test set size: {len(test_data)} trajectories")

    # Create dataset and dataloader
    if cf.mode in ("pos_grad", "grad"):
        test_dataset = datasets.AISDataset_grad(test_data,
                                                max_seqlen=cf.max_seqlen + 1,
                                                device=torch.device("cpu"))
    else:
        test_dataset = datasets.AISDataset(test_data,
                                           max_seqlen=cf.max_seqlen + 1,
                                           device=torch.device("cpu"))

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=cf.batch_size,
                                 shuffle=False)

    # Evaluate model
    results = evaluate_model(args.model_path, cf, test_dataloader, device, init_seqlen)

    # Save results
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output_json}")
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Test samples: {results['total_test_samples']}")
    print(f"\nPrediction Errors:")
    for horizon, error in results['prediction_errors_km'].items():
        if error is not None:
            print(f"  {horizon:8s}: {error:.4f} km")
    print(f"\nInference Time: {results['inference_time_ms']:.2f} ± {results['inference_time_std_ms']:.2f} ms")
    print("="*60)


if __name__ == "__main__":
    main()
