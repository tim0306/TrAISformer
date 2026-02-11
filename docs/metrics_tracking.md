# TrAISformer Optimization Metrics Tracking

## Summary

This file tracks model performance across optimization phases.

## Metrics Table

| Phase | Train Time | Val Loss | 10min (km) | 1h (km) | 2h (km) | 3h (km) | Inference (ms) | Status |
|-------|-----------|----------|-----------|---------|---------|---------|----------------|--------|
| **Baseline** | ~17m (50ep) | 1.396 | **0.3346** | **0.7352** | **1.4589** | **2.3926** | **86.36** | ✅ Established |
| **Phase 1** | ~7m (20ep) | 3.629 | **0.3338** (+0.2%) | **0.6865** (+6.6%) | **1.3420** (+8.0%) | **2.4079** (-0.6%) | **86.89** | ✅ Complete |
| Phase 3 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 🔄 Pending |
| Phase 3+extras | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 🔄 Pending |

## Baseline Details (from training log)

- Best epoch: 12
- Validation loss at best epoch: 1.39569
- Training time: ~20 seconds/epoch × 50 epochs = ~17 minutes total
- Model path: `baselines/github_original_model.pt`
- Evaluation results: `baselines/baseline_results.json`

## Target Goals

### Phase 1 (Speed-focused):
- ✅ Training time reduced by ≥30% (17 min → ≤12 min)
- ✅ Prediction accuracy maintained or improved (≤5% change)
- ✅ No NaN losses or training instabilities

### Phase 3 (Quality-focused):
- ✅ Prediction errors reduced by ≥5% across all horizons
- ✅ Validation loss improves (≤1.39 or better)
- ✅ Training time also improved (2-3× faster as secondary benefit)

### Final Combined (All phases):
- ✅ Prediction errors reduced by ≥10-15% on test set
- ✅ Training time reduced by ≥50% (17 min → ≤8 min)
- ✅ Inference speed improved by ≥5× (Flash Attention + batched sampling)

## Notes

- Baseline established: 2026-02-11
- Framework: PyTorch 2.10+
- Device: CUDA (cuda:0)
- Dataset: ct_dma (1453 test samples)
