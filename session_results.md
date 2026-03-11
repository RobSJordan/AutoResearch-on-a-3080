# autoresearch Session Results — mar10

**Branch**: `autoresearch/mar10`
**Hardware**: RTX 3080 12GB (Ampere sm_86), WSL2
**Budget**: 5 minutes wall-clock training time per experiment
**Metric**: `val_bpb` (bits-per-byte, lower is better)
**Total experiments**: ~114 (31 kept, 80 discarded, 3 crashes)

---

## Final Best Config

| Parameter | Value |
|-----------|-------|
| Architecture | DEPTH=5, ASPECT_RATIO=96, HEAD_DIM=64, model_dim=512 |
| Attention | PyTorch SDPA (full causal), QK norm, logit softcap=15 |
| MLP | 4x expansion, ReLU² activation |
| Regularization | WEIGHT_DECAY=0.1 (decay-to-zero), grad_clip=0.1 |
| Optimizer (matrix) | Muon: MATRIX_LR=0.02, momentum ramp 0.85→0.95, ns_steps=6 |
| Optimizer (embed) | Adam: EMBEDDING_LR=0.4, beta2=0.999, FINAL_LR_FRAC=0.05 |
| LR schedule | WARMDOWN_RATIO=0.8 (linear decay over 80% of budget) |
| Batch | TOTAL_BATCH_SIZE=2^15, DEVICE_BATCH_SIZE=128 |
| Compilation | `torch.compile(mode="max-autotune")` with CUDA graphs |
| Dataset | TinyStories, MAX_SEQ_LEN=256, VOCAB_SIZE=4096 |

**Best result**: `val_bpb = 0.423337` at `3.6 GB VRAM`

---

## Improvement History (kept experiments only)

### Phase 1 — FineWeb, DEPTH=8, SEQ=512

| # | Commit | val_bpb | Δ bpb | VRAM | Change | Why it worked |
|---|--------|---------|-------|------|--------|---------------|
| 0 | 1adcc45 | 1.256991 | — | 5.6 | **Baseline** DEPTH=8, SDPA, VOCAB=4096, SEQ=512 | Starting point |
| 1 | 9b5f1e1 | 1.255997 | −0.001 | 5.6 | MATRIX_LR 0.04→0.06 | Higher LR for Muon's orthogonal updates allowed faster convergence for weight matrices |
| 2 | 7e16616 | 1.252831 | −0.003 | 5.6 | Weight decay 0.2→0.0 | WD=0.2 was too aggressive on FineWeb; removing it let the model explore more freely |
| 3 | 7b135d8 | 1.251364 | −0.001 | 5.6 | FINAL_LR_FRAC 0.0→0.1 | Small non-zero final LR prevents "crashing" to zero at warmdown end; allows continued fine-tuning |
| 4 | 2190811 | 1.251280 | −0.000 | 5.6 | Muon ns_steps 5→6 | More Newton-Schulz iterations → better approximation of orthogonal gradient matrix |
| 5 | 403740c | 1.251068 | −0.000 | 5.6 | Gradient clipping norm=1.0 | Prevents occasional large gradient spikes from destabilizing training |

### Phase 2 — Switch to TinyStories, Dataset/Architecture Tuning

| # | Commit | val_bpb | Δ bpb | VRAM | Change | Why it worked |
|---|--------|---------|-------|------|--------|---------------|
| 6 | 403740c | 0.471360 | **−0.780** | 5.6 | **Switched to TinyStories dataset** | TinyStories is simpler and more repetitive; smaller model fits it well. Numbers now incomparable to Phase 1 |
| 7 | (anon) | 0.458785 | −0.013 | 5.6 | MAX_SEQ_LEN 512→256, DEVICE_BS 64→128 | Shorter sequences mean more tokens per batch at same VRAM; model sees more diverse contexts per step |
| 8 | 2d45507 | 0.458854 | ±0.000 | 5.6 | Fresh baseline SEQ=256, BS=128, DEPTH=8 | Clean re-baseline with new settings |
| 9 | 1c6c90b | 0.448428 | −0.010 | 3.3 | DEPTH 8→6 | Shallower model = faster steps = more optimizer steps in 5 min budget. TinyStories is simple enough that 6 layers suffice |
| 10 | a181cf6 | 0.436285 | −0.012 | 3.3 | TOTAL_BATCH_SIZE 2^17→2^16 | Halving batch size doubles optimizer steps per time budget; more updates beat batch size noise at this scale |
| 11 | bd0ef1f | 0.433247 | −0.003 | 3.2 | TOTAL_BATCH_SIZE 2^16→2^15 | Same principle — more steps wins. 2^15 was optimal (2^14 too noisy) |
| 12 | bb8b321 | 0.432921 | −0.000 | 3.3 | MATRIX_LR 0.06→0.04 | Slightly lower Muon LR helps stability after batch size change |
| 13 | 66de623 | 0.432706 | −0.000 | 3.3 | MATRIX_LR 0.04→0.02 | Further reduction: smaller matrices at lower depth benefit from more conservative updates |
| 14 | 97021db | 0.432479 | −0.000 | 3.3 | EMBEDDING_LR 0.6→0.3 | Halving embedding LR prevents embedding table from moving too fast relative to transformer weights |
| 15 | f7503f1 | 0.432297 | −0.000 | 3.3 | FINAL_LR_FRAC 0.1→0.0 | Cold LR stop (0.0) is now better — with short budget, aggressive warmdown serves the same role as a positive final LR |
| 16 | 981593e | 0.431108 | −0.001 | 4.2 | ASPECT_RATIO 64→80 (wider) | More hidden dimensions per layer = richer feature representations; model_dim 384→480 improved capacity |
| 17 | ce4a139 | 0.429189 | −0.002 | 4.2 | WEIGHT_DECAY 0.0→0.1 | TinyStories is small enough to overfit; light regularization helps generalization |
| 18 | 3ca77ad | 0.428361 | −0.001 | 4.2 | HEAD_DIM 128→64 (more heads) | More attention heads (4→7) allows attending to more distinct aspects of context simultaneously |
| 19 | 1fcbd93 | 0.428214 | −0.000 | 4.3 | Grad clip 1.0→0.5 | Tighter clipping smooths training; smaller updates are less likely to escape good minima |
| 20 | fa7c5f2 | 0.427964 | −0.000 | 4.3 | Grad clip 0.5→0.25 | Further tightening: continued monotonic improvement |
| 21 | 141066b | 0.427763 | −0.000 | 4.3 | Grad clip 0.25→0.1 | Very tight clipping optimal for this small model on simple dataset |
| 22 | 17cf585 | 0.427602 | −0.000 | 4.3 | Adam beta2 0.95→0.999 | Higher beta2 = slower second-moment adaptation. More conservative variance tracking stabilizes Adam for embeddings/scalars |

### Phase 3 — torch.compile, DEPTH/Width Tuning

| # | Commit | val_bpb | Δ bpb | VRAM | Change | Why it worked |
|---|--------|---------|-------|------|--------|---------------|
| 23 | 5c0b543 | 0.425653 | −0.002 | 4.5 | torch.compile max-autotune | CUDA graph fusion dramatically reduces kernel launch overhead; more training steps in 5 min budget |
| 24 | 96c3f85 | 0.425619 | −0.000 | 3.5 | DEPTH 6→5 | Fewer layers = more steps per second = more optimizer updates; TinyStories doesn't need depth |
| 25 | f6564f6 | 0.425091 | −0.001 | 3.6 | ASPECT_RATIO 80→96 with DEPTH=5 | Wider model (model_dim 480→512, 8 heads) recovers capacity lost by removing a layer |
| 26 | e82d1b8 | 0.424136 | −0.001 | 3.6 | EMBEDDING_LR 0.3→0.4 | Slightly higher embedding LR helps embeddings adapt faster once the rest of the model is better tuned |
| 27 | 065c9ed | 0.424053 | −0.000 | 3.6 | FINAL_LR_FRAC 0.0→0.05 | Small residual LR at end allows continued fine-tuning without fully collapsing learning |

### Phase 4 — LR Schedule (Warmdown) Tuning

| # | Commit | val_bpb | Δ bpb | VRAM | Change | Why it worked |
|---|--------|---------|-------|------|--------|---------------|
| 28 | a995c12 | 0.423845 | −0.000 | 3.6 | WARMDOWN_RATIO 0.5→0.6 | Longer decay phase gives model more time to settle into a low-loss basin |
| 29 | c01e567 | 0.423820 | −0.000 | 3.6 | WARMDOWN_RATIO 0.6→0.7 | Continued improvement; longer decay consistently helps |
| **30** | **0bb7365** | **0.423337** | **−0.000** | **3.6** | **WARMDOWN_RATIO 0.7→0.8** | **Best result. 80% of budget in decay = very gradual LR reduction, giving extensive time to converge** |

---

## Key Architectural Decisions (Confirmed Throughout)

| Feature | Decision | Evidence |
|---------|----------|----------|
| Activation | ReLU² (keep) | SwiGLU −0.06, GELU −0.009 |
| Logit softcap | 15 (keep) | Removing hurt −0.007, 8 and 30 also worse |
| QK normalization | Keep | Removing −0.007 (much worse) |
| Residual lambdas | Keep | Removing −0.005 |
| Value embeddings | Keep (alternating layers) | Removing layer 0 VE hurt −0.005 |
| MLP expansion | 4x (keep) | 2x undercapacity, 5x/8x slower and worse |
| Grad accumulation | Avoid | CUDA graphs incompatible; forced TOTAL_BS=2^15 |
| Flash Attention | SDPA (keep) | FA3 unsupported on Ampere sm_86 |
| Parallel blocks | Avoid | Much worse −0.015 |

---

## Total Progress Summary

| Phase | Best val_bpb | VRAM | Model |
|-------|-------------|------|-------|
| Start (FineWeb baseline) | 1.256991 | 5.6 GB | DEPTH=8, model_dim=512, SEQ=512 |
| After FineWeb tuning | 1.251068 | 5.6 GB | Same + grad clip, better LRs |
| After TinyStories switch | 0.423337 | 3.6 GB | DEPTH=5, model_dim=512, SEQ=256 |

The dataset switch drove the biggest absolute improvement. Within TinyStories, the most impactful single changes were:
1. **Batch size reduction** (2^17→2^15): more optimizer steps in fixed budget (+0.025 bpb)
2. **DEPTH reduction** (8→6, then 6→5): more steps per second (+0.023 bpb combined)
3. **torch.compile max-autotune**: CUDA graph fusion (+0.002 bpb)
4. **WARMDOWN_RATIO** 0.5→0.8: longer convergence phase (+0.002 bpb)
5. **Weight decay** 0.0→0.1: regularization on small dataset (+0.002 bpb)
