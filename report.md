# JIPNet Reproducibility Audit Report

**Paper**: "Joint Identity Verification and Pose Alignment for Partial Fingerprints" (arXiv:2405.03959v4, TIFS 2025)
**Repository audited**: `/home/lang-phu-quy/Documents/Sem6/Pattern_Recognition/JIPNet`
**Audit date**: 2026-06-27

---

## Summary of Critical Issues

| Severity | ID | Description |
|---|---|---|
| Critical | BUG-01 | `EnhanceNet` crashes with any decoder (missing `ModuleList` init) |
| Critical | BUG-02 | `JIPNet()` with defaults crashes (int vs list type for `dec_nhead`, `dec_local_num`) |
| Critical | BUG-03 | `squeeze()` without dim causes `IndexError` at `batch_size=1` |
| Critical | MISS-01 | Pre-training script entirely absent |
| Critical | MISS-02 | `Lseg` segmentation loss not implemented |
| Critical | MISS-03 | `fp_verifinger.py` missing — alignment step unusable |
| High | DISC-01 | Attention is linear (ELU kernel), not softmax as stated in paper Eq. 3 |
| High | DISC-02 | Config batch size = 2, paper says 128 |
| High | DISC-03 | Released model has 38.44M params vs 25.7M claimed (parameter count measured on pruned model) |
| Moderate | DISC-04 | Patch center sampling uses Gaussian+ring approach, not uniform polar as described |
| Moderate | BUG-04 | Logging format string silently drops epoch number |
| Moderate | BUG-05 | Dead code assignment in `generate_pos` (line 140 overwritten at 144) |
| Minor | BUG-06 | `AffinePatch` class duplicated in two files |
| Minor | BUG-07 | `interence_AKAZE.py` filename typo breaks documented command |
| Minor | DISC-05 | FVC2006 dataset name: abstract/README say DB1\_A, paper body/tables say DB2\_A |
| Minor | DISC-06 | Transformer FFN uses ReLU, not GELU (paper cites ViT which uses GELU) |
| Info | UNDC-01 | `RidgeNet`, `TokenMerging`, `GlobalFeatureTransformer` are dead code |
| Info | UNDC-02 | `skimage==0.0` in `requirements.txt` is an invalid pip version spec |

---

## 1. Architecture Discrepancies

### DISC-01 (High): Attention Mechanism — Linear vs Softmax

**Paper** Section III-B (Eq. 3): writes the standard dot-product softmax attention:
> *Attention(Q, K, V) = softmax(QKᵀ / √dk) · V*

**Code** `models/JIPNet.py` line 81 hardcodes `attention='linear'`. The actual implementation in `models/ViT/module/linear_attention.py` is **ELU-based linear attention** from Katharopoulos et al. ("Transformers are RNNs"), not dot-product softmax. The paper's formula is therefore **not implemented**.

### DISC-06 (Minor): FFN Activation — ReLU vs GELU

**Paper** cites ViT [49] (Dosovitskiy et al.) for its FFN design; ViT uses **GELU**. The FE Block (`models/Enhancer.py` line 102) correctly uses `nn.GELU()`. However, the transformer FFN in `models/ViT/module/transformer.py` lines 26–30 uses `nn.ReLU(True)`.

### DISC-03 (High): Parameter Count — 25.7M vs 38.44M

**Paper** Figure 13: *"JIPNet: 25.7M parameters"*.

With config `ckpts/JIPNet/config.yaml` (`dec_local_num=[8,4]`), the full released model has **38.44M parameters**. Removing only the alignment branch (`local_transformer_align` + `align_head`) yields **25.74M**, which matches the paper. The paper measured the **classification-only (pruned) model** without the pose regression head, but the released model includes both heads. This is undocumented.

### DISC-07 (Minor): Stage Numbering Inconsistency Inside the Paper

- Paper body (line 540): encoder blocks in "Stage 2", residual blocks in "Stage 3"
- Paper Table I (lines 491–501): encoder blocks in "Stage 1", residual blocks in "Stage 2"

The code matches Table I, not the body text. This is an internal paper error.

### INFO: Architecture Points That Match

- Stage 3 output: 264 channels, 20×20 → Transformer×8 → PatchMerging → 384, 10×10 ✓
- Stage 4 uses separate but symmetric branches for classification and alignment ✓
- Two separate `nn.Module` heads (`local_transformer_cla` + `local_transformer_align`) with independent weights ✓
- `width=32`, encoder doubles channels each stage (32→64→128→256→264) ✓
- Patch Merging: 2×2 neighbourhood concatenation + linear projection (4×264→384) ✓

---

## 2. Training Configuration

### DISC-02 (High): Batch Size

| Parameter | Paper (Section V-A) | `configs/JIPNet.yaml` |
|---|---|---|
| Batch size | 128 | `2  # 128` (128 commented out) |

The working config uses **batch size 2**. Training at the paper's batch size requires uncommenting the value. This makes the released training config non-reproducible at paper settings.

### Epoch Count

Paper: *"until convergence (about 12 epochs)"*. Config line 6: `epochs: 16`. Minor discrepancy; checkpoints are saved only from epoch 10 onward (`train_JIPNet.py` line 285: `epochs - 6`).

### GPU Device Hardcoded

Config `cuda_ids: [2]` and `train_JIPNet.py` line 53 both hardcode GPU index 2. Will fail on machines with fewer GPUs or different indices.

### Parameters That Match

| Parameter | Paper | Code/Config |
|---|---|---|
| Initial LR | 1e-3 | 1.0e-3 ✓ |
| End LR | 1e-6 | 1.0e-6 ✓ |
| Scheduler | Cosine annealing | CosineAnnealingLR ✓ |
| Optimizer | AdamW | adamW, weight_decay=1e-2 ✓ |
| Dec heads Stage 3 | 6 | dec_nhead[0]=6 ✓ |
| Dec heads Stage 4 | 12 | dec_nhead[1]=12 ✓ |
| Stage 3 blocks | 8 | dec_local_num[0]=8 ✓ |
| Stage 4 blocks | 4 each | dec_local_num[1]=4 ✓ |

---

## 3. Loss Function

### MISS-02 (Critical): `Lseg` Not Implemented

Paper Section III-D-3 (lines 968–983) describes `Lseg`, a simplified focal loss for binary fingerprint segmentation used during **enhancement pre-training**. The segmentation head exists in `EnhancerDecoder.ending` (`models/Enhancer.py`) but the corresponding loss function is **completely absent** from `loss.py` and any other file.

### Loss Components That Match

| Parameter | Paper | `loss.py` |
|---|---|---|
| Focal α | 0.2 | `alpha=0.2` (line 26) ✓ |
| Focal γ | 2.0 | `gamma=2.0` (line 26) ✓ |
| λ (loss weight) | 0.002 | `w=0.002` (line 60) ✓ |
| ω (rotation weight) | 0.99 | `lambda_2=0.99` (line 97) ✓ |

### Undocumented: Regression Loss Masked to Genuine Pairs

`loss.py` line 99: `Lr = torch.sum(l2 * cla_gt.reshape((-1,))) / (torch.sum(cla_gt) + eps)` — the regression loss is masked by `cla_gt`, so it is only computed for genuine matching pairs. This is sensible but **not stated in the paper** (Eq. 9 does not mention this masking).

---

## 4. Data Generation Pipeline

### MISS-03 (Critical): VeriFinger Wrapper Missing

`make_data/affine_pairs.py` line 21:
```python
from fptools.fp_verifinger import Verifinger, load_minutiae
```
`fptools/fp_verifinger.py` **does not exist**. The README (line 130) acknowledges this: *"This script will not run successfully due to licensing restrictions."* The alignment step of the data generation pipeline **cannot be executed**.

### DISC-04 (Moderate): Patch Center Sampling Differs from Paper

**Paper** Section IV-A: *"randomly select patch center from the common mask … another center is uniformly sampled in polar coordinates, random angle ±180°, random radius 0–100/70/20 pixels."*

**Code** `make_data/generate_patch.py`:
- `pos1` is sampled with **Gaussian weighting** centered on the mask centroid (not uniform over the mask)
- `pos2` is selected **uniformly from a ring** at a fixed radius ±15 px (not uniform polar sampling)

Implementation is functionally different from the description.

### BUG-05 (Moderate): Dead Code in `generate_pos`

`make_data/generate_patch.py` lines 140–144:
```python
pos2 = sample_gaussian_points(ring_mask, center=center, variance=var2)  # DEAD — result overwritten
pos_ring_arrs = np.where(ring_mask > 0)
idx = random.randint(0, len(pos_ring_arrs[0]) - 1)
pos2 = np.array([pos_ring_arrs[0][idx], pos_ring_arrs[1][idx]])          # actual result
```
Line 140 is dead code. Suggests an unfinished refactor.

### Suspicious: Nested `np.meshgrid`

`generate_patch.py` lines 133–135:
```python
x, y = np.meshgrid(np.array(np.arange(0, w)),
                   np.meshgrid(np.arange(0, h)),   # <-- inner meshgrid returns a tuple
                   indexing='xy')
```
The inner `np.meshgrid(np.arange(0, h))` returns a 1-tuple; passing a tuple as the second argument is accidental but produces the correct shapes. Should be `np.meshgrid(np.arange(w), np.arange(h))`.

### Undocumented: FVC2000 in Data Generation

`make_data/affine_pairs.py` lines 128–143 includes `/disk1/finger/FVC2000/DB{1,2,3}_A/`. FVC2000 is not mentioned anywhere in the paper as training data.

---

## 5. Data Augmentation

### MISS-01 (Critical): Pre-training Augmentation Script Absent

Paper Section III-A (lines 755–797) describes: mirror flip, 90/180/270 rotation, grayscale inversion, Gaussian blur/noise, dilation/erosion, white blob augmentation for **enhancement pre-training**. No pre-training training script exists in this repository. The README links to `https://github.com/XiongjunGuan/FpEnhancer` and notes released weights are **incompatible** with JIPNet. The pre-training pipeline is **not reproducible** from this codebase.

### Main Training Augmentation (Matches Paper)

`augmentation.py` implements:
- Perlin-based blob sensor noise (dry/wet variants) ✓
- `dryness` (morphological dilation) ✓
- `heavypress` (morphological erosion) ✓
- Gaussian blur + Gaussian noise ✓
- Grayscale inversion ✓

These match the partial-fingerprint augmentation described in Section III-C.

**Undocumented**: `data_loader.py` line 138 swaps `(img1, img2)` with 50% probability. Sensible but not documented.

---

## 6. Evaluation Protocol

### MISS-04: No Metric Computation Scripts

No script computes AUC, EER, TAR@FAR described in Section V-C. The inference scripts (`inference_AFRNet.py`, `inference_DeepPrint.py`, etc.) exist but only output per-pair scores; metric aggregation is absent.

### BUG-07 (Minor): Filename Typo — `interence_AKAZE.py`

The file is named `interence_AKAZE.py` (note: `interence` instead of `inference`). The README documents it as `inference_AKAZE.py`. Running `python inference_AKAZE.py` will fail with `No such file or directory`.

### DISC-05 (Minor): FVC2006 Dataset Name Inconsistency

| Location | Dataset name |
|---|---|
| Paper abstract (line 28) | FVC2006 **DB1\_A** |
| README (line 37) | FVC2006 **DB1A** |
| Paper body Section IV (line 987) | FVC2006 **DB2\_A** |
| Paper Table III (line 1106) | FVC2006 **DB2\_A** |

The body and tables consistently use DB2\_A, which is correct for the experiments. The abstract and README contain an error.

---

## 7. Bugs Found in Code

### BUG-01 (Critical): `EnhanceNet` — Missing `ModuleList` Initialization

**File**: `models/Enhancer.py`, `EnhanceNet.__init__`

```python
# These lines run before self.ups and self.decoders are initialized:
for num in dec_blk_nums:
    self.ups.append(...)       # AttributeError: 'EnhanceNet' object has no attribute 'ups'
    self.decoders.append(...)  # AttributeError
```

Compare with `EnhancerDecoder.__init__` (lines 272–273) which correctly does:
```python
self.ups = nn.ModuleList()
self.decoders = nn.ModuleList()
```

`EnhanceNet` is **unusable** with any non-empty `dec_blk_nums`. `EnhancerDecoder` (a separate class) is the one actually used in training and works correctly.

### BUG-02 (Critical): `JIPNet` Default Arguments Crash on Iteration

**File**: `models/JIPNet.py`, lines 33–34:
```python
def __init__(self, ..., dec_nhead=8, dec_local_num=6, ...):
```

Line 72: `for num in dec_local_num:` → `TypeError: 'int' object is not iterable`
Line 126 (`ViT_reg_cla.py`): `nheads[0]` → `TypeError: 'int' object is not subscriptable`

`JIPNet()` with default arguments **crashes immediately**. Only works when called with explicit list values (as in `train_JIPNet.py` and `inference.py` via config YAML).

### BUG-03 (Critical): `squeeze()` Without Dimension Argument — Breaks at Batch Size 1

**Files**: `models/ViT/ViT_reg_cla.py`, `ClassificationHead.forward` line 60 and `AlignHead.forward` line 99:
```python
x = self.avgpool(x).squeeze()
```

- **B > 1**: `(B, C, 1)` → squeeze removes dim 2 → `(B, C)` — correct
- **B = 1**: `(1, C, 1)` → squeeze removes **both** dim 0 and 2 → `(C,)` — wrong shape

At `batch_size=1` (inference), `loss.py` line 78 (`align_pred[:, 0]`) raises:
```
IndexError: too many indices for tensor of dimension 1
```

**Fix**: replace `.squeeze()` with `.squeeze(-1)` in both locations.

### BUG-04 (Moderate): Silent Logging Bug — Epoch Number Dropped

**File**: `train_JIPNet.py`, lines 136 and 187:
```python
logging_info = "\tTRAIN: ".format(epoch)   # epoch is never inserted (no {} placeholder)
logging_info = "\tVALID: ".format(epoch)   # same
```
The `.format(epoch)` call is a no-op. No epoch number appears in log output.
**Fix**: change to `"\tTRAIN epoch {}: ".format(epoch)`.

### BUG-06 (Minor): `AffinePatch` Duplicated in Two Files

`models/utils.py` (lines 22–68) and `utils.py` (lines 45–95) both define `class AffinePatch(nn.Module)` with functionally equivalent but independently maintained implementations. `inference.py` imports from `models.utils`; `data_loader.py` imports from `utils`. Risk of silent divergence.

---

## 8. Missing Components

| ID | Component | Where described in paper | Status in code |
|---|---|---|---|
| MISS-01 | Enhancement pre-training script | Section III-A, Table II | Absent — linked to separate FpEnhancer repo (incompatible weights) |
| MISS-02 | `Lseg` segmentation loss | Eq. 10, Section III-D-3 | Not implemented |
| MISS-03 | `fp_verifinger.py` (VeriFinger wrapper) | Section IV-A, make_data/ | Absent (proprietary SDK) |
| MISS-04 | Evaluation metric scripts (AUC, EER, TAR@FAR) | Section V-C | Absent |
| MISS-05 | Pre-training data prep (128×128 patches from 5,000 rolled FPs) | Section III-A | Absent |

---

## 9. Undocumented Additions

| ID | Item | Location | Note |
|---|---|---|---|
| UNDC-01a | `RidgeNet` baseline | `other_models/RidgeNet/` | Not in paper comparisons; likely removed baseline |
| UNDC-01b | `TokenMerging` class | `models/ViT/utils/merge.py` lines 17–38 | Never imported or used; dead code |
| UNDC-01c | `GlobalFeatureTransformer` | `models/ViT/module/transformer.py` lines 116–156 | Never imported or used; dead code |
| UNDC-01d | `TransformAffinePred` | `utils.py` lines 23–42 | Dead code; replaced by inline `norm_pred()` in inference.py |
| UNDC-01e | `sample_mask_points()` | `make_data/generate_patch.py` lines 93–112 | Defined but never called |
| UNDC-02 | FVC2000 in data gen | `make_data/affine_pairs.py` lines 128–143 | Not in paper training datasets |
| UNDC-03 | Gaussian-weighted patch sampling | `generate_patch.py` `generate_pos()` | Not described; paper says uniform sampling |
| UNDC-04 | Pair swap augmentation (50%) | `data_loader.py` line 138 | Not documented |
| UNDC-05 | `skimage==0.0` | `requirements.txt` line 8 | Invalid version; `pip install` fails. Correct: `scikit-image>=0.19` |

---

## 10. Hyperparameter Summary Table

| Parameter | Paper (Section V-A) | Config/Code | Match |
|---|---|---|---|
| Initial LR | 1e-3 | 1.0e-3 | ✓ |
| End LR | 1e-6 | 1.0e-6 | ✓ |
| Scheduler | Cosine annealing | CosineAnnealingLR | ✓ |
| Optimizer | AdamW | adamW | ✓ |
| Weight decay | default (1e-2) | 1e-2 | ✓ |
| Batch size | 128 | 2 (128 commented out) | ✗ |
| Epochs | ~12 | 16 | ~ |
| λ (loss weight) | 0.002 | 0.002 | ✓ |
| α (focal loss) | 0.2 | 0.2 | ✓ |
| γ (focal loss) | 2.0 | 2.0 | ✓ |
| ω (rot/trans weight) | 0.99 | 0.99 | ✓ |
| γ (seg loss) | 2.0 | N/A — not implemented | ✗ |
| Stage 3 transformer heads | 6 | dec_nhead[0]=6 | ✓ |
| Stage 4 transformer heads | 12 | dec_nhead[1]=12 | ✓ |
| Stage 3 blocks | 8 | dec_local_num[0]=8 | ✓ |
| Stage 4 blocks | 4 per branch | dec_local_num[1]=4 | ✓ |
| Encoder width | 32 | width=32 | ✓ |
| Pre-train patches | 128×128 | not in repo | N/A |
| Pre-train image pool | 5,000 rolled FPs | not in repo | N/A |
| Pre-train epochs | 200 | not in repo | N/A |
| Model params | 25.7M | 38.44M (full) / 25.74M (CLA only) | Conditional ✓ |
| Attention type | softmax (paper Eq. 3) | linear (ELU kernel) | ✗ |
