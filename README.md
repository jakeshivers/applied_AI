# Applied_AI — My fast.ai Journey (CPU)

> A clean, reproducible record of my fast.ai coursework using **uv**, **PyTorch (CPU)**, and **Jupyter**.  
> Repo name: `applied_AI` • Env: `.applied_ai`

---

## Quick Start

```bash
# 1) Create / activate env
uv venv .applied_ai
source .applied_ai/bin/activate

# 2) Minimal CPU stack
uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
uv pip install fastai jupyterlab ipykernel pandas

# 3) Register kernel
python -m ipykernel install --user --name=.applied_ai --display-name "Python (.applied_ai)"

# 4) Launch notebooks
jupyter lab
```

> Optional (nice hygiene):  
> ```bash
> uv pip install nbstripout
> nbstripout --install
> ```

---

## Repo Layout

```
applied_AI/
  notebooks/
    00_env_check.ipynb
    01_lesson1_image_classification_mnist.ipynb
    02_lesson2_augmentation_and_lr_finder.ipynb
    03_lesson3_transfer_learning_and_tuning.ipynb
    04_lesson4_text_classification_imdb.ipynb
    05_lesson5_multilabel_image_classification_planet.ipynb
    06_lesson6_image_segmentation_camvid.ipynb
    07_lesson7_tabular_rossmann_regression.ipynb
    08_lesson8_collaborative_filtering_movielens.ipynb
    09_lesson9_interpretation_and_error_analysis.ipynb
    10_lesson10_export_inference_and_logging.ipynb
    99_sandbox.ipynb
  metrics/
    fastai_run_metrics.csv
  artifacts/            # saved models, images, exports (gitignored)
  data/                 # datasets (gitignored)
  scripts/
  .applied_ai/          # uv venv (gitignored)
  README.md
  requirements.txt
  .gitignore
```

Suggested `.gitignore`:
```
.applied_ai/
__pycache__/
*.pyc
.ipynb_checkpoints/
data/
artifacts/
*.pkl
.DS_Store
```

---

## Logging & Artifacts

Each notebook appends a row to `metrics/fastai_run_metrics.csv`:

| timestamp_utc | run_tag | model | epochs | valid_loss | accuracy |
|---|---|---:|---:|---:|---:|
| 2025-08-26T19:12:00Z | mnist_base | resnet18 | 1 | 0.1683 | 0.9377 |

Conventions:
- Set `RUN_TAG` at the top of each notebook (e.g., `mnist_base`, `mnist_aug_128`).
- Save outputs to `artifacts/<RUN_TAG>_<UTC_TIMESTAMP>/...`.

---

## Lessons Overview

Use this table to track progress. Links go to the notebooks; metrics are pasted from the CSV.

| #  | Lesson (fast.ai)                                    | Notebook                                                    | Best metric | Status |
|----|------------------------------------------------------|-------------------------------------------------------------|------------:|:------:|
| 01 | Image classification (MNIST intro)                   | `notebooks/01_lesson1_image_classification_mnist.ipynb`     | `acc=…`     | [x]    |
| 02 | Augmentation + LR finder                             | `notebooks/02_lesson2_augmentation_and_lr_finder.ipynb`     | `acc=…`     | [ ]    |
| 03 | Transfer learning & fine-tuning                      | `notebooks/03_lesson3_transfer_learning_and_tuning.ipynb`   | `acc=…`     | [ ]    |
| 04 | Text classification (IMDb)                           | `notebooks/04_lesson4_text_classification_imdb.ipynb`       | `acc=…`     | [ ]    |
| 05 | Multi-label images (Planet)                          | `notebooks/05_lesson5_multilabel_image_classification_planet.ipynb` | `f1=…` | [ ]    |
| 06 | Segmentation (CamVid)                                | `notebooks/06_lesson6_image_segmentation_camvid.ipynb`      | `dice=…`    | [ ]    |
| 07 | Tabular (Rossmann)                                   | `notebooks/07_lesson7_tabular_rossmann_regression.ipynb`    | `rmse=…`    | [ ]    |
| 08 | Collaborative filtering (MovieLens)                  | `notebooks/08_lesson8_collaborative_filtering_movielens.ipynb` | `rmse=…` | [ ]    |
| 09 | Interpretation & error analysis                      | `notebooks/09_lesson9_interpretation_and_error_analysis.ipynb` | `—`     | [ ]    |
| 10 | Export, inference & logging                          | `notebooks/10_lesson10_export_inference_and_logging.ipynb`  | `—`         | [ ]    |

> Check a box when you’ve tagged the lesson in git:  
> `git tag -a lesson-01 -m "Lesson 01 complete"`

---

## Per-Lesson Notes (compact, collapsible)

<details><summary>Lesson 01 — MNIST intro</summary>

**Goal**: Baseline image classifier; understand `DataLoaders`, `vision_learner`, `fine_tune`.  
**Run tags**: `mnist_base`, `mnist_aug_128`  
**What I learned**:
- Transfer learning basics (ResNet18), `fine_tune(1)`
- Quick metrics logging pattern
- How aug + image size affect accuracy/time

</details>

<details><summary>Lesson 02 — Augmentation & LR finder</summary>

**Goal**: Use `aug_transforms`, `lr_find`, discriminative LRs.  
**Run tags**: `mnist_tuned`  
**Notes**:
- Pick `base_lr` from `lr_find()`
- `fit_one_cycle` vs `fine_tune` tradeoffs

</details>

<details><summary>Lesson 03 — Transfer learning & tuning</summary>

**Goal**: Unfreeze, discriminative learning rates, regularization.  
**Notes**: …

</details>

<details><summary>Lesson 04 — Text (IMDb)</summary>

**Goal**: AWD-LSTM classifier on IMDb SAMPLE/IMDb.  
**Notes**: …

</details>

<!-- Repeat the pattern through Lesson 10 -->

---

## Reproducibility

Freeze the environment once it’s stable:
```bash
source .applied_ai/bin/activate
uv pip freeze > requirements.txt
```

Rebuild on a new machine:
```bash
uv venv .applied_ai
source .applied_ai/bin/activate
uv pip install -r requirements.txt
python -m ipykernel install --user --name=.applied_ai --display-name "Python (.applied_ai)"
```

---

## GPU (Optional Later)

Local box is CPU-only. For GPU speed, run the same notebooks on **Kaggle** or **Colab** (GPU runtime) and keep results logged here.

---

## License

`MIT`