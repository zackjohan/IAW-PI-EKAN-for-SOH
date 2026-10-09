# PI-EKAN

Research code for battery state-of-health estimation using PI-EKAN, including PS-BO SOC-window selection, feature extraction, seven comparison models, ablation studies, and single-cell adaptation.

<!-- Add the paper title and publication link here. -->
<!-- Add your images under images/ and uncomment these lines when available.
![PI-EKAN framework](images/framework.png)

![Experimental results](images/results.png)
-->

# 1. System requirements

Python **3.12**. The model implementations were checked using the following environment:

| Package | Version |
| --- | --- |
| PyTorch | 2.14.0 |
| NumPy | 2.3.5 |
| pandas | 2.2.3 |
| scikit-learn | 1.8.0 |
| psutil | 7.2.2 |

PS-BO additionally requires SciPy, matplotlib, seaborn, scikit-optimize, tqdm, and h5py. CPU execution was validated; CUDA is selected when available.

# 2. Installation guide

## 2.1 Create and activate an environment

```bash
conda create -n pi-ekan python=3.12
conda activate pi-ekan
```

## 2.2 Install dependencies

Run from the PI-EKAN project root, beside `main_pi_ekan.py` and `requirements.txt`:

```bash
python -m pip install -r requirements.txt
python -m pip install scipy matplotlib seaborn scikit-optimize tqdm h5py
```

## 2.3 Organize the project

Place this README and the three experiment folders at the PI-EKAN project root.

| Folder | Contents |
| --- | --- |
| `model/` | PI-EKAN and comparison architectures |
| `dataloader/` | Shared model data loading and preprocessing |
| `util/` | Training, evaluation, logging, and computational tracking |
| `ps-bo_and_feature_extraction/` | Dataset-specific SOC-window selection and feature extraction |
| `ablation_study/` | Ablation configuration, runner, and its supplied support files |
| `singel_cell_adaptation/` | Transfer runner and configuration; imports the shared data processor |

Keep the folder spelling `singel_cell_adaptation`. The ablation package uses its supplied local loader.

# 3. Demo

Run the commands below from the project root. Replace example paths with your local directories.

## 3.1 PS-BO and feature extraction

Set paths in each dataset script, or use `--root` while retaining its configured subfolder layout.

```bash
python ps-bo_and_feature_extraction/xjtu_ncm_psbo.py --root /path/to/raw_data --run 1 2
```

Option **1** performs PS-BO window selection and feature extraction for all configured cells. Option **2** extracts features from the configured fixed SOC windows.

| Dataset | Script |
| --- | --- |
| XJTU | `xjtu_ncm_psbo.py` |
| TJU: NCA and NCM_NCA | `tju_nca_psbo.py` |
| CALCE | `calce_lco_psbo.py` |
| HNEI | `hnei_psbo_.py` |
| NA-ion | `na_ion_psbo.py` |
| SNL | `snl_lfp_psbo.py` |
| Stanford | `stanford_nca_psbo.py` |

SNL and Stanford also provide option **3**, the sampling-frequency sweep, and option **4**, robustness scenarios A–D:

```bash
python ps-bo_and_feature_extraction/snl_lfp_psbo.py --root /path/to/raw_data --run 1 2 3 4
python ps-bo_and_feature_extraction/stanford_nca_psbo.py --root /path/to/raw_data --run 1 2 3 4
```

Inputs are dataset-specific PKL, CSV, or MAT files. Use the exported feature CSVs for model training; raw-data and feature-data roots are separate.

## 3.2 PI-EKAN and comparison models

Each cell CSV must contain seven feature columns followed by capacity/SOH. The shared loader inserts the cycle index and fits normalization on training cells only. Configure cell identifiers and feature paths in `dataloader/dataloader.py`.

```bash
python main_pi_ekan.py --dataset NCM --data-root /path/to/features
python main_ekan.py --dataset NCM --data-root /path/to/features
python main_pinn.py --dataset NCM --data-root /path/to/features
python main_bc_pinn.py --dataset NCM --data-root /path/to/features
python main_baseline.py --dataset NCM --data-root /path/to/features --models LSTM Transformer LSTM-KAN RNN-KAN
```

NCM features default to `<data-root>/NCM/PS-BO Window Selected`. Use `--dataset-dir` for an exact directory or `--dataset all` for all eight configurations.

For small-sample experiments and parameter counts:

```bash
python main_pi_ekan.py --dataset NCM --data-root /path/to/features --small-samples 1 2 3 4
python count_parameters.py
```

All model runners accept `--small-samples` and default to ten runs. Logs, losses, predictions, checkpoints, metrics, and computational summaries are saved under `results/`; override it with `--results-root`.

## 3.3 Ablation study

Configure experiments in `ablation_study/ablation_config.py`. Families A–F cover inputs, dynamics-network inputs, loss weighting, SOC windows, reference-cell selection, and basis functions.

```bash
python -m ablation_study.ablation_runner --list --data-root /path/to/features
python -m ablation_study.ablation_runner --families A B C --data-root /path/to/features --out results/ablation
```

Family A reads feature names from LFP. Families D/E require their configured feature directories. Outputs include losses, adaptive weights, validation results, checkpoints, and summary CSVs. Computational profiling is omitted.

## 3.4 Single-cell adaptation

Set paths in `singel_cell_adaptation/config.py`. Supply a pretrained PI-EKAN checkpoint at `pretrained_models/<source>/model.pth` for each source chemistry.

```bash
python -m singel_cell_adaptation.main_adaptation_hybrid_ekan_LoRA_fine_tuning --method hybrid --source NCM --target LFP --data-root /path/to/features
python -m singel_cell_adaptation.main_adaptation_hybrid_ekan_LoRA_fine_tuning --method standard --all-pairs --data-root /path/to/features
```

The study contains these **16 directed transfer pairs**:

| Source | Targets |
| --- | --- |
| NCM | LFP, NCA, LCO |
| LFP | NCM, NCA, LCO |
| NCA | NCM, LFP, LCO |
| LCO | NCM, NCA, LFP |
| NA-ion | NCM, NCA, LFP, LCO |

Defaults: five runs, one labeled adaptation cell per run, and 100 fixed epochs. Both evaluations use adaptation-cell scaling; remaining cells are tested without validation-based checkpoint selection. NCM/LCO use the feature-window paths in the adaptation configuration.

# 4. Datasets and additional information

| Dataset | Name in code | Source |
| --- | --- | --- |
| XJTU / NCM | `NCM` | [Zenodo](https://doi.org/10.5281/zenodo.10963339) |
| TJU / NCA | `NCA` | [Zenodo](https://zenodo.org/record/6405084) |
| TJU / mixed chemistry | `NCM_NCA` | [Zenodo](https://zenodo.org/record/6405084) |
| CALCE / LCO | `LCO` | [CALCE Battery Data](https://calce.umd.edu/battery-data) |
| SNL / LFP | `LFP` | [Battery Archive](https://www.batteryarchive.org) |
| HNEI / mixed chemistry | `NCM_LCO` | [Battery Archive](https://www.batteryarchive.org) |
| Stanford / NCA | `Stanford` | [Stanford Digital Repository](https://purl.stanford.edu/td676xr4322) |
| Sodium-ion | `NA-ion` | [Zenodo](https://zenodo.org/records/17960956) |

Names follow the code: HNEI is `NCM_LCO`; TJU includes `NCM_NCA`.

See the module READMEs and runner docstrings for protocol details. Sequence models forecast the next cycle, while paired models estimate SOH at the input cycle. Adaptation test sets overlap across runs. Results may vary with seeds and hardware; synthetic execution checks do not establish real-data research performance.

# 5. Citation

Please cite the associated PI-EKAN work and the original dataset publications when using this repository.

**PI-EKAN paper:** Z. Yohannes, Y. Xu, Y. Wei, J. Li, B. Jia, "Cross-chemistry battery state-of-health estimation from adaptive partial charging using interpretable physics-informed Efficient Kolmogorov–Arnold networks," *Energy and AI*, 2026, 100928. https://doi.org/10.1016/j.egyai.2026.100928

```bibtex
@article{yohannes2026piekan,
  author  = {Yohannes, Zekariyas and Xu, Yonghong and Wei, Yidi and Li, Jian and Jia, Boru},
  title   = {Cross-chemistry battery state-of-health estimation from adaptive partial charging using interpretable physics-informed Efficient {Kolmogorov--Arnold} networks},
  journal = {Energy and AI},
  year    = {2026},
  pages   = {100928},
  doi     = {10.1016/j.egyai.2026.100928},
  url     = {https://doi.org/10.1016/j.egyai.2026.100928},
}
```
