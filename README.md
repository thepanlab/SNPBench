# Benchmarking interpretability of deep learning for predictive genomics: recall, precision, and variability of feature attribution

## Conda environments

**PLINK environment (PLINK v1.9 and v2.0)**

```bash
# use the bioconda channel
conda config --add channels bioconda
# create plink environment
conda create -n plink
conda install -c bioconda plink plink2
# activate plink environment for plink step(s)
conda activate plink
```

**Python (3.9) environment**

```bash
conda config --add channels conda-forge
conda config --set channel_priority strict
# create python environment
conda create -n pyukb python=3.9
# standard data science libraries
conda install numpy pandas scipy scikit-learn matplotlib seaborn pandas-plink pyyaml
# for GPU support (recommended)
conda install pytorch-gpu torchvision
# or, for CPU-only support
conda install pytorch-gpu torchvision
# install captum for interpretation
conda install captum
# activate python environment
conda activate pyukb
```

## Preprocessing

The `preprocessing.ipynb` notebook helps prepare genotype, phenotype, and other data items for downstreamtasks. It uses helper functions from `main_dataset_utils.py` to extract relevant UKB fields, apply sample QC, and adjust phenotype labels. 

**Inputs**:

- UK Biobank main dataset (ukb*.csv)
- Field IDs for sample QC and phenotype (age, sex, height, ethnicity, etc.)
- Merged genotypes (ukb_c1-22.[bed|bim|fam])
- Withdrawals file

**Main Steps**

- **Load and parse main dataset**: Get cleaned data fields from the main dataset.
- **Main dataset sample QC**: Remove withdrawn participants, sex mismatches, sex-chromosome aneuploidy cases, and non-European ancestry samples.
    - Outputs `main_keep_ids.txt` for genotype filtering.
- Suggested PLINK genotype QC command: 
    ```bash 
    plink2 --bfile ukb_c1-22 \
    --autosome \
    --keep main_keep_ids.txt \
    --geno 0.1 \
    --hwe 1e-15 \
    --mac 100 \
    --maf 0.01 \
    --mind 0.1 \
    --write-samples --write-snplist \
    --make-bed \
    --out ukb_c1-22_qc
    ```
- **Train/Validation/Test Split**: Randomly partition post-QC samples into non-overlapping training, validation, and test sets.
- **Phenotype Adjustment**: Fit OLS model of the phenotype (e.g., height) on covariates such as age and sex using the training set only. Residuals represent phenotype variation unexplained by covariates, providing a covariate-corrected label for model training. Phenotype labels are transformed (z-score or inverse normal) and saved.
- **GWAS**: Fit a GWAS model (i.e., generalized linear model or glm) using genotypes and adjusted phenotypes from the train set:
    ```bash
    plink2 --bfile ./data/ukb_c1-22_qc \
        --keep ./data/train_rs1234.id \
        --glm allow-no-covars --variance-standardize \
        --pheno ./data/height_adj.pheno \
        --pheno-name height_adj_z \
        --out ./data/gwas
    ```
- Compute allele frequencies (PLINK v2.0; used with spike-sin simulation):
    ```bash
    plink2 --bfile ./data/ukb_c1-22_qc \
        --keep ./data/train_rs1234.id \
        --freq --out ./data/maf_train_rs1234
    ```

**Outputs**

```bash
./data/
├── ukb_c1-22_qc.bed
├── ukb_c1-22_qc.bim
├── ukb_c1-22_qc.fam
├── ukb_c1-22_qc.id
├── ukb_c1-22_qc.snplist
├── main_keep_ids.txt
├── height.pheno
├── main_covars.covar
├── train_rs1234.id
├── val_rs1234.id
├── test_rs1234.id
├── height_adj.pheno
├── maf_train_rs1234.afreq
└── gwas.height_adj_z.glm.linear
```

## Spikein Synthetic SNP Simulation

The `spikein_simulation.py` module generates synthetic spike-in variants for benchmarking feature attribution methods. The simulator produces additive, dominant, recessive, and epistatic (pairwise) variants that are statistically correlated with a user-provided phenotype vector while maintaining realistic allele frequency distributions based on real genotype data (training set only MAFs is recommended).

**Overview**

- Purpose: Create ground-truth variants with known (and controlled) genotype–phenotype relationships for evaluating interpretability metrics (e.g., recall, precision, stability).
- Input: PLINK-formatted genotype data (`.bed/.bim/.fam`), a phenotype table (`FID/IID/<phenotype>`), and optional precomputed allele frequencies (.afreq, .frq, .npy, etc.).
- Output:
    - Synthetic genotypes (`.syn.csv`)
    - A corresponding variant manifest (`.manifest.csv`)
    - A FAM-aligned synthetic VCF (`.syn.vcf`)
    - A YAML run configuration snapshot (`.run_config.yaml`)

**How it Works**

1. Phenotype alignment:

    Loads the phenotype file, aligns it to the .fam sample order, and drops individuals with missing labels.

2. MAF sampling:

    Draws allele frequencies from user-supplied PLINK output (real_mafs_path), typically computed from the training set using:

    ```bash
    # train only
    plink2 --bfile ./data/ukb_c1-22_qc \
        --keep ./data/train_rs1234.id \
        --freq \
        --out ./data/maf_train_rs1234
    ```

3. Synthetic SNP generation:

    For each effect type (additive, dominant, recessive, epistatic), the simulator samples alleles according to the target MAF range, adjusts correlation with the phenotype ($\rho$), and validates shape constraints (e.g., carrier/homozygote counts, correlation tolerance).

4. Output alignment and export:

    Synthetic genotypes are aligned to the .fam order from the real dataset and written as both CSV and VCF files.
    The VCF’s `#CHROM`, `POS`, and `ID` fields are synthetic placeholders, with unique variant IDs and consistent spacing (`vcf_step`).

**Example Configuration** (`spikein_config.yml`):
```yaml
meta:
  mode: "s6"
  seed: 42
  rng_seed: 42

inputs:
  phenotypes: "./data/height_adj.pheno"
  pheno_col: "height_adj_z"
  genotypes: "./data/ukb_c1-22_qc"
  real_mafs_path: "./data/maf_train_rs1234.afreq"

design:
  n_add: 100
  n_nonlin: 300
  nonlin_design:
    dominant: 100
    recessive: 100
    epistatic: 50   # 2 columns per pair -> 100 epistatic columns total

ranges:
  add_rho_range: [0.025, 0.055]
  dom_rho_range: [0.018, 0.030]
  rec_rho_range: [0.035, 0.065]
  epi_rho_range: [0.045, 0.070]

  maf_ranges:
    dominant:  [0.050, 0.080]
    recessive: [0.060, 0.120]
    epistatic: [0.080, 0.200]

controls:
  additive:
    add_min_n22: 1500
    add_dev_max: 0.20
    add_max_tries: 20
  dominant:
    min_max_dom_carrier_frac: [0.10, 0.14]
    dom_shape_max_tries: 40
  recessive:
    min_max_rec_hom_n: [1200, 4500]
    rec_shape_max_tries: 40
  epistatic:
    main_abs_cap: 0.010
    rho_tol: 0.004
    max_alpha_iters: 16
    max_resamples_per_alpha: 8

outputs:
  out_root: "./data/spikein_sim" # or provide out_prefix instead
  write_csv: true
  write_vcf: true
  vcf_chrom: "26"
  vcf_start: 1000000
  vcf_step: 10
```

**Running the Simulation**

```bash
python spikein_simulation.py --config ./spikein_config.yml
```


This will create a directory such as:

```bash
./data/spikein_sim/spikein_s6_add100_dom100_rec100_epi100_seed42/
```

containing:

```bash
simset.syn.vcf
simset.syn_genotypes.csv
simset.manifest.csv
simset.run_config.yaml
```

Convert synthetic VCF to BED (use PLINK v2.0):

```bash
plink2 \
  --vcf ./data/spikein/spikein_s6_add100_dom100_rec100_epi100_seed42/simset.syn.vcf \
  --chr-set 26 \
  --double-id \
  --make-bed \
  --out ./data/spikein/spikein_s6_add100_dom100_rec100_epi100_seed42/simset.syn_only
```

- `--double-id` since UKB `.fam` expects FID = IID. The `--double-id` makes the new BED use the same convention, so FID/IID pairs will match exactly when merging with the real dataset.
- `--chr-set 26` tells PLINK2 to accept the synthetic chromosome “26”.

Merge with existing `ukb_c1–22_qc` dataset (PLINK v1.9):

```bash
plink --bfile ./data/ukb_c1-22_qc \
      --bmerge ./data/spikein/spikein_s6_add100_dom100_rec100_epi100_seed42/simset.syn_only \
      --make-bed \
      --out ./data/spikein/spikein_s6_add100_dom100_rec100_epi100_seed42/ukb_c1-22_qc_simset.real_plus_syn
```

Run GWAS on real + simulated variants

```bash
plink2 \
  --bfile ./spikein/spikein_s6_add100_dom100_rec100_epi100_seed42/ukb_c1-22_qc_simset.real_plus_syn \
  --glm allow-no-covars hide-covar \
  --variance-standardize \
  --keep ./data/train_rs1234.id
  --pheno ./data/height_adj.pheno \
  --pheno-name height_adj_z \
  --no-input-missing-phenotype
  --out ./data/spikein/spikein_s6_add100_dom100_rec100_epi100_seed42/gwas_real_plus_syn \
```

## DNN Training

Feed forward multilayer perceptrons (DNNs) are trained in three scenarios depending on the target interpretation task:

1. Train with the real + simulated genotypes for attribution recall benchmarking.
2. Train with the real + decoy genotypes for attribution precision benchmarking.
3. Train an ensemble of models for ensemble stability benchmarking. 

Data location, hyperparameters, and other settings are specified in the `config.yaml` file found in the directory where trained models and other results will be stored. An example configuration file is shown below. 

```yaml
# config.yaml located in model_dir
task_type: "regression" # task type (used for determining metrics to evaluate during training)
verbose: true # logs additional information during training run when true
seed: 42 # global seed
model_dir: "./experiments_qc1/dnn_spikein" # directory where config.yml is stored
ref_allele: "a0" # specify allele "a0" for appropriate genotype encoding with pandas-plink
# bim/fam/bed PLINK genotypes fileset
genotypes: "./data/spikein/spikein_s6_add100_dom100_rec100_epi100_seed42/ukb_c1-22_qc_simset.real_plus_syn"
phenotype_path: "./data/height_adj.pheno" # phenotype labels file
phenotype_name: "height_adj_z" # phenotype labels column header (column in phenotype_path)
qc_snplist_path: null # path to list of SNPs to use for training 
qc_idlist_path: "./data/ukb_c1-22_qc.id" # set of all sample IDs to consider (from PLINK)
train_ids_path: "./data/train_rs1234.id" # train IDs path
val_ids_path: "./data/val_rs1234.id" # val IDs path
test_ids_path: "./data/test_rs1234.id" # test IDs path
use_decoys: false # flag controlling whether to train with decoys or not.
decoy_seed: null # int decoy seed (must use with decoy model training)
model_name: "VanillaNet" # name of model class
use_x_transforms: false # x data transformation
use_y_transforms: false # y data transformation
missing_policy: null # determines how to handle missing X data
batch_size: 128 # batch size
optimizer: "adam" # optimizer [adamw | adam | sgd]
loss_function: "mae" # loss function [mse | mae | bce]
activation_function: 'relu'  # relu | leakyrelu | silu | gelu | elu | selu | none
learning_rate: 1e-6 # learning rate
num_epochs: 100 # number of epochs for training
l2_lambda: 0.001 # l2 regularization factor
l1_lambda: 0.0 # l1 regularization factor
dropout_rate: 0.25 # dropout rate
grad_clip: null # gradient clipping value (no gradient clipping when null)
# DNN hidden layers units
hidden_layers:
- 1000
- 200
- 50
rows_per_chunk: 20480 # number of rows per dask chunk
cols_per_chunk: 20480 # number of cols per dask chunk
num_workers: 0
precision: 'fp32' # amp_fp16 | amp_bf16 | fp32
es_enabled: false # early stopping enabled
es_patience: 25 # early stopping patience
es_min_delta: 0.0 # early stopping min improvement
es_min_epochs: 50 # min epochs before considering early stopping
es_mode: "max" # min | max
torch_compile: false # whether to use torch.compile()
torch_compile_mode: null # default | reduce-overhead | max-autotune
monitor_metric: "r2" # the metric to monitor during training (on validation set)
save_summary_steps: 10 # controls the frequency in which training metrics are stored 
restore_file: null # resume training from checkpoint file
```

- For models used to evaluate attribution recall, make sure the `genotypes` key is assigned to the real + spike-in synthetic genotype data path. 
- For models used to evaluate attribution precision, make sure that `use_decoys` is set to true and a `decoy_seed` value is provided. Use the the path to the true genotypes (after QC). Decoy genotypes are generated on-the-fly.
- For models used to evaluate stability/consistency, create a config file for each ensemble model (one model per config file; each config file should be in its own unique model directory).

Training is executed with `train.py` with the path to the model containing the experiment's configuration file specified with the `--model_dir` argument.

```bash
$ python3 -u train.py --model_dir ./experiments/dnn_spikein
```

To run it in the background (i.e., maintain training in the event of lost connection):

```bash
$ nohup python3 -u train.py --model_dir ./experiments/dnn_spikein &
```

## Generating Attributions or Feature Importance Scores

Run `interpretation.py` to generate SNP-wise feature importance scores across several interpretation algorithms (Saliency, Gradient SHAP, DeepLIFT, Integrated Gradients) in both the presence and absence of smoothing via SmoothGrad. Importance scores are generated and aggregated from only the samples found in the testing set. 

```bash
$ python3 -u attribution.py --model_dir ./experiments/dnn_spikein
```

## Interpretation Benchmarking

Each benchmarking notebook produces both quantitative metrics and visual summaries of the interpretation benchmarks.

- **Attribution Recall** (`benchmark_recall.ipynb`)
  - Measures the ability of each interpretation method to recover known causal (spike-in) SNPs.
  - Evaluates recall across effect types (additive, dominant, recessive, epistatic).
  - Line plots compare recall variability between smoothed and non-smoothed methods across quantiles.
- **Attribution Precision** (`benchmark_precision.ipynb`)\
  - Computes precision by measuring the ability to distinguish real SNPs from decoys.
  - Line plot compares precision variability for smoothed vs. non-smoothed methods across quantiles.
- **Ensemble Stability** (`benchmark_stability.ipynb`)
  - Quantifies consistency of feature attributions across independently trained ensemble members.
  - Computes per-SNP relative standard deviation (RSD) of attribution scores; median RSD summarizes stability, and median absolute deviation captures dispersion.
  - Violin plots illustrate stability distributions across interpretation methods.