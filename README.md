
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
    epistatic: 50   # 2 columns per pair => 100 epistatic columns total

ranges:
  add_rho_range: [0.025, 0.055]
  dom_rho_range: [0.018, 0.030]
  rec_rho_range: [0.035, 0.065]
  epi_rho_range: [0.045, 0.070]

  maf_ranges:
    dominant:  [0.050, 0.080]  # carriers ~= 1 - (1-p)^2 -> about 9.8–14.8%
    recessive: [0.060, 0.120]  # n22 ~= p^2 * N -> about 1150–4600 at N~3.2e5
    epistatic: [0.080, 0.200]  # dense joint table, few sparse cells

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

## DNN Training

- `ukb_data.py` for data management


## DNN Interpretation


## Interpretation Benchmarking