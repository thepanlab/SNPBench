"""
UKBData: Data loader/manager for UKB PLINK genotypes & phenotype/covariates.

Overview:
- Loads PLINK genotypes (merged base OR per-chrom) via pandas_plink (dask-backed xarray).
- Aligns phenotype (and optional covariates) to genotype sample order.
- Builds a reusable SNP filter (QC list and/or GWAS top-N).
- Provides split-aware sample filtering (train/val/test/all) from ID lists.
- Returns a Partition object with dask Array X, numpy y, and handy ID tables.
- Optional chunk optimization for dask arrays (done once per split view).

Expected configuration keys (optional unless marked *):
* genotypes: str               # merged base path OR directory with chr*.{bed,bim,fam}
  genotypes_by_chrom: bool     # default False; if True, expects per-chrom files
  chrom_filebase: str          # default "chr{}"; joined with genotypes
  ref_allele: str              # default "a0" (pandas_plink ref)
* phenotype_path: str          # TSV with 'IID' column and target phenotype column
  phenotype_name: str          # column name for phenotype; if missing, last column
  covariate_path: str          # TSV with IID (and optional FID) + covariate cols
  qc_snplist_path: str         # file with one SNP ID per line (keep-only)
  snp_filter_path: str         # GWAS-style TSV with P and SNP/ID (optional TEST=='ADD')
  n_snps: int                  # take top-N by P (applied after QC list)
  qc_idlist_path: str          # sample QC keep list (two-col FID IID or IID-indexed)
  train_ids_path: str          # IID-indexed TSV
  val_ids_path: str            # IID-indexed TSV
  test_ids_path: str           # IID-indexed TSV
  rows_per_chunk: int          # default 51200
  cols_per_chunk: int          # default 20480
  verbose: bool                # default False
"""

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats
import xarray as xr
import dask.array as da
import pandas_plink as pdp
import yaml
from tqdm import tqdm

# partition view container
class Partition:
    """
    Container returned by UKBData.get_partition(...).
    Attributes
    ---------
    split        : one of {"train","val","test","all"}
    X            : dask array (rows = samples, cols = SNPs kept)
    y            : numpy float32 vector aligned to X rows
    row_idx_abs  : numpy int64 absolute indices into the full genotype order
    sample_df    : pandas DataFrame with columns FID, IID (index=row_idx_abs)
    variant_df   : pandas DataFrame with column SNP (kept SNPs in order)
    """
    def __init__(self, split, X, y, row_idx_abs, sample_df=None, variant_df=None):
        self.split = split
        self.X = X
        self.y = y
        self.row_idx_abs = row_idx_abs
        self.sample_df = sample_df
        self.variant_df = variant_df

    @property
    def sample_ids(self):
        if self.sample_df is None or "IID" not in self.sample_df:
            return np.array([], dtype=str)
        return self.sample_df["IID"].values

    @property
    def snp_ids(self):
        if self.variant_df is None or "SNP" not in self.variant_df:
            return np.array([], dtype=str)
        return self.variant_df["SNP"].values

# Data utilities 
def print_kv(rows, print_func=print):
    pad = 28
    for k, v in rows:
        print_func(f"{(k + ':'): <{pad}}  {v}")

def load_yaml(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config YAML not found: {p}")
    return yaml.safe_load(p.read_text()) or {}

def load_csv(path, **kwargs):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return pd.read_csv(p, **kwargs)

def read_plink_merged(base_path, ref="a0"):
    """
    Read a merged PLINK fileset (base.[bed,bim,fam]) into an xarray Dataset.
    """
    base = Path(base_path).expanduser().resolve()
    if base.suffix in (".bed", ".bim", ".fam"):
        base = base.with_suffix("")
    base_str = str(base)
    bed = Path(base_str + ".bed")
    bim = Path(base_str + ".bim")
    fam = Path(base_str + ".fam")
    missing = [str(p) for p in (bed, bim, fam) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "PLINK trio not found. Missing:\n  " + "\n  ".join(missing) +
            f"\n(Coerced base: {base})"
        )

    return pdp.read_plink1_bin(bed=str(bed), bim=str(bim), fam=str(fam), ref=ref, verbose=False)

def read_plink_by_chrom(genotypes_dir, filebase="chr{}", ref="a0"):
    """
    Load PLINK genotype files split by chromosome in a directory and concat by variant.
    Expects files like chr1.bed/.bim/.fam ... chr22.* in genotypes_dir.
    """
    genotypes_dir = Path(genotypes_dir)
    ds_list = []
    for chrom in range(1, 23):
        base = genotypes_dir / filebase.format(chrom)
        bed, bim, fam = base.with_suffix(".bed"), base.with_suffix(".bim"), base.with_suffix(".fam")
        if not (bed.exists() and bim.exists() and fam.exists()):
            raise FileNotFoundError(f"Missing per-chrom files: {base}.[bed/bim/fam]")
        Gc = pdp.read_plink1_bin(str(bed), str(bim), str(fam), verbose=False, ref=ref)
        # SNP IDs <rs#> as the variant coordinate for clean concat
        Gc = Gc.assign_coords(variant=Gc.variant["snp"].values)
        ds_list.append(Gc)
    return xr.concat(ds_list, dim="variant")


# UKBData dataset manager
class UKBData:
    """
    Dataset manager for UKB PLINK genotypes + phenotypes.

    Notes
    -----
    - Uses pandas_plink -> xarray (dask-backed) so X-ops are lazy until .compute().
    - Aligns by IID; FID is passed through when available from PLINK.
    - SNP filtering is applied once to build a boolean mask reused across splits.
    - Sample filtering uses: non-missing y (and covs, if provided), optional QC list,
      and optional split ID lists (train/val/test).
    """

    def __init__(self, config, verbose=None, print_func=print):
        self.cfg = dict(config)  # shallow copy
        self.verbose = bool(self.cfg.get("verbose", False)) if verbose is None else bool(verbose)
        self.print_func = print_func
        self.validate_config()
        # Genotypes
        ref = str(self.cfg.get("ref_allele", "a0"))
        by_chrom = bool(self.cfg.get("genotypes_by_chrom", False))
        # relative path -> absolute path
        self.cfg["genotypes"] = str(
            Path(self.cfg["genotypes"]
        ).expanduser().resolve())

        if by_chrom:
            self.G = read_plink_by_chrom(
                self.cfg["genotypes"],
                filebase=str(self.cfg.get("chrom_filebase", "chr{}")),
                ref=ref,
            )
        else:
            self.G = read_plink_merged(self.cfg["genotypes"], ref=ref)

        # Basic IDs from PLINK (IID always present; FID may be missing)
        self.sample_iid = self.G.sample["iid"].values.astype(str)
        self.sample_fid = (
            self.G.sample["fid"].values.astype(str) if "fid" in self.G.sample else self.sample_iid.copy()
        )
        self.snp_ids_full = self.G.variant["snp"].astype(str).values
        self.n_samples, self.n_snps_full = len(self.sample_iid), len(self.snp_ids_full)

        if self.verbose:
            self.print_func("[UKBData] Genotypes:")
            print_kv([
                ("files", f"{self.cfg['genotypes']} (by_chrom={by_chrom})"),
                ("shape (samples, snps)", f"({self.n_samples:,}, {self.n_snps_full:,})"),
                ("dtype", getattr(self.G.data, "dtype", "n/a")),
                ("chunksize", getattr(self.G.data, "chunksize", "n/a")),
            ], print_func=self.print_func)

        # Phenotypes & covariates (aligned to genotype order)
        self.y_all = self.load_and_align_phenotypes()  # float32 vector
        self.covs, self.cov_cols = self.load_and_align_covariates()  # (array or None, list of names)
        # Variant mask (computed once; reused across splits)
        self.variant_mask, self.variant_ids_kept = self.build_variant_mask()
        # Split index cache (absolute indices into full order)
        self.split_idx_cache = {}  # keys: 'train','val','test'

    @classmethod
    def from_yaml(cls, cfg_path, verbose=None):
        return cls(load_yaml(cfg_path), verbose=verbose)

    def validate_config(self):
        """
        Check that required config keys are present and files exist.
        """
        errs = []
        warn = []
        # Required-ish
        if "genotypes" not in self.cfg or not str(self.cfg["genotypes"]).strip():
            errs.append("Missing required key: genotypes")
        # Filesystem checks (best-effort)
        genotypes = self.cfg.get("genotypes")
        by_chrom = bool(self.cfg.get("genotypes_by_chrom", False))
        if genotypes:
            p = Path(genotypes)
            if by_chrom:
                if not p.exists() or not p.is_dir():
                    errs.append(f"genotypes '{p}' does not exist or is not a directory (expected per-chrom files).")
            else:
                bed = Path(f"{p}.bed"); bim = Path(f"{p}.bim"); fam = Path(f"{p}.fam")
                if not (bed.exists() and bim.exists() and fam.exists()):
                    errs.append(f"Missing merged PLINK trio: {p}.[bed|bim|fam]")
        # phenotype recommended but optional
        if "phenotype_path" in self.cfg:
            ph = Path(self.cfg["phenotype_path"])
            if not ph.exists():
                errs.append(f"phenotype_path not found: {ph}")
        else:
            warn.append("No phenotype_path provided; y will be zeros (allowed).")
        # optional inputs (only check if provided)
        for key in ("covariate_path","qc_snplist_path","snp_filter_path",
                    "qc_idlist_path","train_ids_path","val_ids_path","test_ids_path"):
            if key in self.cfg and self.cfg.get(key):
                p = Path(self.cfg[key])
                if not p.exists():
                    errs.append(f"{key} not found: {p}")
        n_snps = self.cfg.get("n_snps")
        if n_snps is not None:
            try:
                n_val = int(n_snps)
                if n_val <= 0:
                    errs.append("n_snps must be a positive integer if provided.")
            except Exception:
                errs.append("n_snps must be an integer if provided.")
        if errs:
            msg = "Config validation failed:\n- " + "\n- ".join(errs)
            if warn:
                msg += "\n\nNotes:\n- " + "\n- ".join(warn)
            raise ValueError(msg)
        if self.verbose and warn:
            self.print_func("[UKBData] Config notes:")
            for w in warn:
                self.print_func(f" - {w}")

    # ------------------------ 
    #  loaders & builders
    # ------------------------ 

    def load_and_align_phenotypes(self):
        pheno_path = self.cfg.get("phenotype_path")
        if not pheno_path:
            if self.verbose:
                self.print_func("[UKBData] No phenotype file provided; using zeros.")
            return np.zeros(self.n_samples, dtype=np.float32)

        df = load_csv(pheno_path, sep="\t", dtype={"IID": str})
        if "IID" in df.columns:
            df = df.set_index("IID")

        if self.cfg.get("phenotype_name"):
            col = self.cfg["phenotype_name"]
            if col not in df.columns:
                raise KeyError(f"phenotype_name '{col}' not found in {pheno_path}")
            s = df[col].astype(np.float32)
        else:
            s = df.iloc[:, -1].astype(np.float32)

        y = s.reindex(self.sample_iid, fill_value=np.nan).values.astype(np.float32)
        if self.verbose:
            self.print_func("[UKBData] Phenotypes:")
            print_kv([
                ("file", pheno_path),
                ("column", self.cfg.get("phenotype_name", f"<last: {s.name}>")),
                ("n_samples in file", f"{len(s):,}"),
                ("n_missing after align", f"{np.isnan(y).sum():,}"),
            ], print_func=self.print_func)

        return y

    def load_and_align_covariates(self):
        cov_path = self.cfg.get("covariate_path")
        if not cov_path:
            if self.verbose:
                self.print_func("[UKBData] No covariates.")
            return None, []

        df = load_csv(cov_path, sep="\t", dtype={"IID": str})
        if "IID" in df.columns:
            df = df.set_index("IID")
        df = df.drop(columns=["FID"], errors="ignore")
        df = df.reindex(self.sample_iid)
        cov_cols = list(df.columns)
        covs = df.values.astype(np.float32)
        if self.verbose:
            self.print_func("[UKBData] Covariates:")
            print_kv([
                ("file", cov_path),
                ("fields", cov_cols),
                ("n_missing", int(np.isnan(covs).sum())),
            ], print_func=self.print_func)

        return covs, cov_cols

    def build_variant_mask(self):
        snps = self.snp_ids_full
        keep = np.ones_like(snps, dtype=bool)

        # Keep-only QC SNP list (if provided)
        snp_qc_path = self.cfg.get("qc_snplist_path")
        if snp_qc_path:
            qc_ids = load_csv(snp_qc_path, header=None, dtype={0: str}).squeeze().astype(str).values
            qc_set = set(qc_ids)
            keep = np.fromiter((s in qc_set for s in snps), dtype=bool, count=len(snps))

        # Optional GWAS top-N filter (applied after QC restriction if both provided)
        filt_path = self.cfg.get("snp_filter_path")
        top_n = self.cfg.get("n_snps")
        if filt_path and top_n:
            gwas = load_csv(filt_path, sep="\t")
            if "TEST" in gwas.columns:
                gwas = gwas[gwas["TEST"] == "ADD"]
            snp_col = "ID" if "ID" in gwas.columns else "SNP"
            top_ids = gwas.nsmallest(int(top_n), "P")[snp_col].astype(str).values
            keep &= np.isin(snps, top_ids)

        kept_ids = snps[keep]

        if self.verbose:
            self.print_func("[UKBData] Variant mask:")
            print_kv([
                ("kept variants", f"{keep.sum()}/{len(snps)}"),
                ("gwas top_n applied", bool(filt_path and top_n)),
                ("qc list applied", bool(snp_qc_path)),
            ], print_func=self.print_func)
        return keep, kept_ids

    def load_qc_iids(self):
        path = self.cfg.get("qc_idlist_path")
        if not path:
            return None
        # Try "FID IID" two-column TSV (skip header), else IID-indexed TSV
        try:
            df = load_csv(path, sep="\t", names=["FID", "IID"], skiprows=1, dtype={"FID": str, "IID": str})
            return df["IID"].values.astype(str)
        except Exception:
            df = load_csv(path, sep="\t", dtype={"IID": str}, index_col="IID")
            return df.index.values.astype(str)

    def split_ids_from_path(self, key):
        p = self.cfg.get(key)
        if not p:
            return None
        df = load_csv(p, sep="\t", dtype={"IID": str}, index_col="IID")
        return df.index.values.astype(str)

    def ids_for_split(self, split):
        key = {"train": "train_ids_path", "val": "val_ids_path", "test": "test_ids_path"}[split]
        return self.split_ids_from_path(key)

    def sample_mask_for_split(self, split):
        iids = self.sample_iid
        keep = np.ones_like(iids, dtype=bool)

        # require non-missing y
        keep &= ~np.isnan(self.y_all)
        # require non-missing covariates (if provided)
        if self.covs is not None:
            keep &= ~np.any(np.isnan(self.covs), axis=1)
        # apply sample QC list
        qc_iids = self.load_qc_iids()
        if qc_iids is not None:
            keep &= np.isin(iids, qc_iids)
        # apply split lists for train/val/test
        if split in ("train", "val", "test"):
            split_iids = self.ids_for_split(split)
            if split_iids is not None:
                keep &= np.isin(iids, split_iids)
        return keep

    def get_split_indices(self):
        """
        Returns (train_idx, val_idx, test_idx) as absolute integer indices into the full order.
        Cached after first call.
        """
        if not all(k in self.split_idx_cache for k in ("train", "val", "test")):
            for split in ("train", "val", "test"):
                mask = self.sample_mask_for_split(split)
                self.split_idx_cache[split] = np.flatnonzero(mask).astype(np.int64)
        return (
            self.split_idx_cache["train"],
            self.split_idx_cache["val"],
            self.split_idx_cache["test"],
        )

    @property
    def all_sample_df(self):
        """PLINK-friendly full sample table."""
        return pd.DataFrame({"FID": self.sample_fid, "IID": self.sample_iid})

    @property
    def all_snp_df(self):
        return pd.DataFrame({"SNP": self.snp_ids_full})

    def snp_ids(self, kept_only=True):
        return self.variant_ids_kept if kept_only else self.snp_ids_full

    def sample_ids(self, split="all"):
        if split == "all":
            mask = self.sample_mask_for_split("all")
            return self.sample_iid[mask]
        if split not in self.split_idx_cache:
            self.get_split_indices()
        idx = self.split_idx_cache[split]
        return self.sample_iid[idx]
    
    # partition view
    def chunks_match_target(self, chunks, target):
        if not target or target <= 0 or not chunks:
            return True
        # all but last equal target; last <= target
        return all(c == target for c in chunks[:-1]) and (chunks[-1] <= target)

    def get_partition(
        self,
        split="train",
        optimize_chunks=True,
        row_chunk_target=None,
        col_chunk_target=None,
        return_tables=True,
        verbose=None,
    ):
        """
        Build a lazily-indexed view of X plus aligned y and metadata for a split.

        Returns a Partition with:
          - X: dask.array (rows = selected samples, cols = kept variants)
          - y: np.float32 aligned to rows
          - row_idx_abs: absolute indices into full genotype order
          - sample_df (FID, IID) and variant_df (SNP) if return_tables=True
        """
        verbose = False if verbose is None else bool(verbose)

        # row indices
        if split == "all":
            mask = self.sample_mask_for_split("all")
            row_idx_abs = np.flatnonzero(mask).astype(np.int64)
        else:
            if split not in self.split_idx_cache:
                self.get_split_indices()
            row_idx_abs = self.split_idx_cache[split]

        # xarray view and dask array
        Gp = self.G.isel(sample=row_idx_abs, variant=self.variant_mask).copy()
        X = Gp.data
        y = self.y_all[row_idx_abs].astype(np.float32)

        # target chunk sizes
        default_row_chunk_size, default_col_chunk_size = 51200, 20480
        row_tgt = int(row_chunk_target) if row_chunk_target is not None else int(self.cfg.get("rows_per_chunk", default_row_chunk_size))
        col_tgt = int(col_chunk_target) if col_chunk_target is not None else int(self.cfg.get("cols_per_chunk", default_col_chunk_size))

        # optional rechunk
        rechunked = False
        if optimize_chunks and hasattr(X, "chunks"):
            r_chunks = X.chunks[0]
            c_chunks = X.chunks[1]
            spec = {}
            if not self.chunks_match_target(tuple(int(c) for c in r_chunks), row_tgt):
                spec[0] = row_tgt
            if not self.chunks_match_target(tuple(int(c) for c in c_chunks), col_tgt):
                spec[1] = col_tgt
            if spec:
                X = X.rechunk(spec)
                rechunked = True

        sample_df = None
        variant_df = None
        if return_tables:
            sample_df = pd.DataFrame(
                {"FID": self.sample_fid[row_idx_abs], "IID": self.sample_iid[row_idx_abs]},
                index=row_idx_abs,
            )
            variant_df = pd.DataFrame({"SNP": self.variant_ids_kept})

        if verbose:
            self.print_func(f"[UKBData] '{split}' partition:")
            print_kv([
                ("X.shape", X.shape),
                ("X.dtype", X.dtype),
                ("X rechunked", str(rechunked)),
                ("X.chunksize", getattr(X, "chunksize", "n/a")),
                ("y.shape", y.shape),
                ("rows kept", f"{len(row_idx_abs):,}"),
                ("cols kept", f"{int(self.variant_mask.sum()):,}"),
            ], print_func=self.print_func)

        return Partition(split, X, y, row_idx_abs, sample_df=sample_df, variant_df=variant_df)

    def get_maf(self, split="train"):
        """
        MAF per kept SNP for the given split.
        Missing values are considered as 0 for frequency calculation.
        """
        part = self.get_partition(split, optimize_chunks=True, return_tables=False, verbose=False)
        X_filled = da.nan_to_num(part.X)                 # fill NaNs with 0
        pA = da.mean(X_filled / 2.0, axis=0).compute()  # allele freq
        maf = np.minimum(pA, 1.0 - pA)
        return maf

    def variant_means(self, split="train"):
        """
        Mean genotype per kept SNP for the given split (ignores NaNs).
        """
        part = self.get_partition(split, optimize_chunks=False, return_tables=False, verbose=False)
        col_means = da.nanmean(part.X, axis=0).compute()
        return np.nan_to_num(col_means, nan=0.0)
    
    def variant_modes(self, split="train", snap_to_int=False):
        """
        Column-wise mode genotype for the given split, ignoring NaNs.
        Uses Dask reductions.

        Args:
            split (str): data split ("train"/"val"/"test"/"all").
            snap_to_int (bool): if True, round values to nearest {0,1,2}
                before counting (floats->ints).
        Returns:
            np.ndarray[float32] of shape (n_snps,)
        """
        part = self.get_partition(split, optimize_chunks=True,
                                return_tables=False, verbose=False)
        X = part.X  # Dask array, shape (n_samples, n_snps)
        if snap_to_int:
            # Snap to {0,1,2} (leave NaNs alone)
            X = da.where(da.isnan(X), X, da.clip(da.rint(X), 0, 2))
        # Count occurrences of 0, 1, 2 per column (ignoring NaNs).
        c0 = da.sum(X == 0, axis=0)
        c1 = da.sum(X == 1, axis=0)
        c2 = da.sum(X == 2, axis=0)
        # Stack counts so index 0->genotype 0, 1->genotype 1, 2->genotype 2.
        counts = da.stack([c0, c1, c2], axis=0)
        # Mode is the argmax over the first axis.
        # (ties break toward the smallest genotype)
        modes = da.argmax(counts, axis=0).astype("float32")
        # All-NaN columns produce counts [0,0,0] -> argmax=0 => mode=0.0
        return modes.compute()


    def get_impute_array(self, split='train', policy='mean'):
        """
        Returns a per-SNP imputation vector for the given split.
        policy is one of {'mean','means','mode','modes'}
        """
        if policy in ['mean', 'means']:
            arr = self.variant_means(split)
        elif policy in ['mode', 'modes']:
            arr = self.variant_modes(split)
        else:
            raise ValueError(f"Invalid imputation policy: {policy}")
        if arr.ndim != 1 or arr.shape[0] != int(self.variant_mask.sum()):
            raise RuntimeError(f"Imputer array has wrong shape: {arr.shape}")
        return arr

    def genotype_transforms_stats(self, split='train', min_var=1e-6):
        """
        Compute per-SNP mean and scale (TRAIN-based) for standardizing genotype dosages.
        Center at 2p and scale by sqrt(2p(1-p)), with clipping to avoid div-by-zero.
        Returns (mean_np, scale_np) as float32 arrays of length P (kept SNPs).
        """
        part = self.get_partition(split, optimize_chunks=False, return_tables=False, verbose=False)
        X = part.X
        mean_g = da.nanmean(X, axis=0)
        p = mean_g / 2.0
        var = 2.0 * p * (1.0 - p)
        scale = da.sqrt(da.maximum(var, min_var))
        mean = 2.0 * p
        return mean.astype(np.float32).compute(), scale.astype(np.float32).compute()

    def write_plink_id_list(self, split, out_path):
        """
        Write a two-column FID IID file suitable for PLINK --keep/--remove.
        Only includes rows kept after filters for the requested split.
        """
        part = self.get_partition(split, optimize_chunks=False, return_tables=True)
        df = part.sample_df.copy() if part.sample_df is not None else pd.DataFrame({"FID": [], "IID": []})
        df.to_csv(out_path, sep="\t", header=True, index=False)

    def write_snp_list(self, out_path, kept_only=True, with_header=False):
        snps = self.variant_ids_kept if kept_only else self.snp_ids_full
        pd.DataFrame({'SNP': snps}).to_csv(out_path, sep="\t", header=with_header, index=False)

    def xarray_view(self, split="all"):
        """
        Return the xarray Dataset slice (for advanced workflows) rather than just dask array.
        """
        if split == "all":
            mask = self.sample_mask_for_split("all")
            row_idx_abs = np.flatnonzero(mask).astype(np.int64)
        else:
            if split not in self.split_idx_cache:
                self.get_split_indices()
            row_idx_abs = self.split_idx_cache[split]
        return self.G.isel(sample=row_idx_abs, variant=self.variant_mask)
