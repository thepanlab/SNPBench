# datasets.py
import numpy as np
import dask.array as da
import torch

def resolve_imputers_and_dtype(imputers, n_snps):
    """
    Return (impute_arr or None and the realize_dtype)
    - If imputers is None: fill NaNs with 0 -> uint8
    - If imputers is provided: must be 1D of length n_snps with no NaNs.
    - If all values are integer-like and in {0,1,2} -> uint8, else -> float32.
    """
    # when imputers is None, fill NaNs with 0
    if imputers is None:
        return None, np.uint8
    
    imp = np.asarray(imputers)
    if imp.ndim != 1 or imp.shape[0] != n_snps or np.isnan(imp).any():
        # invalid impute arr
        raise ValueError(
            f"imputers must be 1D of length {n_snps} with no NaNs; got shape={imp.shape}"
        )
    # allow 0.0/1.0/2.0 etc. and keep uint8 if so
    integer_like = np.allclose(imp, np.rint(imp))
    in_012 = np.all((imp >= 0) & (imp <= 2))
    if integer_like and in_012:
        # impute arr is valid and can realize uint8 dtype
        return imp.astype(np.uint8, copy=False), np.uint8
    else:
        # impute arr is valid but must realize float32 dtype 
        # bc impute arr has non-integer-like (or non-{0,1,2} values)
        return imp.astype(np.float32, copy=False), np.float32


class DaskIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, X, y, batch_size, shuffle=False, drop_last=False, imputers=None, 
                 transform_x=None, transform_y=None, use_decoys=False, decoy_seed=97):
        assert X.ndim == 2, "X must be 2D (n_samples, n_snps)"
        assert len(y) == X.shape[0], "X and y must have same #samples"
        assert batch_size > 0
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.transform_x = transform_x
        self.transform_y = transform_y
        self.use_decoys = bool(use_decoys)
        self.decoy_seed = int(decoy_seed) if self.use_decoys else None
        self.imputers, self.rlz_dtype = resolve_imputers_and_dtype(imputers, X.shape[1])

        # Can rechunk here if needed and not already done beforehand
        # (already rechunking in UKBData())
        # rechunk_factor = 20
        # self.X = self.X.rechunk((1024*rechunk_factor, 1024*rechunk_factor))
        
        # row chunk sizes
        self.row_chunk_sizes = list(self.X.chunks[0])
        # number of chunks
        self.num_row_chunks = len(self.row_chunk_sizes)
        # chunk indices
        self.row_chunk_indices = list(range(self.num_row_chunks))
        # chunk start offsets
        self.row_offsets = np.cumsum([0]+self.row_chunk_sizes[:-1])
        self.decoy_perm_cache = {}

    def realize_rows(self, chunk_idx):
        # slice dask array for target chunk
        view = self.X.blocks[chunk_idx, :]
        if self.imputers is None:
            # fill NaN with 0 and realize with self.rlz_dtype
            view = da.nan_to_num(view).astype(self.rlz_dtype) # fill NaNs with 0 by default
        else:
            # broadcast imputers with shape (1, n_snps) across rows
            view = da.where(da.isnan(view), self.imputers[None, :], view).astype(self.rlz_dtype)
        return view.compute()

    def __iter__(self):
        # split row-chunk indices across workers for parallel loading
        worker_info = torch.utils.data.get_worker_info()
        all_indices = self.row_chunk_indices
        if worker_info is not None:
            per_worker = int(np.ceil(len(all_indices) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(all_indices))
            iter_indices = all_indices[iter_start:iter_end]
        else:
            iter_indices = all_indices.copy()
        
        # shuffle chunk visitation order
        if self.shuffle:
            np.random.shuffle(iter_indices)
        
        # iterate dask row chunks
        for chunk_idx in iter_indices: 
            # use pre-computed chunk offsets
            start = self.row_offsets[chunk_idx] 
            end = start + self.row_chunk_sizes[chunk_idx]
            # print(f"[__iter__] chunk {chunk_idx+1}/{self.num_row_chunks} [{start}:{end}]")

            # realize X (dask -> np) and y (np)
            Xv_np = self.realize_rows(chunk_idx) # slow
            yv_np = self.y[start:end]
            
            # convert to torch tensors
            Xv = torch.from_numpy(Xv_np)
            yv = torch.as_tensor(yv_np)

            # apply decoy genotypes for the current chunk
            if self.use_decoys:
                # check if perm for this chunk_idx is cached
                if chunk_idx not in self.decoy_perm_cache:
                    # random generator with chunk-dependent seed for reproducibility
                    rng_dec = torch.Generator().manual_seed(self.decoy_seed + int(chunk_idx))
                    # cache the permuation
                    self.decoy_perm_cache[chunk_idx] = torch.randperm(Xv.size(0), generator=rng_dec)

                # apply the cached permutation for the current chunk_idx 
                # (preserves decoy assignment across epochs)
                decoy_perm = self.decoy_perm_cache[chunk_idx]
                Xd = Xv[decoy_perm]

            # within-chunk shuffle
            if self.shuffle:
                perm = torch.randperm(Xv.size(0))
                Xv = Xv[perm]
                yv = yv[perm]
                if self.use_decoys:
                    # apply shuffle perm to decoys as well
                    Xd = Xd[perm]
            
            # iterate minibatches
            n_samples_in_chunk = Xv.size(0)
            for i in range(0, n_samples_in_chunk, self.batch_size):
                # slice X and y minibatch
                Xb = Xv[i:i + self.batch_size]
                yb = yv[i:i + self.batch_size]
                
                # drop last
                if Xb.size(0) < self.batch_size and self.drop_last:
                    continue
                
                # append decoys
                if self.use_decoys:
                    Xb_dec = Xd[i:i + Xb.size(0)]
                    Xb = torch.cat([Xb, Xb_dec], dim=1)
                
                # cast the X minibatch to float32 before transforms
                Xb = Xb.float()

                # apply transformations to the minibatch
                if self.transform_x is not None:
                    Xb = self.transform_x(Xb)
                if self.transform_y is not None:
                    yb = self.transform_y(yb)

                # ensure yb is 2D for even single output neuron
                if yb.ndim == 1:
                    # [B] -> [B,1]
                    yb = yb.unsqueeze(dim=1)
                yield Xb, yb
            
            del Xv, yv
            if self.use_decoys:
                del Xd
