import logging
import scipy
from scipy.spatial.distance import cdist, pdist
import numpy as np
from torch.utils.data import Dataset
from src.utils.util import *
import scanpy as sc
import pandas as pd
from os import path as osp
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit
import time
import torch
import gc
from src.utils.states import Dynamic_neigh_level
from tifffile import imread
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import torchvision
from torchvision.transforms import v2
from torchvision import transforms as v1
import torch.distributed as dist
try:
    import cuml
    import cupy
except:
    pass

class GraphDataset(Dataset):
    """Graph dataset object, loading dataset in a batch-wise manner.

    Args:
        k (int): k nearest neighbor used in neighbors.
        rec_neigh_num (boolean): whether to use the reconstructing
            neighbors strategy.

    """
    def __init__(
        self,
        data_dir: str = "",
        dataset_dir: str = "",
        data_type: str = "10x",
        in_type: str = "raw",
        out_type: str = "raw",
        compared_type: str = "raw",
        count_key=None,
        annotation_key=None,
        num_classes="auto",
        data_file_name=None,
        num_hvg: int = 2048,
        lib_norm=True,
        n_pc: int = 50,
        max_dynamic_neigh=False,
        dynamic_neigh_level=Dynamic_neigh_level.unit,
        unit_fix_num=None,
        unit_dynamic_num=None,
        unit_dynamic_candidate_num=None,
        k: int = 18,
        resolution="bulk",
        rec_neigh_num=1,
        rec_mask_neigh_threshold=None,
        use_image=False,
        img_file_name=None,
        img_coord_key="spatial",
        img_patch_origin_loc="center",
        img_batch_size=256,
        bin_size=50,
        patch_size_factor=5,
        use_cell_mask=False,
        keep_tiles=False,
        device="cpu",
        seed: int = 42,
        test_with_gt_sp: bool = False,
        forward_neigh_num=0,
        exchange_forward_neighbor_order=False,
        sample_id: str = "sample_id",
        multi_slides=False,
        img_id: str = "",
        z_scale: float = 2.,
        resample_to=None,
        use_ddp=False,
        weighted_neigh=False,
        supervise_cat=False,
        n_jobs=-1,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.dataset_dir = dataset_dir
        self.data_type = data_type
        self.in_type = in_type
        self.out_type = out_type
        self.compared_type = compared_type
        self.count_key = count_key
        self.annotation_key = annotation_key
        self.num_classes = num_classes
        self.data_file_name = data_file_name
        self.num_hvg = num_hvg
        self.lib_norm = lib_norm
        self.n_pc = n_pc
        self.max_dynamic_neigh = max_dynamic_neigh
        self.dynamic_neigh_level = dynamic_neigh_level
        self.unit_fix_num = unit_fix_num
        self.unit_dynamic_num = unit_dynamic_num
        self.unit_dynamic_candidate_num = unit_dynamic_candidate_num
        self.k = k
        self.rec_neigh_num = rec_neigh_num
        if self.rec_neigh_num == 'half':
            self.rec_neigh_num = self.k // 2
        self.rec_mask_neigh_threshold = rec_mask_neigh_threshold
        self.supervise_cat = supervise_cat
        self.use_image = use_image
        self.use_cell_mask = use_cell_mask
        self.keep_tiles = keep_tiles
        self.img_file_name = img_file_name
        self.img_coord_key = img_coord_key
        self.img_patch_origin_loc = img_patch_origin_loc
        self.img_batch_size = img_batch_size
        self.bin_size = bin_size
        self.patch_size_factor = patch_size_factor
        self.use_ddp = use_ddp
        self.device = device
        self.seed = seed
        self.test_with_gt_sp = test_with_gt_sp
        self.forward_neigh_num = forward_neigh_num
        if self.forward_neigh_num == 'half':
            self.forward_neigh_num = self.k // 2
        self.exchange_forward_neighbor_order = exchange_forward_neighbor_order
        self.weighted_neigh = weighted_neigh
        self.rng = np.random.default_rng(seed)
        self.hexagonal_num_list = [n * (n + 1) * 3 for n in range(1, 100)]
        self.square_num_list = [n * (n + 1) * 4 for n in range(1, 100)]
        self.n_jobs = n_jobs
        self.resolution = resolution
        self.sample_id = str(sample_id)
        self.multi_slides = multi_slides
        self.img_id = str(img_id)
        self.z_scale = z_scale
        self.resample_to = resample_to
        self.full_adata = None

        if self.data_type == '10x':
            self.get_10x_data()
        elif self.data_type == 'bgi':
            self.bin_size = 50
            self.get_bgi_data()
        elif self.data_type == 'STARmap':
            self.get_STARmap_data()
        elif self.data_type == "custom":
            self.get_custom_data()
        self.compute_nearest()
        self.compute_out_log_sigma()
        if self.rec_mask_neigh_threshold:
            self.get_rec_mask()

        gc.collect()

    def __getitem__(self, index: int):
        return

    def __len__(self):
        return len(self.adata)

    def compute_out_log_sigma(self):
        if self.out_type == "raw":
            self.out_log_sigma = np.log(self.adata.layers[self.count_key].std(0))
        elif self.out_type == "unscaled":
            self.out_log_sigma = np.log(self.adata.layers['unscaled'].std(0))
        elif self.out_type == "scaled":
            self.out_log_sigma = np.log(self.adata.layers['scaled'].std(0))
        elif self.out_type.startswith("pca"):
            self.out_log_sigma = np.log(self.adata.obsm['X_pca'].std(0))
        # assert np.isfinite(self.out_log_sigma).all()

    def normalized_gaussian_kernel(self, dist, axis=-1):
        if self.weighted_neigh:
            return scipy.special.softmax(np.exp(-np.square(dist) / (2 * np.square(self.weighted_neigh))), axis)
        else:
            return np.ones_like(dist)

    def compute_nearest(self):
        if 'X_pca' in self.adata.obsm.keys():
            assert isinstance(self.adata.obsm['X_pca'], np.ndarray)
        assert isinstance(self.adata.obsm['spatial'], np.ndarray)

    def get_gt_neighbors(self, expected_row_repeat_time):
        ks = np.array(self.hexagonal_num_list)
        for possible_neighbors in ks[ks > expected_row_repeat_time]:
            neigh = NearestNeighbors(n_neighbors=possible_neighbors, n_jobs=self.n_jobs)
            neigh.fit(self.adata.obsm['spatial'])
            neighbor_dist, neighbors = neigh.kneighbors(return_distance=True)
            # if all target spots have at least k neighbors with the same label
            if ((self.adata.obs["spatialLIBD"][neighbors.flatten()].to_numpy().reshape(neighbors.shape) == self.adata.obs["spatialLIBD"].to_numpy().reshape(-1, 1)).sum(1) < expected_row_repeat_time).sum() == 0:
                print(f"Search for {possible_neighbors} neighbors.")
                break
        row_idx = []
        col_idx = []
        row_repeat_time = 0
        last_row = 0
        # identify exactly k neighbors (because of some spots may have more than k neighbors with the same label)
        for r, c in zip(*(self.adata.obs["spatialLIBD"][neighbors.flatten()].to_numpy().reshape(neighbors.shape) == self.adata.obs["spatialLIBD"].to_numpy().reshape(-1, 1)).nonzero()):
            if r == last_row:
                if row_repeat_time < expected_row_repeat_time:
                    row_repeat_time += 1
                    row_idx.append(r)
                    col_idx.append(c)
            else:
                row_repeat_time = 1
                last_row = r
                row_idx.append(r)
                col_idx.append(c)
        row_idx = np.array(row_idx)
        col_idx = np.array(col_idx)
        gt_neighbor_dist = neighbor_dist[row_idx, col_idx].reshape(self.adata.X.shape[0], -1)
        gt_neighbors = neighbors[row_idx, col_idx].reshape(self.adata.X.shape[0], -1)
        gt_neighbor_graph = np.zeros((self.adata.X.shape[0], self.adata.X.shape[0]))
        gt_neighbor_graph[row_idx, gt_neighbors.flatten()] = 1.
        return gt_neighbor_dist, gt_neighbors, gt_neighbor_graph

    def get_rec_mask(self):
        print("rec_mask_neigh_threshold", self.rec_mask_neigh_threshold)
        if isinstance(self.adata.layers[self.count_key], np.ndarray):
            self.adata.layers["rec_mask"] = self.adata.layers[self.count_key] != 0
        else:
            self.adata.layers["rec_mask"] = self.adata.layers[self.count_key].A != 0
        if self.rec_neigh_num:
            if isinstance(self.adata.layers[self.count_key], np.ndarray):
                for i in range(len(self.adata)):
                    self.adata.layers["rec_mask"][i] *= ((self.adata.layers[self.count_key][self.rec_neigh_idx[i], :] > 0).mean(0) >= self.rec_mask_neigh_threshold)
            else:
                for i in range(len(self.adata)):
                    self.adata.layers["rec_mask"][i] *= ((self.adata.layers[self.count_key][self.rec_neigh_idx[i], :].A > 0).mean(0) >= self.rec_mask_neigh_threshold)

    def set_class_num(self):
        if self.annotation_key is not None:
            auto_classes = len(np.unique(self.adata.obs[f'{self.annotation_key}_int'][self.adata.obs['is_labelled']]))
        if self.num_classes == "auto" or self.num_classes == 0 or self.num_classes == "0":
            self.num_classes = auto_classes
        elif isinstance(self.num_classes, int):
            pass
        elif self.num_classes.startswith("plus_"):
            self.num_classes = auto_classes + int(self.num_classes[5:])
        elif self.num_classes.startswith("minus_"):
            self.num_classes = auto_classes - int(self.num_classes[6:])
        elif self.num_classes.startswith("multiply_"):
            self.num_classes = auto_classes * int(self.num_classes[9:])
        elif self.num_classes.startswith("divide_"):
            self.num_classes = auto_classes / int(self.num_classes[7:])

    def init_neighs_num(self):
        assert self.rec_neigh_num <= self.max_dynamic_neigh
        assert self.forward_neigh_num <= self.max_dynamic_neigh
        if self.dynamic_neigh_level == Dynamic_neigh_level.domain:
            self.dynamic_neigh_nums = [self.k] * self.num_classes
        elif self.dynamic_neigh_level.name.startswith("unit"):
            if self.dynamic_neigh_level.name.startswith("unit_fix_domain") or self.dynamic_neigh_level.name.startswith("unit_img"):
                self.dynamic_neigh_nums = [self.max_dynamic_neigh] * len(self.adata)
            else:
                self.dynamic_neigh_nums = [self.k] * len(self.adata)

    # @profile
    def preprocess_data(self, filter_by_counts=True, reuse_existing_pca=False, reuse_existing_knn=False, max_scale_value=None, not_labelled_value=None, skip_preprocess=False):
        if self.resample_to:
            resample_idx = self.rng.integers(len(self.adata), size=self.resample_to)
            self.adata = self.adata[resample_idx, :].copy()
            # self.adata = self.adata + np.abs(self.rng.normal(10, 10, self.adata.X.shape))
            self.adata.obs["resample_idx"] = resample_idx

        if self.annotation_key is not None:
            self.adata.obs[f"{self.annotation_key}"] = pd.Categorical(self.adata.obs[f"{self.annotation_key}"])
            sorted_cluster_idx = self.adata.obs[f"{self.annotation_key}"].value_counts().sort_index().index
            cluster_str2int_dict = dict(zip(sorted_cluster_idx, range(len(sorted_cluster_idx))))
            self.adata.obs[f"{self.annotation_key}_int"] = self.adata.obs[f"{self.annotation_key}"].map(cluster_str2int_dict)
            if np.sum(pd.isna(self.adata.obs[f"{self.annotation_key}_int"])) > 0:
                self.adata.obs[f"{self.annotation_key}_int"] = self.adata.obs[f"{self.annotation_key}_int"].cat.add_categories([-1])
                self.adata.obs.loc[pd.isna(self.adata.obs[f"{self.annotation_key}_int"]), f"{self.annotation_key}_int"] = -1
            if self.supervise_cat:
                self.adata = self.adata[self.adata.obs[f"{self.annotation_key}_int"] != -1]
                if self.supervise_cat == "random":
                    self.adata.obs["random_label"] = self.rng.integers(len(np.unique(self.adata.obs[f"{self.annotation_key}_int"].to_numpy())), size=self.adata.shape[0])
                    # from sklearn.metrics import adjusted_rand_score
                    # raise ValueError(f"Random label ARI: {adjusted_rand_score(self.adata.obs[f'{self.annotation_key}_int'], self.adata.obs['random_label'])}")
            self.adata.obs[f"{self.annotation_key}_int"] = self.adata.obs[f"{self.annotation_key}_int"].astype(np.int32)
            if not_labelled_value is not None:
                self.adata.obs['is_labelled'] = self.adata.obs[f"{self.annotation_key}"] != not_labelled_value
            else:
                self.adata.obs['is_labelled'] = ~self.adata.obs[f"{self.annotation_key}"].isna()
            if self.test_with_gt_sp:
                self.adata = self.adata[~self.adata.obs[f"{self.annotation_key}"].isna(), :].copy()
        assert self.adata.obsm['spatial'] is not None
        if reuse_existing_knn:
            assert self.adata.obsm['sp_k'] is not None
            raise
        self.adata.var_names_make_unique()

        if self.count_key is not None:
            whether_copy = self.adata.X != self.adata.layers[self.count_key]
            if (isinstance(whether_copy, np.ndarray) and (whether_copy).any()) or (scipy.sparse.issparse(whether_copy) and whether_copy.nnz > 0):
                self.adata.X = self.adata.layers[self.count_key].astype(np.float32).copy()
        if self.out_type == "raw":
            self.adata.layers[self.count_key] = self.adata.layers[self.count_key].toarray() if scipy.sparse.issparse(self.adata.layers[self.count_key]) else self.adata.layers[self.count_key]
            gene_mean = self.adata.layers[self.count_key].mean(axis=0)
            gene_var = self.adata.layers[self.count_key].var(axis=0)
            # good_gene_idx = gene_var < np.quantile(gene_var, 0.99)
            # popt, _ = curve_fit(nb_func, gene_mean[good_gene_idx], gene_var[good_gene_idx])
            print("gene_mean", gene_mean)
            print("gene_var", gene_var)
            popt, _ = curve_fit(nb_func, gene_mean, gene_var)
            self.adata.uns["r"] = popt[0]
        else:
            self.adata.uns["r"] = -1
        logging.debug("Filtering genes and cells.")
        tic = time.time()

        if (not skip_preprocess) and filter_by_counts:
            sc.pp.filter_genes(self.adata, min_counts=1)
            if self.dataset_dir == "CTL":
                sc.pp.filter_genes(self.adata, min_cells=100)
                sc.pp.filter_genes(self.adata, min_counts=500)
            sc.pp.filter_cells(self.adata, min_counts=1)
        toc = time.time()
        logging.debug(f"Filtering genes and cells takes {(toc - tic)/60:.2f} mins.")
        if self.device.type.startswith("cpu"):
            self.cpu_adata = self.adata.copy()
        if reuse_existing_pca:
            assert self.in_type == 'pca_scaled' or self.in_type == 'pca'
            self.adata.obsm["X_pca"] = self.adata.obsm[self.pca_key]
        else:
            if (not skip_preprocess) and (self.num_hvg is not None):
                if isinstance(self.num_hvg, float):
                    self.num_hvg = int(self.num_hvg * self.adata.shape[1])
                elif isinstance(self.num_hvg, int) and self.num_hvg < self.adata.X.shape[1]:
                    logging.debug("Selecting highly variable genes.")
                    tic = time.time()
                    # sc.pp.highly_variable_genes(self.adata, n_top_genes=self.num_hvg, flavor="cell_ranger", subset=True)
                    sc.pp.highly_variable_genes(self.adata, n_top_genes=self.num_hvg, flavor="seurat_v3", subset=True)
                    toc = time.time()
                    logging.debug(f"Selecting highly variable genes takes {(toc - tic)/60:.2f} mins.")

            if (not skip_preprocess) and filter_by_counts:
                sc.pp.filter_cells(self.adata, min_counts=1)
            try:
                self.adata.X = self.adata.X.astype(np.float32).toarray()
            except:
                pass
            if (not skip_preprocess) and self.lib_norm:
                logging.debug("Normalizing data to the median libaray size.")
                tic = time.time()
                if self.lib_norm is True:
                    self.adata.uns['target_library_size'] = np.median(self.adata.X.sum(1)).astype(np.int32)
                    sc.pp.normalize_total(self.adata)
                elif isinstance(self.lib_norm, int):
                    self.adata.uns['target_library_size'] = self.lib_norm
                    sc.pp.normalize_total(self.adata, target_sum=self.lib_norm)
                toc = time.time()
                logging.debug(f"Normalizing data to the median libaray size takes {(toc - tic)/60:.2f} mins.")
            if not skip_preprocess:
                logging.debug("Log transforming data.")
                tic = time.time()
                sc.pp.log1p(self.adata)
                toc = time.time()
                logging.debug(f"Log transforming data takes {(toc - tic)/60:.2f} mins.")
                self.adata.layers['unscaled'] = self.adata.X.copy()
                logging.debug("Scaling data.")
                tic = time.time()
                if max_scale_value is not None:
                    sc.pp.scale(self.adata)
                else:
                    sc.pp.scale(self.adata, max_value=max_scale_value)
                # sc.pp.scale(self.adata, zero_center=False, max_value=10)
                self.adata.var['mean'] = self.adata.var['mean'].astype(np.float32)
                self.adata.var['std'] = self.adata.var['std'].astype(np.float32)
                toc = time.time()
                logging.debug(f"Scaling data takes {(toc - tic)/60:.2f} mins.")
                self.adata.layers['scaled'] = self.adata.X.copy()
                logging.debug("Running PCA.")
                tic = time.time()
                if self.adata.shape[1] > self.n_pc:
                    pca = PCA(n_components=self.n_pc, random_state=self.seed)
                    if self.in_type == 'pca_unscaled':
                        self.adata.obsm["X_pca"] = pca.fit_transform(self.adata.layers['unscaled'])
                    elif self.in_type == 'pca_scaled' or self.in_type == 'pca' or self.in_type == "pca_harmony":
                        self.adata.obsm["X_pca"] = pca.fit_transform(self.adata.layers['scaled'])
                        if self.in_type == "pca_harmony":
                            import scanpy.external as sce
                            sce.pp.harmony_integrate(self.adata, "batch")
                    else:
                        self.adata.obsm["X_pca"] = pca.fit_transform(self.adata.X)
                    self.adata.uns["normalized_pca_explained_variance_ratio"] = pca.explained_variance_ratio_ / pca.explained_variance_ratio_.sum()
                    self.adata.uns['pca'] = pca
                else:
                    logging.warning(f"Number of genes is smaller than the number of PCs. Using scaled genes as PCs.")
                    self.adata.obsm["X_pca"] = self.adata.layers['scaled'].copy()
                toc = time.time()
                logging.debug(f"Running PCA takes {(toc - tic)/60:.2f} mins.")

        self.set_class_num()
        self.adata.obs["pred_labels"] = 0
        self.adata.obsm["prob_cat"] = np.zeros((len(self.adata), self.num_classes), dtype=np.float32)
        self.adata.obsm["prob_cat"][:, 0] = 1.
        self.adata = self.adata.copy()
        self.init_neighs_num()
        logging.debug("Computing spatial nearest neighbors.")
        tic = time.time()
        if not self.test_with_gt_sp:
            if self.unit_dynamic_candidate_num is None:
                neigh = NearestNeighbors(n_neighbors=self.max_dynamic_neigh, n_jobs=self.n_jobs)
                exp_neigh = NearestNeighbors(n_neighbors=self.max_dynamic_neigh, n_jobs=self.n_jobs)
            else:
                assert self.unit_dynamic_candidate_num >= self.max_dynamic_neigh
                neigh = NearestNeighbors(n_neighbors=self.unit_dynamic_candidate_num, n_jobs=self.n_jobs)
                exp_neigh = NearestNeighbors(n_neighbors=self.unit_dynamic_candidate_num, n_jobs=self.n_jobs)
            if (self.data_type == "10x" and len(self.sample_id.split('_')) > 1 and "batch" in self.adata.obs.keys()):
                neigh.fit(self.adata[self.adata.obs["batch"] == 0].obsm['spatial'])
                first_slide_sp_dist, _ = neigh.kneighbors(return_distance=True)
                min_unit_dist = first_slide_sp_dist.min()
                self.adata.obsm['spatial'] = np.hstack((self.adata.obsm['spatial'], (self.adata.obs["batch"] * min_unit_dist * self.z_scale).to_numpy().reshape(-1, 1)))
            elif self.multi_slides:
                neigh.fit(self.adata[self.adata.obs["batch"] == self.adata.obs["batch"].unique()[0]].obsm['spatial'])
                first_slide_sp_dist, _ = neigh.kneighbors(return_distance=True)
                min_unit_dist = first_slide_sp_dist.min()
                self.adata.obsm['spatial'] = np.hstack((self.adata.obsm['spatial'], (self.adata.obs["batch"].astype(np.float32) * min_unit_dist * self.z_scale).to_numpy().reshape(-1, 1)))
            neigh.fit(self.adata.obsm['spatial'])
            if "X_pca" not in self.adata.obsm.keys():
                if self.adata.X.shape[0] > 10000 and self.adata.X.shape[-1] > 1000:
                    print("Identifying exp_k may be slow on this large dataset.")
                exp_neigh.fit(self.adata.X)
            else:
                exp_neigh.fit(self.adata.obsm['X_pca'])
            if self.unit_dynamic_candidate_num is None:
                self.adata.obsm['sp_dist'], self.adata.obsm['sp_k'] = neigh.kneighbors(return_distance=True)
                self.adata.obsm['exp_dist'], self.adata.obsm['exp_k'] = exp_neigh.kneighbors(return_distance=True)
            else:
                self.adata.obsm['sp_dist_origin'], self.adata.obsm['sp_k_origin'] = neigh.kneighbors(return_distance=True)
                self.adata.obsm['exp_dist_origin'], self.adata.obsm['exp_k_origin'] = exp_neigh.kneighbors(return_distance=True)
                self.adata.obsm['sp_dist'] = self.adata.obsm['sp_dist_origin'][:, :self.max_dynamic_neigh]
                self.adata.obsm['sp_k'] = self.adata.obsm['sp_k_origin'][:, :self.max_dynamic_neigh]
                self.adata.obsm['exp_dist'] = self.adata.obsm['exp_dist_origin'][:, :self.max_dynamic_neigh]
                self.adata.obsm['exp_k'] = self.adata.obsm['exp_k_origin'][:, :self.max_dynamic_neigh]
                if self.annotation_key:
                    if self.dynamic_neigh_level == Dynamic_neigh_level.unit_img:
                        # add units relevant to fixed image neighbors
                        self.network_key = "img_k"
                        self.alt_network_key = None
                    elif self.dynamic_neigh_level == Dynamic_neigh_level.unit_img_sp:
                        # add units relevant to fixed spatial neighbors
                        self.network_key = "sp_k"
                        self.alt_network_key = "exp_k"
                    elif self.dynamic_neigh_level == Dynamic_neigh_level.unit_img_exp:
                        # add units relevant to fixed expression neighbors
                        self.network_key = "exp_k"
                        self.alt_network_key = "sp_k"
        else:
            if len(self.sample_id.split('_')) > 1:
                raise NotImplementedError
            gt_neighbor_dist, gt_neighbors, gt_neighbor_graph = self.get_gt_neighbors(expected_row_repeat_time=self.max_dynamic_neigh)
            self.adata.obsm['sp_k'] = gt_neighbors
            self.adata.obsm['sp_dist'] = gt_neighbor_dist
        self.adata.obsm['sp_dist'] = self.adata.obsm['sp_dist'].astype(np.float32)
        if self.supervise_cat or self.exchange_forward_neighbor_order:
            raise NotImplementedError
        if not self.test_with_gt_sp:
            self.adata.obsp['sp_adj'] = neigh.kneighbors_graph()
            self.adata.obsp['exp_adj'] = exp_neigh.kneighbors_graph()
        else:
            self.adata.obsp['sp_adj'] = gt_neighbor_graph
        self.adata.obsm["consist_adj"] = np.ones_like(self.adata.obsm["sp_k"]).astype(np.bool_)
        toc = time.time()
        logging.debug(f"Computing spatial nearest neighbors takes {(toc - tic)/60:.2f} mins.")
        gc.collect()

    def get_curr_tile(self, adata_img, row, col):
        if self.img_patch_origin_loc == "center":
            left_side_len = self.patch_size_factor * self.bin_size // 2
            right_side_len = left_side_len
            # ><>< inside
            if row - left_side_len >= 0 and row + right_side_len < adata_img.shape[0] and col - left_side_len >= 0 and col + right_side_len < adata_img.shape[1]:
                curr_tile = adata_img[(row - left_side_len):(row + right_side_len), (col - left_side_len):(col + right_side_len)]
            else:
                if len(adata_img.shape) == 3:
                    curr_tile = np.zeros((self.bin_size * self.patch_size_factor, self.bin_size * self.patch_size_factor, adata_img.shape[-1]), dtype=adata_img.dtype)
                else:
                    curr_tile = np.zeros((self.bin_size * self.patch_size_factor, self.bin_size * self.patch_size_factor), dtype=adata_img.dtype)
                # (0, 0) is the top left corner of the image, unlike the general scientific figures where (0, 0) is the bottom left corner.
                # <?<>< top
                if row - left_side_len < 0 and row + right_side_len > 0 and row + right_side_len < adata_img.shape[0] and col - left_side_len >= 0 and col + right_side_len < adata_img.shape[1]:
                    curr_tile[(left_side_len - row):, :] = adata_img[:(row + right_side_len), (col - left_side_len):(col + right_side_len)]
                # >?>>< bottom
                elif row - left_side_len >= 0 and row - left_side_len < adata_img.shape[0] and row + right_side_len >= adata_img.shape[0] and col - left_side_len >= 0 and col + right_side_len < adata_img.shape[1]:
                    curr_tile[:(adata_img.shape[0] - row + left_side_len), :] = adata_img[(row - left_side_len):, (col - left_side_len):(col + right_side_len)]
                # ><<?< left
                elif row - left_side_len >= 0 and row + right_side_len < adata_img.shape[0] and col - left_side_len < 0 and col + right_side_len > 0 and col + right_side_len < adata_img.shape[1]:
                    curr_tile[:, (left_side_len - col):] = adata_img[(row - left_side_len):(row + right_side_len), :(col + right_side_len)]
                # ><>?> right
                elif row - left_side_len >= 0 and row + right_side_len < adata_img.shape[0] and col - left_side_len >= 0 and col - left_side_len < adata_img.shape[1] and col + right_side_len >= adata_img.shape[1]:
                    curr_tile[:, :(adata_img.shape[1] - col + left_side_len)] = adata_img[(row - left_side_len):(row + right_side_len), (col - left_side_len):]
                # <?<<?< top left
                elif row - left_side_len < 0 and row + right_side_len > 0 and row + right_side_len < adata_img.shape[0] and col - left_side_len < 0 and col + right_side_len > 0 and col + right_side_len < adata_img.shape[1]:
                    curr_tile[(left_side_len - row):, (left_side_len - col):] = adata_img[:(row + right_side_len), :(col + right_side_len)]
                # >?>>?> bottom right
                elif row - left_side_len >= 0 and row - left_side_len < adata_img.shape[0] and row + right_side_len >= adata_img.shape[0] and col - left_side_len >= 0 and col - left_side_len < adata_img.shape[1] and col + right_side_len >= adata_img.shape[1]:
                    curr_tile[:(adata_img.shape[0] - row + right_side_len), :(adata_img.shape[1] - col + right_side_len)] = adata_img[(row - left_side_len):, (col - left_side_len):]
                # <?<>?> top right
                elif row - left_side_len < 0 and row + right_side_len > 0 and row + right_side_len < adata_img.shape[0] and col - left_side_len >= 0 and col - left_side_len < adata_img.shape[1] and col + right_side_len >= adata_img.shape[1]:
                    curr_tile[(left_side_len - row):, :(adata_img.shape[1] - col + right_side_len)] = adata_img[:(row + right_side_len), (col - left_side_len):]
                # >?><?< bottom left
                elif row - left_side_len >= 0 and row - left_side_len < adata_img.shape[0] and row + right_side_len >= adata_img.shape[0] and col - left_side_len < 0 and col + right_side_len > 0 and col + right_side_len < adata_img.shape[1]:
                    curr_tile[:(adata_img.shape[0] - row + right_side_len), (left_side_len - col):] = adata_img[(row - left_side_len):, :(col + right_side_len)]

        elif self.img_patch_origin_loc == "left_top":
            left_side_len = (self.patch_size_factor - 1) * self.bin_size // 2
            right_side_len = (self.patch_size_factor + 1) * self.bin_size // 2
            # ><>< inside
            if row - left_side_len >= 0 and row + right_side_len < adata_img.shape[0] and col - left_side_len >= 0 and col + right_side_len < adata_img.shape[1]:
                curr_tile = adata_img[(row - left_side_len):(row + right_side_len), (col - left_side_len):(col + right_side_len)]
            else:
                curr_tile = np.zeros((self.bin_size * self.patch_size_factor, self.bin_size * self.patch_size_factor), dtype=adata_img.dtype)
                # <<>< top
                if row - left_side_len < 0 and row + right_side_len < adata_img.shape[0] and col - left_side_len >= 0 and col + right_side_len < adata_img.shape[1]:
                    curr_tile[(left_side_len - row):, :] = adata_img[:(row + right_side_len), (col - left_side_len):(col + right_side_len)]
                # >>>< bottom
                elif row - left_side_len >= 0 and row + right_side_len >= adata_img.shape[0] and col - left_side_len >= 0 and col + right_side_len < adata_img.shape[1]:
                    curr_tile[:(adata_img.shape[0] - row + left_side_len), :] = adata_img[(row - left_side_len):, (col - left_side_len):(col + right_side_len)]
                # ><<< left
                elif row - left_side_len >= 0 and row + right_side_len < adata_img.shape[0] and col - left_side_len < 0 and col + right_side_len < adata_img.shape[1]:
                    curr_tile[:, (left_side_len - col):] = adata_img[(row - left_side_len):(row + right_side_len), :(col + right_side_len)]
                # ><>> right
                elif row - left_side_len >= 0 and row + right_side_len < adata_img.shape[0] and col - left_side_len >= 0 and col + right_side_len >= adata_img.shape[1]:
                    curr_tile[:, :(adata_img.shape[1] - col + left_side_len)] = adata_img[(row - left_side_len):(row + right_side_len), (col - left_side_len):]
                # <<<< top left
                elif row - left_side_len < 0 and row + right_side_len < adata_img.shape[0] and col - left_side_len < 0 and col + right_side_len < adata_img.shape[1]:
                    curr_tile[(left_side_len - row):, (left_side_len - col):] = adata_img[:(row + right_side_len), :(col + right_side_len)]
                # >>>> bottom right
                elif row - left_side_len >= 0 and row + right_side_len >= adata_img.shape[0] and col - left_side_len >= 0 and col + right_side_len >= adata_img.shape[1]:
                    curr_tile[:(adata_img.shape[0] - row + left_side_len), :(adata_img.shape[1] - col + left_side_len)] = adata_img[(row - left_side_len):, (col - left_side_len):]
                # <<>> top right
                elif row - left_side_len < 0 and row + right_side_len < adata_img.shape[0] and col - left_side_len >= 0 and col + right_side_len >= adata_img.shape[1]:
                    curr_tile[(left_side_len - row):, :(adata_img.shape[1] - col + left_side_len)] = adata_img[:(row + right_side_len), (col - left_side_len):]
                # >><< bottom left
                elif row - left_side_len >= 0 and row + right_side_len >= adata_img.shape[0] and col - left_side_len < 0 and col + right_side_len < adata_img.shape[1]:
                    curr_tile[:(adata_img.shape[0] - row + left_side_len), (left_side_len - col):] = adata_img[(row - left_side_len):, :(col + right_side_len)]

        return curr_tile

    def get_STARmap_data(self):
        self.count_key = "counts"
        self.annotation_key = "label"
        self.adata = sc.read_h5ad(osp.join(self.data_dir, self.dataset_dir, self.sample_id + ".h5ad"))
        self.adata.layers[self.count_key] = self.adata.X.copy()
        self.preprocess_data(max_scale_value=10.)

    def get_custom_data(self):
        self.adata = sc.read_h5ad(osp.join(self.data_dir, self.dataset_dir, self.data_file_name))
        if "counts" in self.adata.layers.keys():
            self.count_key = "counts"
        elif "count" in self.adata.layers.keys():
            self.count_key = "count"
        else:
            self.count_key = "counts"
            print("No count key found in layers, using default adata.X as count data.")
            self.adata.layers[self.count_key] = self.adata.X.copy()
        self.adata.obsm["spatial"] = self.adata.obsm["spatial"].astype(np.float32)
        self.adata = self.adata[self.adata.obs[self.annotation_key] != "nan"]
        self.preprocess_data()
        if self.use_image:
            logging.debug(f"Loading image.")
            tic = time.time()
            if "similarity" in self.use_image:
                img_embed_file_path = osp.join(self.data_dir, self.dataset_dir, f"{self.sample_id}_img_{self.patch_size_factor * self.bin_size}_embed.npz")
                if not osp.exists(img_embed_file_path):
                    img_file_path = osp.join(self.data_dir, self.dataset_dir, self.img_file_name)
                    if img_file_path.endswith(".h5ad"):
                        img_adata = sc.read_h5ad(img_file_path)
                        img_adata.X = img_adata.X.toarray() if scipy.sparse.issparse(img_adata.X) else img_adata.X
                        if self.img_coord_key not in self.adata.obsm.keys():
                            adata_anno_sp_split = self.adata.obs.index.str.split("_").to_list()
                            axis0s = [s[0] for s in adata_anno_sp_split]
                            axis1s = [s[1] for s in adata_anno_sp_split]

                            axis0s = np.array(axis0s, dtype=np.int32)
                            axis1s = np.array(axis1s, dtype=np.int32)
                            self.adata.obs["axis0"] = axis0s
                            self.adata.obs["axis1"] = axis1s
                            # self.adata.obsm[self.img_coord_key] = np.hstack((self.adata.obs["axis0"].values[:, None], self.adata.obs["axis1"].values[:, None])) * self.bin_size

                            # origin = np.hstack((self.adata.obsm[self.img_coord_key][:, [0]], self.adata.obsm[self.img_coord_key][:, [1]])).astype(np.int32)
                        if "stain" in img_adata.layers.keys() and "cell_labels" in img_adata.layers.keys():
                            adata_img = img_adata.layers["stain"] * (img_adata.layers["cell_labels"] > 0)
                        elif "stain" in img_adata.layers.keys():
                            adata_img = img_adata.layers["stain"]
                        else:
                            adata_img = img_adata.X.toarray()
                    else:
                        img = imread(img_file_path)
                        is_img_normalized = False
                        if img.dtype == np.uint16:
                            img = (img / (2**16 - 1) * 255).astype(np.uint8)
                        elif isinstance(img, np.floating):
                            img_max = img.max()
                            img_min = img.min()
                            if img_min < 0:
                                is_img_normalized = True
                            elif img_min >= 0 and img_max <= 1:
                                img = (img * 255).astype(np.uint8)
                            elif img_max > 255 and img_max <= 65535:
                                img = (img / 65535 * 255).astype(np.uint8)
                            elif img_max <= 255:
                                img = img.astype(np.uint8)
                    origin = np.hstack((self.adata.obsm[self.img_coord_key][:, [1]], self.adata.obsm[self.img_coord_key][:, [0]])).astype(np.int32)
                    adata_img = img

                    if not self.use_ddp or dist.get_rank() == 0:
                        # tile_path = osp.join(self.data_dir, self.dataset_dir, f"{self.sample_id}_img_{self.tiles.shape[-1]}.npz")
                        # if not osp.exists(tile_path):
                        self.tiles = []
                        assert adata_img.max() <= 255 and adata_img.min() >= 0
                        for row, col in tqdm(origin, "image cropping"):
                            curr_tile = self.get_curr_tile(adata_img, row, col)
                            self.tiles.append(curr_tile)

                        #     self.adata.obs.to_csv(osp.join(self.data_dir, self.dataset_dir, f"{self.sample_id}_img_obs.csv.gz"))
                        #     np.savez_compressed(osp.join(self.data_dir, self.dataset_dir, f"{self.sample_id}_img_{self.tiles.shape[-1]}.npz"), tiles=self.tiles.astype(np.uint8))
                        # else:
                        #     self.tiles = np.load(tile_path)["tiles"]

                        # https://github.com/ozanciga/self-supervised-histopathology
                        MODEL_PATH = 'tenpercent_resnet18.ckpt'

                        def load_model_weights(model, weights):

                            model_dict = model.state_dict()
                            weights = {k: v for k, v in weights.items() if k in model_dict}
                            if weights == {}:
                                print('No weight could be loaded..')
                            model_dict.update(weights)
                            model.load_state_dict(model_dict)

                            return model


                        # model = torchvision.models.__dict__['resnet18'](weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
                        model = torchvision.models.__dict__['resnet18'](weights=None)

                        state = torch.load(MODEL_PATH, map_location=self.device)
                        if 'state_dict' in state:
                            state_dict = state['state_dict']
                        else:
                            state_dict = state

                        if MODEL_PATH.endswith(".ckpt"):
                            for key in list(state_dict.keys()):
                                state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)
                        # elif MODEL_PATH.endswith(".tar"):
                            for key in list(state_dict.keys()):
                                state_dict[key.replace('encoder.', '').replace('projector.0.', 'fc.1.').replace('projector.2.', 'fc.3.')] = state_dict.pop(key)
                        model = load_model_weights(model, state_dict)

                        model.fc = torch.nn.Sequential()

                        model = model.to(self.device)
                        model.eval()
                        # print(self.device)
                        # raise
                        self.tiles = np.array(self.tiles, dtype=adata_img.dtype)
                        assert len(self.tiles) == len(self.adata)

                        cnn_features = []
                        for i in trange(0, len(self.tiles), self.img_batch_size):
                            if not is_img_normalized:
                                curr_tiles = ((self.tiles[i:i+self.img_batch_size] / 255.) - 0.5) / 0.5
                            else:
                                curr_tiles = self.tiles[i:i+self.img_batch_size]
                            if len(self.tiles.shape) == 3:
                                images = torch.from_numpy(curr_tiles).unsqueeze(1).expand(-1, 3, self.tiles.shape[-2], self.tiles.shape[-1]).float().to(self.device)
                            elif len(self.tiles.shape) == 4:
                                images = torch.from_numpy(curr_tiles).expand(-1, 3, self.tiles.shape[-2], self.tiles.shape[-1]).float().to(self.device)
                            out = model(images).detach().cpu().numpy()
                            cnn_features.append(out)
                        del model, self.tiles
                        torch.cuda.empty_cache()
                        self.adata.obsm['img_embed'] = np.concatenate(cnn_features, axis=0)
                        del cnn_features
                        np.savez_compressed(img_embed_file_path, img_embed=self.adata.obsm['img_embed'])
                else:
                    logging.debug(f"Loading image embedding.")
                    self.adata.obsm['img_embed'] = np.load(img_embed_file_path)["img_embed"]
                self.img_embed_metric = "cosine"
                logging.debug(f"Building image kNN.")
                if self.device.type.startswith("cpu"):
                    img_neigh = NearestNeighbors(n_neighbors=self.max_dynamic_neigh, metric=self.img_embed_metric, n_jobs=self.n_jobs)
                    img_neigh.fit(self.adata.obsm['img_embed'])
                else:
                    img_neigh = cuml.neighbors.NearestNeighbors(n_neighbors=self.max_dynamic_neigh, metric=self.img_embed_metric, output_type="numpy")
                    img_neigh.fit(cupy.asarray(self.adata.obsm['img_embed']))
            else:
                self.img_embed_metric = "minkowski"
                img_neigh = NearestNeighbors(n_neighbors=self.max_dynamic_neigh, n_jobs=self.n_jobs)
                self.adata.obsm['img_embed'] = self.adata.obs.loc[:, self.use_image].to_numpy()
                self.adata.obsm['img_embed'] = scipy.stats.zscore(self.adata.obsm['img_embed'], axis=0)
                img_neigh.fit(self.adata.obsm['img_embed'])
            self.adata.obsm['img_dist'], self.adata.obsm['img_k'] = img_neigh.kneighbors(return_distance=True)
            toc = time.time()
            logging.debug(f"Loading image takes {(toc - tic)/60:.2f} mins.")

    def get_10x_data(self):
        self.count_key = "counts"
        self.annotation_key = "spatialLIBD"
        self.adata = load_10x_with_meta(self.data_dir, self.dataset_dir, self.sample_id, self.count_key, self.annotation_key, filter_genes=1, filter_cells=1, filter_unlabelled=self.test_with_gt_sp)
        self.preprocess_data()
