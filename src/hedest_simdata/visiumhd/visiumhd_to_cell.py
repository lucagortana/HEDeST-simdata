"""
Visium HD Processing Pipeline with bin2cell
============================================

A modular pipeline for processing Visium HD spatial transcriptomics data,
including H&E segmentation and bin-to-cell conversion.

Author: Custom pipeline based on bin2cell library
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import numpy as np
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
from PIL import Image
import tifffile as tf

# Ensure PIL can handle large images
Image.MAX_IMAGE_PIXELS = None
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2**40)

# Import bin2cell - assumes it's available in the environment
import bin2cell as b2c


class VisiumHD2Cell:
    """
    Main processor for Visium HD data with bin2cell integration.
    
    Handles the complete workflow from data loading through segmentation
    to cell-level transcriptomics extraction.
    """
    
    def __init__(
        self,
        binned_output_path: str,
        he_image_path: str,
        spaceranger_spatial_path: str,
        output_dir: str = "output",
        mpp: float = 0.5,
        bin_resolution: str = "002um",
        segmenter: str = "stardist",
        buffer: int = 150,
        pyr_tif_dir: Optional[str] = None,
        bridge=None,
        hovernet_gpu: str = "0",
        hovernet_batch_size: int = 16
    ):
        """
        Initialize the Visium HD processor.

        Parameters
        ----------
        binned_output_path : str
            Path to the binned outputs directory (e.g., 'binned_outputs/square_002um/')
        he_image_path : str
            Path to the high-resolution H&E image (.btf or .tif)
        spaceranger_spatial_path : str
            Path to the spaceranger spatial folder
        output_dir : str, default='output'
            Directory for output files
        mpp : float, default=0.5
            Microns per pixel of the rendered H&E. With ``segmenter='stardist'``
            this is the resolution nuclei are segmented at; with
            ``segmenter='hovernet'`` segmentation runs at the source resolution
            and this only controls the preview image used for QC figures.
        bin_resolution : str, default='002um'
            Bin resolution to use ('002um', '008um', or '016um')
        segmenter : str, default='stardist'
            Nuclear segmentation backend, 'stardist' or 'hovernet'.
        buffer : int, default=150
            Pixels of padding around the capture area when cropping the H&E.
        pyr_tif_dir : str, optional
            Only used with 'hovernet'. Where to write the cropped pyramidal
            TIFF. Defaults to a ``pyr_tif`` folder next to the source image.
        hovernet_gpu : str, default='0'
            GPU id(s) passed through to HoVerNet.
        hovernet_batch_size : int, default=16
            Per-GPU batch size for HoVerNet inference.
        """
        if segmenter not in ("stardist", "hovernet"):
            raise ValueError(
                f"segmenter must be 'stardist' or 'hovernet', got {segmenter!r}"
            )

        self.binned_output_path = Path(binned_output_path)
        self.he_image_path = Path(he_image_path)
        self.spaceranger_spatial_path = Path(spaceranger_spatial_path)
        self.output_dir = Path(output_dir)
        self.mpp = mpp
        self.bin_resolution = bin_resolution
        self.segmenter = segmenter
        self.buffer = buffer
        self.hovernet_gpu = hovernet_gpu
        self.hovernet_batch_size = hovernet_batch_size

        if pyr_tif_dir is None:
            pyr_tif_dir = self.he_image_path.parent / "pyr_tif"
        self.pyr_tif_dir = Path(pyr_tif_dir)
        self.bridge = bridge

        # Create output directories. Segmentation artefacts live in a folder
        # named after the backend, so the two are directly comparable.
        self.seg_dir = self.output_dir / segmenter
        self.stardist_dir = self.seg_dir  # backwards-compatible alias
        self.figures_dir = self.output_dir / "figures"
        self.adata_dir = self.output_dir / "bin2cell"
        self.seg_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.adata_dir.mkdir(parents=True, exist_ok=True)

        # Storage for processed data
        self.adata = None
        self.cdata = None
        # mpp the segmentation labels live at; set during segmentation
        self.seg_mpp = None
        self.spatial_key = f"spatial_cropped_{buffer}_buffer"

        print(f"Initialized VisiumHD2Cell")
        print(f"  Bin resolution: {bin_resolution}")
        print(f"  Segmenter: {segmenter}")
        print(f"  MPP: {mpp}")
        print(f"  Output directory: {output_dir}")
    
    def load_data(
        self,
        min_genes_per_cell: int = 3,
        min_cells_per_gene: int = 1
    ) -> ad.AnnData:
        """
        Load Visium HD data and perform initial filtering.
        
        Parameters
        ----------
        min_genes_per_cell : int, default=3
            Minimum number of genes per cell for filtering
        min_cells_per_gene : int, default=1
            Minimum number of cells per gene for filtering
            
        Returns
        -------
        adata : AnnData
            Loaded and filtered AnnData object
        """
        print("\n" + "="*60)
        print("STEP 1: Loading Visium HD Data")
        print("="*60)
        
        # Load the data
        print(f"Loading from: {self.binned_output_path}")
        self.adata = b2c.read_visium(
            str(self.binned_output_path),
            source_image_path=str(self.he_image_path),
            spaceranger_image_path=str(self.spaceranger_spatial_path)
        )
        
        # Make variable names unique
        self.adata.var_names_make_unique()
        
        print(f"Initial shape: {self.adata.shape[0]} bins × {self.adata.shape[1]} genes")
        
        # Filter genes and cells
        print(f"Filtering: min_cells={min_cells_per_gene}, min_counts={min_genes_per_cell}")
        sc.pp.filter_genes(self.adata, min_cells=min_cells_per_gene)
        sc.pp.filter_cells(self.adata, min_counts=min_genes_per_cell)
        
        print(f"After filtering: {self.adata.shape[0]} bins × {self.adata.shape[1]} genes")
        
        return self.adata
    
    def destripe_counts(
        self,
        quantile: float = 0.99,
        adjust_counts: bool = True
    ):
        """
        Correct for variable bin dimensions (destriping).
        
        Parameters
        ----------
        quantile : float, default=0.99
            Quantile to use for destriping normalization
        adjust_counts : bool, default=True
            Whether to adjust the count matrix
        """
        print("\n" + "="*60)
        print("STEP 2: Destriping Counts")
        print("="*60)
        
        print(f"Applying destriping with quantile={quantile}")
        b2c.destripe(
            self.adata,
            quantile=quantile,
            adjust_counts=adjust_counts
        )
        
        print("Destriping complete")
        print(f"  n_counts range: [{self.adata.obs['n_counts'].min():.1f}, {self.adata.obs['n_counts'].max():.1f}]")
        print(f"  n_counts_adjusted range: [{self.adata.obs['n_counts_adjusted'].min():.1f}, {self.adata.obs['n_counts_adjusted'].max():.1f}]")
    
    def prepare_he_image(
        self,
        crop: bool = True,
        buffer: int = 150
    ) -> str:
        """
        Prepare H&E image for segmentation.
        
        Parameters
        ----------
        crop : bool, default=True
            Whether to crop to spatial grid area
        buffer : int, default=150
            Buffer pixels around spatial grid (if cropping)
            
        Returns
        -------
        he_image_path : str
            Path to prepared H&E image
        """
        print("\n" + "="*60)
        print("STEP 3: Preparing H&E Image")
        print("="*60)
        
        he_output_path = self.stardist_dir / "he_prepared.tiff"
        
        print(f"Creating {self.mpp} mpp H&E image")
        print(f"  Crop: {crop}, Buffer: {buffer}")
        
        b2c.scaled_he_image(
            self.adata,
            mpp=self.mpp,
            crop=crop,
            buffer=buffer,
            save_path=str(he_output_path)
        )
        
        print(f"H&E image saved to: {he_output_path}")
        
        return str(he_output_path)
    
    def segment_he_nuclei(
        self,
        he_image_path: str,
        prob_thresh: float = 0.01,
        block_size: int = 4096,
        min_overlap: int = 128,
        context: int = 128
    ) -> str:
        """
        Segment nuclei from H&E image using StarDist.
        
        Parameters
        ----------
        he_image_path : str
            Path to prepared H&E image
        prob_thresh : float, default=0.01
            Probability threshold for StarDist (lower = more lenient)
        block_size : int, default=4096
            Size of tiles for processing
        min_overlap : int, default=128
            Minimum overlap between tiles
        context : int, default=128
            Context around each tile
            
        Returns
        -------
        labels_path : str
            Path to segmentation labels (.npz)
        """
        print("\n" + "="*60)
        print("STEP 4: Segmenting Nuclei (H&E)")
        print("="*60)
        
        labels_path = self.stardist_dir / "he_nuclei.npz"
        
        print(f"Running StarDist H&E segmentation")
        print(f"  prob_thresh: {prob_thresh}")
        print(f"  block_size: {block_size}")
        
        b2c.stardist(
            image_path=he_image_path,
            labels_npz_path=str(labels_path),
            stardist_model="2D_versatile_he",
            prob_thresh=prob_thresh,
            block_size=block_size,
            min_overlap=min_overlap,
            context=context
        )
        
        print(f"Segmentation complete. Labels saved to: {labels_path}")

        return str(labels_path)

    def segment_he_nuclei_hovernet(self) -> str:
        """
        Segment nuclei with HoVerNet at the source (full) resolution.

        Unlike the StarDist path, which segments a 0.5 mpp render, this crops
        the H&E to the capture area, runs HoVerNet on it at full resolution via
        the external ``hovernet`` environment, then rasterises the resulting
        polygons back into the sparse label array bin2cell expects.

        Returns
        -------
        labels_path : str
            Path to segmentation labels (.npz)
        """
        from .hovernet_seg import crop_capture_area, run_hovernet, rasterize_json

        print("\n" + "="*60)
        print("STEPS 3-4: Preparing H&E Image and Segmenting Nuclei (HoVerNet)")
        print("="*60)

        library = list(self.adata.uns['spatial'].keys())[0]
        # Labels are produced at the source resolution, so downstream
        # insert_labels must use this mpp (scale factor 1.0), not self.mpp.
        self.seg_mpp = self.adata.uns['spatial'][library]['scalefactors']['microns_per_pixel']
        print(f"Source resolution: {self.seg_mpp:.4f} mpp "
              f"(StarDist path used {self.mpp} mpp)")

        tif_path = self.pyr_tif_dir / f"{library}_capture.tif"
        print("\nCropping to capture area")
        _, crop_shape = crop_capture_area(
            self.adata,
            btf_path=str(self.he_image_path),
            out_tif=tif_path,
            buffer=self.buffer,
            preview_mpp=self.mpp,
        )

        json_path = run_hovernet(
            self.bridge,
            tif_path=tif_path,
            out_dir=self.seg_dir / "seg_json",
            mask_dir=self.seg_dir / "mask",
            # Per-sample, under this sample's own work dir. HoVerNet writes fixed-name
            # scratch (pred_map.npy, pred_inst.npy, cache_chunk.npy) here and clears the
            # directory first, so two samples sharing one cache mmap the same file and both
            # die with SIGBUS -- measured on node003 with P2 and P5 concurrent, 2026-11-08.
            # Upstream run_bin2cell.sh got this right by exporting
            # HOVERNET_CACHE=.../cache/hovernet_$SAMPLE per sample; the env var is gone here
            # because the path is now derived from the sample, and a default that is only
            # correct when a wrapper script happens to override it is a trap.
            cache_dir=self.seg_dir / "cache",
            gpu=self.hovernet_gpu,
            batch_size=self.hovernet_batch_size,
        )

        print("\nRasterising HoVerNet polygons into a label array")
        labels_path = self.seg_dir / "he_nuclei.npz"
        stats = rasterize_json(
            json_path=json_path,
            shape=crop_shape,
            out_npz=labels_path,
            types_csv=self.seg_dir / "nucleus_types.csv",
        )
        self.adata.uns.setdefault("bin2cell", {})["hovernet"] = {
            "json": str(json_path),
            "pyramidal_tif": str(tif_path),
            "seg_mpp": float(self.seg_mpp),
            **stats,
        }

        print(f"Segmentation complete. Labels saved to: {labels_path}")
        return str(labels_path)

    def segment_nuclei(self) -> str:
        """
        Run the configured nuclear segmentation backend.

        Returns
        -------
        labels_path : str
            Path to segmentation labels (.npz)
        """
        if self.segmenter == "hovernet":
            return self.segment_he_nuclei_hovernet()

        # StarDist: render the H&E at self.mpp, segment that render.
        self.seg_mpp = self.mpp
        he_image_path = self.prepare_he_image(buffer=self.buffer)
        return self.segment_he_nuclei(he_image_path)

    def insert_he_labels(
        self,
        labels_path: str,
        spatial_key: Optional[str] = None
    ):
        """
        Insert H&E segmentation labels into AnnData object.

        Parameters
        ----------
        labels_path : str
            Path to segmentation labels
        spatial_key : str, optional
            Spatial coordinates key to use. Defaults to the cropped key
            matching the configured buffer.
        """
        print("\n" + "="*60)
        print("STEP 5: Inserting H&E Labels")
        print("="*60)

        if spatial_key is None:
            spatial_key = self.spatial_key
        # The labels live at the resolution they were produced at: the render
        # mpp for StarDist, the source mpp for HoVerNet.
        seg_mpp = self.seg_mpp if self.seg_mpp is not None else self.mpp
        print(f"Mapping bins onto labels at {seg_mpp:.4f} mpp using '{spatial_key}'")

        b2c.insert_labels(
            self.adata,
            labels_npz_path=labels_path,
            basis="spatial",
            spatial_key=spatial_key,
            mpp=seg_mpp,
            labels_key="labels_he"
        )
        
        n_labeled = (self.adata.obs['labels_he'] > 0).sum()
        print(f"Labeled {n_labeled} bins with nuclear segmentation")
        print(f"  Coverage: {100 * n_labeled / self.adata.shape[0]:.2f}%")
    
    def expand_nuclear_labels(
        self,
        algorithm: str = "max_bin_distance",
        max_bin_distance: int = 2,
        volume_ratio: float = 4.0,
        k: int = 4
    ):
        """
        Expand nuclear labels to include cytoplasm.
        
        Parameters
        ----------
        algorithm : str, default='max_bin_distance'
            Expansion algorithm ('max_bin_distance' or 'volume_ratio')
        max_bin_distance : int, default=2
            Maximum bins to expand (for 'max_bin_distance' algorithm)
        volume_ratio : float, default=4.0
            Cell/nucleus volume ratio (for 'volume_ratio' algorithm)
        k : int, default=4
            Number of neighbors for tie-breaking
        """
        print("\n" + "="*60)
        print("STEP 6: Expanding Nuclear Labels")
        print("="*60)
        
        print(f"Expanding labels using '{algorithm}' algorithm")
        if algorithm == "max_bin_distance":
            print(f"  max_bin_distance: {max_bin_distance}")
        else:
            print(f"  volume_ratio: {volume_ratio}")
        
        b2c.expand_labels(
            self.adata,
            labels_key='labels_he',
            expanded_labels_key='labels_he_expanded',
            algorithm=algorithm,
            max_bin_distance=max_bin_distance if algorithm == "max_bin_distance" else None,
            volume_ratio=volume_ratio,
            k=k
        )
        
        n_expanded = (self.adata.obs['labels_he_expanded'] > 0).sum()
        n_original = (self.adata.obs['labels_he'] > 0).sum()
        print(f"Expanded labels cover {n_expanded} bins")
        print(f"  Increase: {n_expanded - n_original} bins ({100 * (n_expanded - n_original) / n_original:.1f}%)")
    
    def segment_gex_optional(
        self,
        sigma: float = 5.0,
        prob_thresh: float = 0.05,
        nms_thresh: float = 0.5
    ) -> Optional[str]:
        """
        Optional: Segment based on gene expression to fill gaps.
        
        Parameters
        ----------
        sigma : float, default=5.0
            Gaussian filter sigma for smoothing
        prob_thresh : float, default=0.05
            Probability threshold for StarDist
        nms_thresh : float, default=0.5
            Non-maximum suppression threshold
            
        Returns
        -------
        labels_path : str or None
            Path to GEX segmentation labels
        """
        print("\n" + "="*60)
        print("STEP 7 (Optional): GEX-based Segmentation")
        print("="*60)
        
        # Create GEX image
        gex_image_path = self.stardist_dir / "gex.tiff"
        print(f"Creating GEX image (sigma={sigma})")
        
        b2c.grid_image(
            self.adata,
            "n_counts_adjusted",
            mpp=self.mpp,
            sigma=sigma,
            save_path=str(gex_image_path)
        )
        
        # Segment
        labels_path = self.stardist_dir / "gex_cells.npz"
        print(f"Running StarDist fluorescence segmentation")
        
        b2c.stardist(
            image_path=str(gex_image_path),
            labels_npz_path=str(labels_path),
            stardist_model="2D_versatile_fluo",
            prob_thresh=prob_thresh,
            nms_thresh=nms_thresh
        )
        
        # Insert labels
        b2c.insert_labels(
            self.adata,
            labels_npz_path=str(labels_path),
            basis="array",
            mpp=self.mpp,
            labels_key="labels_gex"
        )
        
        print(f"GEX segmentation complete")
        
        return str(labels_path)
    
    def combine_labels(
        self,
        use_gex: bool = True
    ):
        """
        Combine H&E and GEX labels (if available).
        
        Parameters
        ----------
        use_gex : bool, default=True
            Whether to salvage secondary GEX labels
        """
        print("\n" + "="*60)
        print("STEP 8: Combining Label Sources")
        print("="*60)
        
        if use_gex and 'labels_gex' in self.adata.obs.columns:
            print("Salvaging secondary GEX labels")
            b2c.salvage_secondary_labels(
                self.adata,
                primary_label="labels_he_expanded",
                secondary_label="labels_gex",
                labels_key="labels_joint"
            )
            final_label = "labels_joint"
        else:
            print("Using only H&E labels")
            self.adata.obs['labels_joint'] = self.adata.obs['labels_he_expanded'].copy()
            final_label = "labels_joint"
        
        n_final = (self.adata.obs[final_label] > 0).sum()
        print(f"Final labeled bins: {n_final} ({100 * n_final / self.adata.shape[0]:.2f}%)")
        
        return final_label
    
    def create_cells(
        self,
        labels_key: str = "labels_joint",
        spatial_keys: Optional[List[str]] = None
    ) -> ad.AnnData:
        """
        Convert bins to cells.
        
        Parameters
        ----------
        labels_key : str, default='labels_joint'
            Label key to use for grouping bins
        spatial_keys : list of str, optional
            Spatial coordinate keys to preserve
            
        Returns
        -------
        cdata : AnnData
            Cell-level AnnData object
        """
        print("\n" + "="*60)
        print("STEP 9: Creating Cell-Level Object")
        print("="*60)
        
        if spatial_keys is None:
            spatial_keys = ["spatial", self.spatial_key]

        print(f"Converting bins to cells using '{labels_key}'")
        
        self.cdata = b2c.bin_to_cell(
            self.adata,
            labels_key=labels_key,
            spatial_keys=spatial_keys
        )
        
        print(f"Created {self.cdata.shape[0]} cells from {self.adata.shape[0]} bins")
        print(f"  Mean bins per cell: {self.cdata.obs['bin_count'].mean():.1f}")
        print(f"  Median bins per cell: {self.cdata.obs['bin_count'].median():.1f}")
        
        return self.cdata
    
    def save_results(
        self,
        save_bin_data: bool = True,
        save_cell_data: bool = True
    ):
        """
        Save processed data to disk.
        
        Parameters
        ----------
        save_bin_data : bool, default=True
            Save bin-level AnnData
        save_cell_data : bool, default=True
            Save cell-level AnnData
        """
        print("\n" + "="*60)
        print("STEP 10: Saving Results")
        print("="*60)
        
        if save_bin_data and self.adata is not None:
            bin_path = self.adata_dir / "bin_data.h5ad"
            print(f"Saving bin-level data to: {bin_path}")
            self.adata.write_h5ad(bin_path)
        
        if save_cell_data and self.cdata is not None:
            cell_path = self.adata_dir / "cell_data.h5ad"
            print(f"Saving cell-level data to: {cell_path}")
            self.cdata.write_h5ad(cell_path)
        
        print("Save complete!")
    
    def visualize_qc(
        self,
        sample_region: Optional[Tuple[int, int, int, int]] = None,
        spatial_key: Optional[str] = None,
        img_key: Optional[str] = None
    ):
        """
        Create QC visualizations.

        Parameters
        ----------
        sample_region : tuple of int, optional
            (row_min, row_max, col_min, col_max) for visualization
        spatial_key : str, optional
            Spatial coordinates key. Defaults to the configured cropped key.
        img_key : str, optional
            Image key for plotting. Defaults to the preview render at self.mpp.
        """
        print("\n" + "="*60)
        print("Creating QC Visualizations")
        print("="*60)

        if spatial_key is None:
            spatial_key = self.spatial_key
        if img_key is None:
            img_key = f"{self.mpp}_mpp_{self.buffer}_buffer"

        lib_id = list(self.adata.uns['spatial'].keys())[0]
        
        if sample_region is None:
            # Auto-select a central region
            row_min = int(self.adata.obs['array_row'].quantile(0.4))
            row_max = int(self.adata.obs['array_row'].quantile(0.6))
            col_min = int(self.adata.obs['array_col'].quantile(0.4))
            col_max = int(self.adata.obs['array_col'].quantile(0.6))
            sample_region = (row_min, row_max, col_min, col_max)
        
        # Create subset
        mask = (
            (self.adata.obs['array_row'] >= sample_region[0]) &
            (self.adata.obs['array_row'] <= sample_region[1]) &
            (self.adata.obs['array_col'] >= sample_region[2]) &
            (self.adata.obs['array_col'] <= sample_region[3])
        )
        subset = self.adata[mask].copy()
        
        # FIX 1: Destriping QC (Plotting two panels)
        # We don't pass 'ax' here; Scanpy will return an array of axes
        axs = sc.pl.spatial(
            subset,
            color=["n_counts", "n_counts_adjusted"],
            color_map="OrRd",
            img_key=img_key,
            library_id=lib_id,
            basis=spatial_key,
            show=False
        )
        plt.savefig(self.figures_dir / "destriping_qc.png", dpi=150)
        plt.close()
        
        # FIX 2: Label coverage (Plotting single panel)
        if 'labels_he' in self.adata.obs.columns:
            # Avoid creating thousands of categories in the legend
            subset.obs['has_label'] = (subset.obs['labels_he'] > 0).astype(str)
            
            sc.pl.spatial(
                subset,
                color="has_label",
                img_key=img_key,
                library_id=lib_id,
                basis=spatial_key,
                palette={'True': 'red', 'False': 'lightgrey'},
                title="Nuclear Segmentation Alignment",
                show=False
            )
            plt.savefig(self.figures_dir / "nuclear_segmentation.png", dpi=150)
            plt.close()

        # Upstream also spawned analysis/preprocessing/plot_hovernet_types.py here for a
        # per-type crop mosaic. That script was not ported and the mosaic is deliberately
        # not produced: it went to this class's own figures directory, which lives under
        # `_work` and is deleted when the family finishes, and it is in neither the manifest
        # nor any delivered folder -- the 20 figures that ship come from the `plots` stage.
        # Until 2026-11-08 the call was still here, so `visiumhd:bin2cell` ran the full
        # hour of StarDist and HoVerNet and then died with `can't open file ...
        # plot_hovernet_types.py` on its very last step. A port has to bring the callees of
        # what it copies, or drop the call with the copy.

        print(f"QC figures saved to: {self.figures_dir}")
    
    def run_complete_pipeline(
        self,
        use_gex_segmentation: bool = True,
        create_visualizations: bool = True,
        **kwargs
    ) -> Tuple[ad.AnnData, ad.AnnData]:
        """
        Run the complete processing pipeline.
        
        Parameters
        ----------
        use_gex_segmentation : bool, default=True
            Whether to perform GEX-based segmentation
        create_visualizations : bool, default=True
            Whether to create QC visualizations
        **kwargs
            Additional parameters for individual steps
            
        Returns
        -------
        adata : AnnData
            Bin-level data
        cdata : AnnData
            Cell-level data
        """
        print("\n" + "#"*60)
        print("# VISIUM HD PROCESSING PIPELINE")
        print("#"*60)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Destripe
        self.destripe_counts()
        
        # Steps 3-4: Prepare H&E image and segment nuclei
        he_labels_path = self.segment_nuclei()

        # Step 5: Insert labels
        self.insert_he_labels(he_labels_path)
        
        # Step 6: Expand labels
        self.expand_nuclear_labels()
        
        # Step 7: Optional GEX segmentation
        if use_gex_segmentation:
            self.segment_gex_optional()
        
        # Step 8: Combine labels
        final_label = self.combine_labels(use_gex=use_gex_segmentation)
        
        # Step 9: Create cells
        self.create_cells(labels_key=final_label)
        
        # Step 10: Save results
        self.save_results()
        
        # Visualizations
        if create_visualizations:
            self.visualize_qc()
        
        print("\n" + "#"*60)
        print("# PIPELINE COMPLETE!")
        print("#"*60)
        print(f"\nResults saved to: {self.output_dir}")
        print(f"  Bin-level data: {self.adata.shape[0]} bins × {self.adata.shape[1]} genes")
        print(f"  Cell-level data: {self.cdata.shape[0]} cells × {self.cdata.shape[1]} genes")
        
        return self.adata, self.cdata


# Convenience function for quick processing
def process_visiumhd(
    binned_output_path: str,
    he_image_path: str,
    spaceranger_spatial_path: str,
    output_dir: str = "output",
    mpp: float = 0.5,
    use_gex_segmentation: bool = True,
    segmenter: str = "stardist",
    buffer: int = 150,
    pyr_tif_dir: Optional[str] = None,
    bridge=None,
    hovernet_gpu: str = "0",
    hovernet_batch_size: int = 16
) -> Tuple[ad.AnnData, ad.AnnData]:
    """
    Quick processing function for Visium HD data.
    
    Parameters
    ----------
    binned_output_path : str
        Path to binned outputs (e.g., 'binned_outputs/square_002um/')
    he_image_path : str
        Path to H&E image (.btf or .tif)
    spaceranger_spatial_path : str
        Path to spaceranger spatial folder
    output_dir : str, default='output'
        Output directory
    mpp : float, default=0.5
        Microns per pixel
    use_gex_segmentation : bool, default=True
        Use GEX segmentation to fill gaps
        
    Returns
    -------
    adata : AnnData
        Bin-level data
    cdata : AnnData
        Cell-level data
    """
    processor = VisiumHD2Cell(
        binned_output_path=binned_output_path,
        he_image_path=he_image_path,
        spaceranger_spatial_path=spaceranger_spatial_path,
        output_dir=output_dir,
        mpp=mpp,
        segmenter=segmenter,
        buffer=buffer,
        pyr_tif_dir=pyr_tif_dir,
        bridge=bridge,
        hovernet_gpu=hovernet_gpu,
        hovernet_batch_size=hovernet_batch_size
    )

    return processor.run_complete_pipeline(
        use_gex_segmentation=use_gex_segmentation
    )
