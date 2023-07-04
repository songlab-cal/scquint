# scQuint (single-cell quantification of introns)

## Installation
```
pip install git+https://github.com/songlab-cal/scquint.git
```
If you need the VAE, do a full installation (with more complex dependencies):
```
pip install "scquint[vae] @ git+https://github.com/songlab-cal/scquint.git"
```

## Intron quantification
We recommend starting from the intron/junction count matrix obtained by running [STARsolo](https://github.com/alexdobin/STAR/blob/master/docs/STARsolo.md) with options `--soloFeatures Gene SJ`.

[How to run on Smart-seq data](https://github.com/alexdobin/STAR/blob/master/docs/STARsolo.md#plate-based-smart-seq-scrna-seq) 

[How to run on 10X Chromium data](https://github.com/alexdobin/STAR/blob/master/docs/STARsolo.md#running-starsolo-for-10x-chromium-scrna-seq-data) (We note that only a small proportion of alternative splicing events, those close to the 3' end of the gene, can be reliably detected in 10X data.)

Starting from the splice junction output directory of STARsolo, scQuint can prepare the data with a few steps:
```python
from scquint.data import load_adata_from_starsolo, add_gene_annotation, group_introns

adata = load_adata_from_starsolo("path/to/SJ_solo_outs")
adata = add_gene_annotation(adata, "path/to/gtf.gz")
adata = group_introns(adata, by="three_prime")
adata.write_h5ad("adata_spl.h5ad")
```

Precomputed AnnData objects are available at https://github.com/songlab-cal/scquint-analysis.

## Differential splicing
The basic commands would be:
```python
from scquint.differential_splicing import run_differential_splicing

diff_spl_intron_groups, diff_spl_introns = run_differential_splicing(adata, cell_idx_a, cell_idx_b)
```
See `differential_splicing_example.ipynb` for more details using Tabula Muris. <a target="_blank" href="https://colab.research.google.com/github/songlab-cal/scquint/blob/main/differential_splicing_example.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

See `differential_splicing_example_cortex.ipynb` for more details using BICCN mouse primary motor cortex. <a target="_blank" href="https://colab.research.google.com/github/songlab-cal/scquint/blob/main/differential_splicing_example_cortex.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Citation
Benegas, G., Fischer, J., Song., Y.S. Robust and annotation-free analysis of alternative splicing across diverse cell types in mice. eLife 2022;11:e73520  
DOI: [10.7554/eLife.73520](https://doi.org/10.7554/eLife.73520)
