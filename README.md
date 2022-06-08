# scQuint (single-cell quantification of introns)

## Installation
```
pip install git+https://github.com/songlab-cal/scquint.git
```
If you need the VAE, do a full installation (with more complex dependencies):
```
pip install "scquint[vae] @ git+https://github.com/songlab-cal/scquint.git"
```
Read mapping and intron quantification also require STAR, samtools and bedtools. An easy way to install is via conda:

```conda install -c bioconda star samtools bedtools```

## Read mapping
Our read mapping workflow, using [STAR](https://github.com/alexdobin/STAR), is designed for reliable detection of novel junctions. However, it is very computationally demanding. Users should consider going directly to intron quantification step if they already have produced bam alignment files. 

`python -m scquint.quantification.run read_mapping/Snakefile <snakemake_options> --config min_cells_per_intron=<min_cells_per_intron> fastq_paths=<fastq_paths> fasta_path=<fasta_path> gtf_path=<gtf_path> sjdb_overhang=<sjdb_overhang>`

`<snakemake_options>`: any options to pass to snakemake, e.g. `--cores 32 --directory output/mapping/`

`<sjdb_overhang>`: should be read_length-1. See STAR manual for details.

## Intron quantification
`python -m scquint.quantification.run introns/Snakefile <snakemake_options> --config min_cells_per_intron=<min_cells_per_intron> bam_paths=<bam_paths> chromosomes_path=<chromosomes_path> fasta_path=<fasta_path> gtf_path=<gtf_path> chrom_sizes_path=<chrom_sizes_path> sjdb_path=<sjdb_path>`

`sjdb_path`: annotated junctions in the format of `STAR_index/sjdbList.out.tab`.

Precomputed AnnData objects are available at https://github.com/songlab-cal/scquint-analysis.

## Differential splicing
See `differential_splicing_example.ipynb` <a target="_blank" href="https://colab.research.google.com/github/songlab-cal/scquint/blob/main/differential_splicing_example.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Citation
Benegas, G., Fischer, J., Song., Y.S. Robust and annotation-free analysis of alternative splicing across diverse cell types in mice. eLife 2022;11:e73520  
DOI: [10.7554/eLife.73520](https://doi.org/10.7554/eLife.73520)
