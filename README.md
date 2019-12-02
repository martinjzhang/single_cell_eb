# sceb: 'single-cell empirical Bayes'
- Empirical Bayes estimators for single-cell RNA-Seq analysis, accompanying the paper "[Determining sequencing depth in a single-cell RNA-seq experiment](https://doi.org/10.1101/389296)".
- Installation: pip install sceb
- See ./examples/example_pbmc_4k.ipynb for an example for estimating the Pearson correlation.

# ./experiments: the code that reproduces all figures in the paper.

- Fig. 1b-c, Fig. 2b top, Supp. Figs. 1-3: `figure_gamma_schema.ipynb`

- Fig. 2a: The simulation is done using `compute_figure_tradeoff_curve.ipynb` and the figures are generated using `figure_tradeoff_curve.ipynb`

- Supp. Fig. 4: The simulation is done using `tradeoff_simu.py` called by `call_tradeoff_simu.sh`. The figures are generated using `figure_tradeoff_simu.ipynb`

- Fig. 2b: Supp. Figs. 5-6: The simulations are done using `tradeoff_posthoc_guide_pbmc.py` (Supp. Fig. 5) and `tradeoff_posthoc_guide_brain.py` (Supp. Fig. 6). The figures are generated using `figure_tradeoff_posthoc.ipynb`

- Fig. 3a top, Supp. Fig. 7: `figure_subsample_specturm.ipynb`

- Fig. 3a middle, Supp. Figs. 8-9: `figure_consistency.ipynb`

- Fig. 3a bottom, Supp. Fig. 10: `figure_dist_reconstruction.ipynb`

- Fig. 3b: `figure_feature_selection.ipynb`

- Fig. 4a: `figure_gene_module.ipynb`

- Fig. 4b-c: The network data is generated using `figure_gene_network.ipynb` and analyzed using Gephi (an external software). The examples are generated using `figure_network_example.ipynb`

- Fig. 5a-b, Supp. Fig. 13, 14: Comparison between Dropseq data and smFISH data (Figs. 5a-b, Supp. Fig. 14) was done in `figure_Dropseq_smFISH.ipynb`. Comparison between CEL-seq data and smFISH data (Supp. Fig. 13) was done in `figure_CELseq_smfish.ipynb`. 

- Supp. Figs. 15-17: `figure_ercc.ipynb`

- Supp. Fig. 18: `clean_PBMC_Ding_Levin_2019.ipynb` for curating the data from "[Ding et al. 2019](https://www.biorxiv.org/content/10.1101/632216v2)" and `figure_sensitivity_analysis.ipynb` for the analysis

# ./figures: The figures appeared in the paper as well as the simulated data to generate them.


# data
The data are downloaded locally with path specified inside ./sceb/data_loader. See ./examples/PC_estimation_pbmc_4k.ipynb for an example of defining a data loader function. 

The datasets that we use are from 10x genomics v2 chemistry "[Zheng et al. 2017](nature.com/articles/ncomms14049)". pbmc_4k, pbmc_8k contain peripheral blood mononuclear cells (PBMCs) from a healthy donor (the same donor). brain_1k, brain_2k, brain_9k, brain_1.3m contain cells from a combined cortex, hippocampus and sub ventricular zone of an E18 mouse. The pair 293T_1k, 3T3_1k contain 1:1 mixture of fresh frozen human (HEK293T) and mouse (NIH3T3) cells. So are the pair 293T_6k, 3T3_6k and the pair 293T_12k, 3T3_12k. The links of the data links: 

- pbmc_4k: https://support.10xgenomics.com/single-cell-gene-expression/datasets/2.1.0/pbmc4k
- pbmc_8k: https://support.10xgenomics.com/single-cell-gene-expression/datasets/2.1.0/pbmc8k
- brain_1k: https://support.10xgenomics.com/single-cell-gene-expression/datasets/2.1.0/neurons_900
- brain_2k: https://support.10xgenomics.com/single-cell-gene-expression/datasets/2.1.0/neurons_2000
- brain_9k: https://support.10xgenomics.com/single-cell-gene-expression/datasets/2.1.0/neuron_9k
- brain_1.3m: https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.3.0/1M_neurons
- 293T_1k, 3T3_1k: https://support.10xgenomics.com/single-cell-gene-expression/datasets/2.1.0/hgmm_1k
- 293T_6k, 3T3_6k: https://support.10xgenomics.com/single-cell-gene-expression/datasets/2.1.0/hgmm_6k
- 293T_12k, 3T3_12k: https://support.10xgenomics.com/single-cell-gene-expression/datasets/2.1.0/hgmm_12k
- Dropseq data and the corresponding smFISH data: from "[Wang et al. 2018](https://www-pnas-org.ezp-prod1.hul.harvard.edu/content/115/28/E6437.long)" 
- CEL-seq data and the corresonding smFISH data: the CEL-seq data can be found from "[Grün et al. 2014](nature.com/articles/nmeth.2930)". The smFISH can be found by contacting the author of the paper (e.g., Dr. Grün).
- The three ERCC datasets (Zheng, Klein, Svensson): from "[Wang et al. 2018](https://www-pnas-org.ezp-prod1.hul.harvard.edu/content/115/28/E6437.long)" 
- The Klein dataset with the pure RNA controls: from "[Svensson et al. 2017](nature.com/articles/nmeth.4220)"
- The data for sensitivity analysis: from "[Ding et al. 2019](https://www.biorxiv.org/content/10.1101/632216v2)"


# ./test: unit tests
