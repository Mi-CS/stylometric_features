# Datasets

This folder contains the subsample data for running the examples. 

## Gutenberg

The R package [gutenbergr](https://cran.r-project.org/web/packages/gutenbergr/vignettes/intro.html) was employed for selecting the subset and downloading the text files. It was first filtered by books written only in English, and then by the 50 authors that had more books available for download. 

Among these, for running the example, we finally choose Charles Dickens, Nathaniel Hawthorne, and George Alfred Henty. 

## Reuters 

We subsampled Benjamin Kang Lim, Samuel Perry, and William Kazer. The dataset is originally split 50% into train and test sets. We merged these, and sampled 80% of the articles for each author. 

## ArXiv cs.CL preprints

From the metadata file, we filtered Diptesh Kanojia, Hannes Westermann, and Bing Liu.