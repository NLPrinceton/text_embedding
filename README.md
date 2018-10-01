# text_embedding

This repository contains a fast, scalable, highly-parallel Python implementation of the GloVe [1] algorithm for word embeddings (found in <tt>solvers.py</tt>) as well as code and scripts to recreate downstream-task results for [unsupervised DisC embeddings](https://openreview.net/forum?id=B1e5ef-C-&noteId=B1e5ef-C-).

If you find this code useful please cite the following:
  
    @inproceedings{arora2018sensing,
      title={A Compressed Sensing View of Unsupervised Text Embeddings, Bag-of-n-Grams, and LSTMs},
      author={Arora, Sanjeev and Khodak, Mikhail and Saunshi, Nikunj and Vodrahalli, Kiran},
      booktitle={Proceedings of the 6th International Conference on Learning Representations (ICLR)},
      year={2018}
    }
    
# GloVe implementation

An implementation of the GloVe optimization algorithm (as well as code to build the vocab and cooccurrence files, optimize the related SN objective [2], and optimize a source-regularized objective for domain adaptation) can be found in <tt>solvers.py</tt>. The code scales to an arbitrary number of processors with virtually no memory/communication overhead. In terms of problem size the code scales linearly in time and memory complexity with the number of nonzero entries in the (sparse) cooccurrence matrix. 

On a 32-core computer, 25 epochs of AdaGrad run in 3.8 hours on Wikipedia cooccurrences with vocab size ~80K. The original C implementation runs in 2.8 hours on 32 cores. We also implement the option to use regular SGD, which requires about twice as many iterations to reach the same loss; however, the per-iteration complexity is much lower, and on the same 32-core computer 50 epochs finish in 2.0 hours.

Note that our code takes as input an upper-triangular, zero-indexed cooccurrence matrix rather than the full, one-indexed cooccurrence matrix used by the original GloVe code. To convert to our (more disk-memory efficient) version you can use the method <tt>reformat_coocfile</tt> in <tt>solvers.py</tt>. We also allow direct, parallel computation of the vocab and cooccurrence files.

Dependencies: numpy, numba, [SharedArray](https://pypi.org/project/SharedArray/)

Optional: h5py, mpi4py*, scipy, scikit-learn

\* required for parallelism; [MPI](http://www.mpich.org/downloads/) can be easily installed on Linux, Mac, and Windows Subsystem for Linux

# DisC embeddings

Scripts to recreate the results in the paper are provided in the directory <tt>scripts-AKSV2018</tt>. 1600-dimensional GloVe embeddings trained on the Amazon Product Corpus [3] are provided [here](http://nlp.cs.princeton.edu/DisC/amazon_glove1600.txt.bz2).

Dependencies: nltk, numpy, scipy, scikit-learn

Optional: tensorflow    

# References:

[1] Pennington et al., "GloVe: Global Vectors for Word Representation," *EMNLP*, 2014.

[2] Arora et al., "A Latent Variable Model Approach to PMI-based Word Embeddings," *TACL*, 2016.

[3] McAuley et al., "Inferring networks of substitutable and complementary products," *KDD*, 2015.
