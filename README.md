# text_embedding

Dependencies: NLTK, NumPy, SciPy, scikit-learn

Optional: tensorflow

If you find this code useful please cite the following:
  
    @inproceedings{arora2018sensing,
      title={A Compressed Sensing View of Unsupervised Text Embeddings, Bag-of-n-Grams, and LSTMs},
      author={Arora, Sanjeev and Khodak, Mikhail and Saunshi, Nikunj and Vodrahalli, Kiran},
      booktitle={Proceedings of the 6th International Conference on Learning Representations (ICLR)},
      year={2018}
    }
    
Scripts to recreate the results in the paper are provided in the directory <tt>scripts-AKSV2018</tt>. 1600-dimensional GloVe [1] embeddings trained on the Amazon Product Corpus [2] are provided [here](http://nlp.cs.princeton.edu/DisC/amazon_glove1600.txt.bz2).

# References:

[1] Pennington et al., "GloVe: Global Vectors for Word Representation," *EMNLP*, 2014.

[2] McAuley et al., "Inferring networks of substitutable and complementary products," *KDD*, 2015.
