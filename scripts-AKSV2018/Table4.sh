# Script for Table 4 in AKSV, "A Compressed Sensing View of Unsupervised Text Embeddings," ICLR'18.
# sh text_embedding/scripts-AKSV2018/Table4.sh

tasks=mr,cr,subj,mpqa,trec,sst,sst_fine,imdb
for n in 2 3; do
  echo 'DisC, n='$n', elementwise multiplication'
  python text_embedding/cooc.py $tasks $n mult
  echo ''
  echo 'DisC, n='$n', circular convolution'
  python text_embedding/cooc.py $tasks $n conv
  echo ''
done
