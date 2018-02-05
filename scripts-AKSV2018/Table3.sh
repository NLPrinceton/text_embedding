# Script for Table 3 in AKSV, "A Compressed Sensing View of Unsupervised Text Embeddings," ICLR'18.
# sh text_embedding/scripts-AKSV2018/Table3.sh

tasks=mr,cr,subj,mpqa,trec,sst,sst_fine,imdb
for n in 2 3; do
  echo 'BonG, n='$n
  python text_embedding/baselines.py $tasks $n
  echo ''
  echo 'BonC, n='$n
  python text_embedding/cooc.py $tasks $n
  echo ''
done
