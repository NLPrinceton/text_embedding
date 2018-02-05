# Script for Table 1 in AKSV, "A Compressed Sensing View of Unsupervised Text Embeddings," ICLR'18.
# sh text_embedding/scripts-AKSV2018/Table1.sh

tasks=mr,cr,subj,mpqa,trec,sst,sst_fine,imdb

# BonC
for n in 1 2 3; do
  echo 'BonC, n='$n
  python text_embedding/cooc.py $tasks $n
  echo ''
done

# DisC
for n in 1 2 3; do
  echo 'DisC, n='$n
  python text_embedding/cooc.py $tasks $n mult
  echo ''
done

#SIF
for a in 1E-5 1E-4 1E-3 1E-2 1E-1; do
  echo 'SIF, a='$a
  python text_embedding/baselines.py $tasks $a
  echo ''
done

# Sent2Vec
for order in uni bi; do
  echo 'Sent2Vec, order='$order
  python text_embedding/neural.py $tasks Sent2Vec $order
  echo ''
done

# skip-thoughts
echo 'skip-thoughts concat'
python2 text_embedding/neural.py mr,cr,subj,mpqa,trec,sst,sst_fine skipthoughts concat
echo ''

# byte mLSTM
echo 'byte mLSTM'
python text_embedding/neural.py $tasks byte_mLSTM
echo ''
