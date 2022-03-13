# Final experiments

# Baseline
python train.py --do-train --run-name baseline --num-epochs=3 --lr=3e-5 --recompute-features --train-datasets=indomain_train/squad,indomain_train/nat_questions,indomain_train/newsqa --eval-datasets=indomain_val/squad,indomain_val/nat_questions,indomain_val/newsqa --visualize-predictions --eval-after-epoch


Command:
python train.py --do-train --eval-after-epoch --run-name=oo-SynonymAug-3-WordEmbsAug-3 --visualize-predictions --checkpoint-path=save/baseline-01/checkpoint --train-dir=datasets/oodomain_train --val-dir=datasets/oodomain_val --train-datasets=race,relation_extraction,duorc --num-epochs=10 --lr=3e-5 --recompute-features

Augmentation:
(augmenter.SynonymAug(aug_src='wordnet', tokenizer=Tokenizer.tokenizer, reverse_tokenizer=Tokenizer.reverse_tokenizer, include_detail=True), 3),
(augmenter.WordEmbsAug(model_type='word2vec', model_path='./GoogleNews-vectors-negative300.bin', tokenizer=Tokenizer.tokenizer, reverse_tokenizer=Tokenizer.reverse_tokenizer, include_detail=True, top_k=5), 3),

Result:
Eval F1: 51.30, EM: 36.13

# Baseline
```
Eval F1: 49.29, EM: 35.34
```

# RandomSwap, 3 candidates, LR=3e-6

```
python train.py --do-train --eval-after-epoch --run-name=oo-RandomSwap-3 --visualize-predictions --checkpoint-path=save/baseline-01/checkpoint --train-dir=datasets/oodomain_train --val-dir=datasets/oodomain_val --train-datasets=race,relation_extraction,duorc --num-epochs=15 --lr=3e-6 --recompute-features --augmentation-methods=random_swap
python train.py --do-eval --eval-dir datasets/oodomain_val --sub-file submssion.csv --save-dir save/oo-RandomSwap-3-04 --recompute-features

Eval F1: 49.45, EM: 35.60
```

# RandomSwap, 3 candidates, LR=3e-5

```
python train.py --do-train --eval-after-epoch --run-name=oo-RandomSwap-3 --visualize-predictions --checkpoint-path=save/baseline-01/checkpoint --train-dir=datasets/oodomain_train --val-dir=datasets/oodomain_val --train-datasets=race,relation_extraction,duorc --num-epochs=15 --lr=3e-5 --recompute-features --augmentation-methods=random_swap
python train.py --do-eval --eval-dir datasets/oodomain_val --sub-file submssion.csv --save-dir save/oo-RandomSwap-3-05 --recompute-features

Eval F1: 48.98, EM: 35.08
```

# Synonym Wordnet, 3 candidates, LR=3e-6
```
python train.py --do-train --eval-after-epoch --run-name=oo-Synonym-3 --visualize-predictions --checkpoint-path=save/baseline-01/checkpoint --train-dir=datasets/oodomain_train --val-dir=datasets/oodomain_val --train-datasets=race,relation_extraction,duorc --num-epochs=15 --lr=3e-6 --recompute-features --augmentation-methods=synonym_wordnet
python train.py --do-eval --eval-dir datasets/oodomain_val --sub-file submssion.csv --save-dir save/oo-Synonym-3-02 --recompute-features

Eval F1: 49.71, EM: 34.82
```

# Synonym Wordnet, 3 candidates, LR=3e-5
```
python train.py --do-train --eval-after-epoch --run-name=oo-Synonym-3 --visualize-predictions --checkpoint-path=save/baseline-01/checkpoint --train-dir=datasets/oodomain_train --val-dir=datasets/oodomain_val --train-datasets=race,relation_extraction,duorc --num-epochs=15 --lr=3e-5 --recompute-features --augmentation-methods=synonym_wordnet
python train.py --do-eval --eval-dir datasets/oodomain_val --sub-file submssion.csv --save-dir save/oo-Synonym-3-03 --recompute-features

Eval F1: 49.50, EM: 35.34
```

# WordEmbs Word2Vec, 3 candidates, LR=3e-6
```
python train.py --do-train --eval-after-epoch --run-name=oo-WordEmbs-3 --visualize-predictions --checkpoint-path=save/baseline-01/checkpoint --train-dir=datasets/oodomain_train --val-dir=datasets/oodomain_val --train-datasets=race,relation_extraction,duorc --num-epochs=15 --lr=3e-6 --recompute-features --augmentation-methods=wordembs_word2vec
python train.py --do-eval --eval-dir datasets/oodomain_val --sub-file submssion.csv --save-dir save/oo-WordEmbs-3-02 --recompute-features

Eval F1: 49.31, EM: 34.82
```

# ContextEmbs DistilBERT, 3 candidates, LR=3e-6

TODO

# Synonym Wordnet + WordEmbs Word2Vec, LR=3e-5
```
python train.py --do-train --eval-after-epoch --run-name=oo-SynonymAug-3-WordEmbs-3 --visualize-predictions --checkpoint-path=save/baseline-01/checkpoint --train-dir=datasets/oodomain_train --val-dir=datasets/oodomain_val --train-datasets=race,relation_extraction,duorc --num-epochs=10 --lr=3e-5 --recompute-features --augmentation-methods=synonym_wordnet,wordembs_word2vec
python train.py --do-eval --eval-dir datasets/oodomain_val --sub-file submssion.csv --save-dir save/oo-SynonymAug-3-WordEmbs-3-07 --recompute-features
```

```
python train.py --do-train --eval-after-epoch --run-name=mix-all --visualize-predictions --train-datasets=indomain_train/squad,indomain_train/nat_questions,indomain_train/newsqa,oodomain_train/race,oodomain_train/relation_extraction,oodomain_train/duorc --augment-datasets=oodomain_train/race,oodomain_train/relation_extraction,oodomain_train/duorc --eval-datasets=oodomain_val/race,oodomain_val/relation_extraction,oodomain_val/duorc --num-epochs=10 --lr=3e-5 --recompute-features --augment-methods=synonym_wordnet,random_swap
```
[03.09.22 00:28:11] Eval F1: 49.81, EM: 36.13 

```
'synonym_wordnet': (augmenter.SynonymAug(aug_src='wordnet', tokenizer=Tokenizer.tokenizer, reverse_tokenizer=Tokenizer.reverse_tokenizer, include_detail=True), 25),
'random_swap': (augmenter.RandomWordAug(action='swap', tokenizer=Tokenizer.tokenizer, reverse_tokenizer=Tokenizer.reverse_tokenizer, include_detail=True), 25),
'wordembs_word2vec': (augmenter.WordEmbsAug(model_type='word2vec', model_path='./GoogleNews-vectors-negative300.bin', device='cuda', tokenizer=Tokenizer.tokenizer, reverse_tokenizer=Tokenizer.reverse_tokenizer, include_detail=True, top_k=5), 25),
```
[03.09.22 12:19:09] Eval F1: 50.40, EM: 35.34

'synonym_wordnet': (augmenter.SynonymAug(aug_src='wordnet', tokenizer=Tokenizer.tokenizer, reverse_tokenizer=Tokenizer.reverse_tokenizer, include_detail=True), 35),
'random_swap': (augmenter.RandomWordAug(action='swap', tokenizer=Tokenizer.tokenizer, reverse_tokenizer=Tokenizer.reverse_tokenizer, include_detail=True), 35),
'wordembs_word2vec': (augmenter.WordEmbsAug(model_type='word2vec', model_path='./GoogleNews-vectors-negative300.bin', device='cpu', tokenizer=Tokenizer.tokenizer, reverse_tokenizer=Tokenizer.reverse_tokenizer, include_detail=True, top_k=5), 5),
[03.10.22 05:27:43] Eval F1: 49.90, EM: 35.86