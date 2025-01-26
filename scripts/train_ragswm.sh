CUDA_VISIBLE_DEVICES=0 python train_ragswm.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_dev.jsonl \
--corpus-data-path ../data/splitted_polymarket/polymarket_data_processed_train.jsonl \
--output-dir ../saves/saves_crypto_ragswm \
--cache-dir ../cache/cache_crypto_ragswm \
--train-batch-size 16 \
--eval-batch-size 16 \
--retriever-top-k 50 \
--epochs 10
--corpus-data-path ../data/splitted_polymarket/polymarket_data_processed_train.jsonl \
--output-dir ../saves/saves_crypto_ragswm \
--cache-dir ../cache/cache_crypto_ragswm \
--train-batch-size 16 \
--eval-batch-size 16 \
--retriever-top-k 50 \
--epochs 10

CUDA_VISIBLE_DEVICES=1 python train_ragswm.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Sports_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Sports_dev.jsonl \
--corpus-data-path ../data/splitted_polymarket/polymarket_data_processed_train.jsonl \
--output-dir ../saves/saves_sports_ragswm \
--cache-dir ../cache/cache_sports_ragswm \
--train-batch-size 16 \
--eval-batch-size 16 \
--retriever-top-k 50 \
--epochs 10
--corpus-data-path ../data/splitted_polymarket/polymarket_data_processed_train.jsonl \
--output-dir ../saves/saves_sports_ragswm \
--cache-dir ../cache/cache_sports_ragswm \
--train-batch-size 16 \
--eval-batch-size 16 \
--retriever-top-k 50 \
--epochs 10

CUDA_VISIBLE_DEVICES=7 python train_ragswm.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Other_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Other_dev.jsonl \
--corpus-data-path ../data/splitted_polymarket/polymarket_data_processed_train.jsonl \
--output-dir ../saves/saves_other_ragswm \
--cache-dir ../cache/cache_other_ragswm \
--train-batch-size 16 \
--eval-batch-size 16 \
--retriever-top-k 50 \
--epochs 10
--corpus-data-path ../data/splitted_polymarket/polymarket_data_processed_train.jsonl \
--output-dir ../saves/saves_other_ragswm \
--cache-dir ../cache/cache_other_ragswm \
--train-batch-size 16 \
--eval-batch-size 16 \
--retriever-top-k 50 \
--epochs 10

CUDA_VISIBLE_DEVICES=9 python train_ragswm.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Election_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Election_dev.jsonl \
--corpus-data-path ../data/splitted_polymarket/polymarket_data_processed_train.jsonl \
--output-dir ../saves/saves_election_ragswm \
--cache-dir ../cache/cache_election_ragswm \
--train-batch-size 16 \
--eval-batch-size 16 \
--retriever-top-k 50 \
--epochs 10
--corpus-data-path ../data/splitted_polymarket/polymarket_data_processed_train.jsonl \
--output-dir ../saves/saves_election_ragswm \
--cache-dir ../cache/cache_election_ragswm \
--train-batch-size 16 \
--eval-batch-size 16 \
--retriever-top-k 50 \
--epochs 10

CUDA_VISIBLE_DEVICES=5 python train_ragswm.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Politics_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Politics_dev.jsonl \
--corpus-data-path ../data/splitted_polymarket/polymarket_data_processed_train.jsonl \
--output-dir ../saves/saves_politics_ragswm \
--cache-dir ../cache/cache_politics_ragswm \
--train-batch-size 16 \
--eval-batch-size 16 \
--retriever-top-k 50 \
--epochs 10
--corpus-data-path ../data/splitted_polymarket/polymarket_data_processed_train.jsonl \
--output-dir ../saves/saves_politics_ragswm \
--cache-dir ../cache/cache_politics_ragswm \
--train-batch-size 16 \
--eval-batch-size 16 \
--retriever-top-k 50 \
--epochs 10

CUDA_VISIBLE_DEVICES=0 python train_ragswm.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_dev.jsonl \
--corpus-data-path ../data/splitted_polymarket/polymarket_data_processed_train.jsonl \
--output-dir ../saves/saves_all_ragswm \
--cache-dir ../cache/cache_all_ragswm \
--train-batch-size 16 \
--eval-batch-size 16 \
--retriever-top-k 50 \
--epochs 10


# for sanity check
CUDA_VISIBLE_DEVICES=2 python train_ragswm.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--corpus-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_train.jsonl \
--output-dir ../saves/saves_crypto_ragswm_sanitycheck \
--cache-dir ../cache/cache_crypto_ragswm_sanitycheck \
--retriever-top-k 50 \
--epochs 160 \
--save-steps 10 \
--eval-steps 10 \
--warmup-steps 0 \
--sanity-check
--corpus-data-path ../data/splitted_polymarket/polymarket_data_processed_train.jsonl \
--output-dir ../saves/saves_all_ragswm \
--cache-dir ../cache/cache_all_ragswm \
--train-batch-size 16 \
--eval-batch-size 16 \
--retriever-top-k 50 \
--epochs 10


# for sanity check
CUDA_VISIBLE_DEVICES=2 python train_ragswm.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--corpus-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_train.jsonl \
--output-dir ../saves/saves_crypto_ragswm_sanitycheck \
--cache-dir ../cache/cache_crypto_ragswm_sanitycheck \
--retriever-top-k 50 \
--epochs 160 \
--save-steps 10 \
--eval-steps 10 \
--warmup-steps 0 \
--sanity-check
