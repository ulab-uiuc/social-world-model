CUDA_VISIBLE_DEVICES=9 python train_ragswm.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_dev.jsonl \
--corpus-data-path ../data/splitted_polymarket/polymarket_data_processed_train.jsonl \
--output-dir ../saves/saves_crypto \
--cache-dir ../cache/cache_crypto \
--predictions-path ../saves/predictions_crypto.csv \
--train-batch-size 16 \
--eval-batch-size 16 \
--top-k 50 \
--epochs 3

CUDA_VISIBLE_DEVICES=8 python train_ragswm.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Sports_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Sports_dev.jsonl \
--corpus-data-path ../data/splitted_polymarket/polymarket_data_processed_train.jsonl \
--output-dir ../saves/saves_sports \
--cache-dir ../cache/cache_sports \
--predictions-path ../saves/predictions_crypto.csv \
--train-batch-size 16 \
--eval-batch-size 16 \
--top-k 50 \
--epochs 3

CUDA_VISIBLE_DEVICES=7 python train_ragswm.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Other_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Other_dev.jsonl \
--corpus-data-path ../data/splitted_polymarket/polymarket_data_processed_train.jsonl \
--output-dir ../saves/saves_other \
--cache-dir ../cache/cache_other \
--predictions-path ../saves/predictions_other.csv \
--train-batch-size 16 \
--eval-batch-size 16 \
--top-k 50 \
--epochs 3

CUDA_VISIBLE_DEVICES=6 python train_ragswm.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Election_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Election_dev.jsonl \
--corpus-data-path ../data/splitted_polymarket/polymarket_data_processed_train.jsonl \
--output-dir ../saves/saves_election \
--cache-dir ../cache/cache_election \
--predictions-path ../saves/predictions_election.csv \
--train-batch-size 16 \
--eval-batch-size 16 \
--top-k 50 \
--epochs 3

CUDA_VISIBLE_DEVICES=5 python train_ragswm.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Politics_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Politics_dev.jsonl \
--corpus-data-path ../data/splitted_polymarket/polymarket_data_processed_train.jsonl \
--output-dir ../saves/saves_politics \
--cache-dir ../cache/cache_politics \
--predictions-path ../saves/predictions_politics.csv \
--train-batch-size 16 \
--eval-batch-size 16 \
--top-k 50 \
--epochs 3

CUDA_VISIBLE_DEVICES=4 python train_ragswm.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_dev.jsonl \
--corpus-data-path ../data/splitted_polymarket/polymarket_data_processed_train.jsonl \
--output-dir ../saves/saves_all \
--cache-dir ../cache/cache_all \
--predictions-path ../saves/predictions_all.csv \
--train-batch-size 16 \
--eval-batch-size 16 \
--top-k 50 \
--epochs 3


# for sanity check
CUDA_VISIBLE_DEVICES=9 python train_ragswm.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--corpus-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--output-dir ../saves/saves_crypto \
--cache-dir ../cache/cache_crypto \
--predictions-path ../saves/predictions_crypto.csv \
--top-k 50 \
--epochs 300 \
--save-steps 10 \
--eval-steps 10 \
--warmup-steps 0
