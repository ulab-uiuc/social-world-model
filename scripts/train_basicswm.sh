CUDA_VISIBLE_DEVICES=6 python train_basicswm.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_dev.jsonl \
--output-dir ../saves/saves_crypto_basicswm \
--cache-dir ../cache/cache_crypto_basicswm \
--train-batch-size 16 \
--eval-batch-size 16 \
--epochs 10

CUDA_VISIBLE_DEVICES=2 python train_basicswm.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Sports_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Sports_dev.jsonl \
--output-dir ../saves/saves_sports_basicswm \
--cache-dir ../cache/cache_sports_basicswm \
--train-batch-size 16 \
--eval-batch-size 16 \
--epochs 10

CUDA_VISIBLE_DEVICES=7 python train_basicswm.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Other_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Other_dev.jsonl \
--output-dir ../saves/saves_other_basicswm \
--cache-dir ../cache/cache_other_basicswm \
--train-batch-size 16 \
--eval-batch-size 16 \
--epochs 10

CUDA_VISIBLE_DEVICES=6 python train_basicswm.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Election_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Election_dev.jsonl \
--output-dir ../saves/saves_election_basicswm \
--cache-dir ../cache/cache_election_basicswm \
--train-batch-size 16 \
--eval-batch-size 16 \
--epochs 10

CUDA_VISIBLE_DEVICES=7 python train_basicswm.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Politics_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Politics_dev.jsonl \
--output-dir ../saves/saves_politics_basicswm \
--cache-dir ../cache/cache_politics_basicswm \
--train-batch-size 16 \
--eval-batch-size 16 \
--epochs 10

CUDA_VISIBLE_DEVICES=6 python train_basicswm.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_dev.jsonl \
--output-dir ../saves/saves_all_basicswm \
--cache-dir ../cache/cache_all_basicswm \
--train-batch-size 16 \
--eval-batch-size 16 \
--epochs 10


# for sanity check
CUDA_VISIBLE_DEVICES=0 python train_basicswm.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--output-dir ../saves/saves_crypto_basicswm_sanitycheck \
--cache-dir ../cache/cache_crypto_basicswm_sanitycheck \
--epochs 160 \
--save-steps 10 \
--eval-steps 10 \
--warmup-steps 0 \
--sanity-check
