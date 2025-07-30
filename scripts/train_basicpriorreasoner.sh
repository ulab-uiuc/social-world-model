CUDA_VISIBLE_DEVICES=9 python train_basicpriorreasoner.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_dev.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--output-dir ../saves/saves_crypto_basicpriorreasoner_qwen2.5_1.5B_instruct \
--model-name Qwen/Qwen2.5-1.5B-Instruct \
--cache-dir ../cache/cache_crypto_basicpriorreasoner \
--reasoner-cache-dir ../cache/cache_crypto_basicpriorreasoner \
--train-batch-size 1 \
--eval-batch-size 1 \
--reasoner-max-news-items 10 \
--save-steps 200 \
--eval-steps 200 \
--reasoner-cache-dir ../cache/cache_crypto_basicpriorreasoner \
--epochs 20

CUDA_VISIBLE_DEVICES=8 python train_basicpriorreasoner.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Sports_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Sports_dev.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--output-dir ../saves/saves_sports_basicpriorreasoner_qwen2.5_1.5B_instruct \
--model-name Qwen/Qwen2.5-1.5B-Instruct \
--cache-dir ../cache/cache_sports_basicpriorreasoner \
--reasoner-cache-dir ../cache/cache_sports_basicpriorreasoner \
--train-batch-size 1 \
--eval-batch-size 1 \
--reasoner-max-news-items 10 \
--save-steps 200 \
--eval-steps 200 \
--reasoner-cache-dir ../cache/cache_sports_basicpriorreasoner \
--epochs 20

CUDA_VISIBLE_DEVICES=7 python train_basicpriorreasoner.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Other_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Other_dev.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--output-dir ../saves/saves_other_basicpriorreasoner_qwen2.5_1.5B_instruct \
--model-name Qwen/Qwen2.5-1.5B-Instruct \
--cache-dir ../cache/cache_other_basicpriorreasoner \
--reasoner-cache-dir ../cache/cache_other_basicpriorreasoner \
--train-batch-size 1 \
--eval-batch-size 1 \
--reasoner-max-news-items 10 \
--save-steps 200 \
--eval-steps 200 \
--reasoner-cache-dir ../cache/cache_other_basicpriorreasoner \
--epochs 20

CUDA_VISIBLE_DEVICES=6 python train_basicpriorreasoner.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Election_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Election_dev.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--output-dir ../saves/saves_election_basicpriorreasoner_qwen2.5_1.5B_instruct \
--model-name Qwen/Qwen2.5-1.5B-Instruct \
--cache-dir ../cache/cache_election_basicpriorreasoner \
--reasoner-cache-dir ../cache/cache_election_basicpriorreasoner \
--train-batch-size 1 \
--eval-batch-size 1 \
--reasoner-max-news-items 10 \
--save-steps 200 \
--eval-steps 200 \
--reasoner-cache-dir ../cache/cache_election_basicpriorreasoner \
--epochs 20

CUDA_VISIBLE_DEVICES=2 python train_basicpriorreasoner.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Politics_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Politics_dev.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--output-dir ../saves/saves_politics_basicpriorreasoner_qwen2.5_1.5B_instruct \
--model-name Qwen/Qwen2.5-1.5B-Instruct \
--cache-dir ../cache/cache_politics_basicpriorreasoner \
--reasoner-cache-dir ../cache/cache_politics_basicpriorreasoner \
--train-batch-size 1 \
--eval-batch-size 1 \
--reasoner-max-news-items 10 \
--save-steps 200 \
--eval-steps 200 \
--reasoner-cache-dir ../cache/cache_politics_basicpriorreasoner \
--epochs 20


# for sanity check
CUDA_VISIBLE_DEVICES=9 python train_basicpriorreasoner.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--output-dir ../saves/saves_crypto_basicpriorreasoner_sanitycheck \
--cache-dir ../cache/cache_crypto_basicpriorreasoner_sanitycheck \
--reasoner-cache-dir ../cache/cache_crypto_basicpriorreasoner_sanitycheck \
--epochs 200 \
--save-steps 10 \
--eval-steps 10 \
--warmup-steps 0 \
--reasoner-max-news-items 3 \
--sanity-check



CUDA_VISIBLE_DEVICES=0 python train_basicpriorreasoner.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_dev.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--output-dir ../saves/saves_basicpriorreasoner_qwen2.5_0.5B_instruct \
--model-name /mnt/data/models/Qwen2.5-0.5B-Instruct \
--cache-dir ../cache/cache_all_basicpriorreasoner \
--reasoner-cache-dir ../cache/cache_all_basicpriorreasoner \
--train-batch-size 1 \
--eval-batch-size 1 \
--reasoner-max-news-items 10 \
--save-steps 200 \
--eval-steps 200 \
--epochs 20


CUDA_VISIBLE_DEVICES=1 python train_basicpriorreasoner.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_dev.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--output-dir ../saves/saves_basicpriorreasoner_qwen2.5_1.5B_instruct \
--model-name /mnt/data/models/Qwen2.5-1.5B-Instruct \
--cache-dir ../cache/cache_all_basicpriorreasoner \
--reasoner-cache-dir ../cache/cache_all_basicpriorreasoner \
--train-batch-size 1 \
--eval-batch-size 1 \
--reasoner-max-news-items 10 \
--save-steps 200 \
--eval-steps 200 \
--epochs 20

CUDA_VISIBLE_DEVICES=2 python train_basicpriorreasoner.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_dev.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--output-dir ../saves/saves_basicpriorreasoner_qwen2.5_3B_instruct \
--model-name /mnt/data/models/Qwen2.5-3B-Instruct \
--cache-dir ../cache/cache_all_basicpriorreasoner \
--reasoner-cache-dir ../cache/cache_all_basicpriorreasoner \
--train-batch-size 1 \
--eval-batch-size 1 \
--reasoner-max-news-items 10 \
--save-steps 200 \
--eval-steps 200 \
--epochs 20

CUDA_VISIBLE_DEVICES=3 python train_basicpriorreasoner.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_dev.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--output-dir ../saves/saves_basicpriorreasoner_qwen2.5_7B_instruct \
--model-name /mnt/data/models/Qwen2.5-7B-Instruct \
--cache-dir ../cache/cache_all_basicpriorreasoner \
--reasoner-cache-dir ../cache/cache_all_basicpriorreasoner \
--train-batch-size 1 \
--eval-batch-size 1 \
--reasoner-max-news-items 10 \
--save-steps 200 \
--eval-steps 200 \
--epochs 20