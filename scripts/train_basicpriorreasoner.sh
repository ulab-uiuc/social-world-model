CUDA_VISIBLE_DEVICES=9 python train_basicpriorreasoner.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_dev.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--output-dir ../saves/saves_crypto_basicpriorreasoner \
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
--output-dir ../saves/saves_sports_basicpriorreasoner \
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
--output-dir ../saves/saves_other_basicpriorreasoner \
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
--output-dir ../saves/saves_election_basicpriorreasoner \
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
--output-dir ../saves/saves_politics_basicpriorreasoner \
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
