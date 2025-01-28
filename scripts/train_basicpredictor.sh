CUDA_VISIBLE_DEVICES=2 python train_basicpredictor.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_dev.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--output-dir ../saves/saves_crypto_basicpredictor \
--cache-dir ../cache/cache_crypto_basicpredictor \
--reasoner-cache-dir ../cache/cache_crypto_basicpredictor \
--train-batch-size 1 \
--eval-batch-size 1 \
--reasoner-max-news-items 10 \
--save-steps 200 \
--eval-steps 200 \
--reasoner-cache-dir ../cache/cache_crypto_basicpredictor \
--epochs 10

CUDA_VISIBLE_DEVICES=4 python train_basicpredictor.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Sports_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Sports_dev.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--output-dir ../saves/saves_sports_basicpredictor \
--cache-dir ../cache/cache_sports_basicpredictor \
--reasoner-cache-dir ../cache/cache_sports_basicpredictor \
--train-batch-size 1 \
--eval-batch-size 1 \
--reasoner-max-news-items 10 \
--save-steps 200 \
--eval-steps 200 \
--reasoner-cache-dir ../cache/cache_sports_basicpredictor \
--epochs 10

CUDA_VISIBLE_DEVICES=4 python train_basicpredictor.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Other_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Other_dev.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--output-dir ../saves/saves_other_basicpredictor \
--cache-dir ../cache/cache_other_basicpredictor \
--reasoner-cache-dir ../cache/cache_other_basicpredictor \
--train-batch-size 1 \
--eval-batch-size 1 \
--reasoner-max-news-items 10 \
--save-steps 200 \
--eval-steps 200 \
--reasoner-cache-dir ../cache/cache_other_basicpredictor \
--epochs 10

CUDA_VISIBLE_DEVICES=9 python train_basicpredictor.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Election_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Election_dev.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--output-dir ../saves/saves_election_basicpredictor \
--cache-dir ../cache/cache_election_basicpredictor \
--reasoner-cache-dir ../cache/cache_election_basicpredictor \
--train-batch-size 1 \
--eval-batch-size 1 \
--reasoner-max-news-items 10 \
--save-steps 200 \
--eval-steps 200 \
--reasoner-cache-dir ../cache/cache_election_basicpredictor \
--epochs 10

CUDA_VISIBLE_DEVICES=7 python train_basicpredictor.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Politics_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Politics_dev.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--output-dir ../saves/saves_politics_basicpredictor \
--cache-dir ../cache/cache_politics_basicpredictor \
--reasoner-cache-dir ../cache/cache_politics_basicpredictor \
--train-batch-size 1 \
--eval-batch-size 1 \
--reasoner-max-news-items 10 \
--save-steps 200 \
--eval-steps 200 \
--reasoner-cache-dir ../cache/cache_politics_basicpredictor \
--epochs 10

CUDA_VISIBLE_DEVICES=9 python train_basicpredictor.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_dev.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--output-dir ../saves/saves_all_basicpredictor \
--cache-dir ../cache/cache_all_basicpredictor \
--reasoner-cache-dir ../cache/cache_all_basicpredictor \
--train-batch-size 4 \
--eval-batch-size 4 \
--reasoner-max-news-items 10 \
--save-steps 200 \
--eval-steps 200 \
--reasoner-cache-dir ../cache/cache_all_basicpredictor \
--epochs 10


# for sanity check
CUDA_VISIBLE_DEVICES=4 python train_basicpredictor.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--output-dir ../saves/saves_crypto_basicpredictor_sanitycheck \
--cache-dir ../cache/cache_crypto_basicpredictor_sanitycheck \
--reasoner-cache-dir ../cache/cache_crypto_basicpredictor_sanitycheck \
--epochs 200 \
--save-steps 10 \
--eval-steps 10 \
--warmup-steps 0 \
--reasoner-max-news-items 3 \
--sanity-check
