CUDA_VISIBLE_DEVICES=8 python train_basicpredictor.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_dev.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--output-dir ../saves/saves_crypto_basicpredictor_without_event_description \
--cache-dir ../cache/cache_crypto_basicpredictor_without_event_description \
--reasoner-cache-dir ../cache/cache_crypto_basicpredictor_without_event_description \
--train-batch-size 1 \
--eval-batch-size 1 \
--reasoner-max-news-items 10 \
--window-size 5 \
--save-steps 200 \
--eval-steps 200 \
--reasoner-cache-dir ../cache/cache_crypto_basicpredictor_without_event_description \
--epochs 10

CUDA_VISIBLE_DEVICES=9 python train_basicpredictor.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_dev.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--output-dir ../saves/saves_crypto_basicpredictor_without_all_event_info \
--cache-dir ../cache/cache_crypto_basicpredictor_without_all_event_info \
--reasoner-cache-dir ../cache/cache_crypto_basicpredictor_without_all_event_info \
--train-batch-size 1 \
--eval-batch-size 1 \
--reasoner-max-news-items 10 \
--window-size 5 \
--save-steps 200 \
--eval-steps 200 \
--reasoner-cache-dir ../cache/cache_crypto_basicpredictor_without_all_event_info \
--epochs 10

CUDA_VISIBLE_DEVICES=7 python train_basicpredictor.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_dev.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--output-dir ../saves/saves_crypto_basicpredictor_window_size_20 \
--cache-dir ../cache/cache_crypto_basicpredictor_window_size_20 \
--reasoner-cache-dir ../cache/cache_crypto_basicpredictor_window_size_20 \
--train-batch-size 1 \
--eval-batch-size 1 \
--reasoner-max-news-items 10 \
--window-size 20 \
--save-steps 200 \
--eval-steps 200 \
--reasoner-cache-dir ../cache/cache_crypto_basicpredictor_window_size_20 \
--epochs 10

CUDA_VISIBLE_DEVICES=6 python train_basicpredictor.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_dev.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--output-dir ../saves/saves_crypto_basicpredictor_window_size_1 \
--cache-dir ../cache/cache_crypto_basicpredictor_window_size_1 \
--reasoner-cache-dir ../cache/cache_crypto_basicpredictor_window_size_1 \
--train-batch-size 1 \
--eval-batch-size 1 \
--reasoner-max-news-items 10 \
--window-size 1 \
--save-steps 200 \
--eval-steps 200 \
--reasoner-cache-dir ../cache/cache_crypto_basicpredictor_window_size_1 \
--epochs 10


CUDA_VISIBLE_DEVICES=8 python train_basicpredictor.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_dev.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--output-dir ../saves/saves_crypto_basicpredictor_max_news_items_3 \
--cache-dir ../cache/cache_crypto_basicpredictor_max_news_items_3 \
--reasoner-cache-dir ../cache/cache_crypto_basicpredictor_max_news_items_3 \
--train-batch-size 1 \
--eval-batch-size 1 \
--reasoner-max-news-items 3 \
--window-size 5 \
--save-steps 200 \
--eval-steps 200 \
--reasoner-cache-dir ../cache/cache_crypto_basicpredictor_max_news_items_3 \
--epochs 10

CUDA_VISIBLE_DEVICES=2 python train_basicpredictor.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_dev.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--output-dir ../saves/saves_crypto_basicpredictor_max_news_items_5 \
--cache-dir ../cache/cache_crypto_basicpredictor_max_news_items_5 \
--reasoner-cache-dir ../cache/cache_crypto_basicpredictor_max_news_items_5 \
--train-batch-size 1 \
--eval-batch-size 1 \
--reasoner-max-news-items 5 \
--window-size 5 \
--save-steps 200 \
--eval-steps 200 \
--reasoner-cache-dir ../cache/cache_crypto_basicpredictor_max_news_items_5 \
--epochs 10

CUDA_VISIBLE_DEVICES=8 python train_basicpredictor.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Sports_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Sports_dev.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--output-dir ../saves/saves_sports_basicpredictor_qwen2.5_1.5B_Instruct \
--model-name Qwen/Qwen2.5-1.5B-Instruct \
--cache-dir ../cache/cache_sports_basicpredictor_qwen2.5_1.5B_Instruct \
--reasoner-cache-dir ../cache/cache_sports_basicpredictor_qwen2.5_1.5B_Instruct \
--train-batch-size 1 \
--eval-batch-size 1 \
--reasoner-max-news-items 10 \
--save-steps 200 \
--eval-steps 200 \
--reasoner-cache-dir ../cache/cache_sports_basicpredictor_qwen2.5_1.5B_Instruct \
--epochs 10

CUDA_VISIBLE_DEVICES=7 python train_basicpredictor.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Other_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Other_dev.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--output-dir ../saves/saves_other_basicpredictor_qwen2.5_1.5B_Instruct \
--model-name Qwen/Qwen2.5-1.5B-Instruct \
--cache-dir ../cache/cache_other_basicpredictor_qwen2.5_1.5B_Instruct \
--reasoner-cache-dir ../cache/cache_other_basicpredictor_qwen2.5_1.5B_Instruct \
--train-batch-size 1 \
--eval-batch-size 1 \
--reasoner-max-news-items 10 \
--save-steps 200 \
--eval-steps 200 \
--reasoner-cache-dir ../cache/cache_other_basicpredictor_qwen2.5_1.5B_Instruct \
--epochs 10

CUDA_VISIBLE_DEVICES=6 python train_basicpredictor.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Election_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Election_dev.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--output-dir ../saves/saves_election_basicpredictor_qwen2.5_1.5B_Instruct \
--model-name Qwen/Qwen2.5-1.5B-Instruct \
--cache-dir ../cache/cache_election_basicpredictor_qwen2.5_1.5B_Instruct \
--reasoner-cache-dir ../cache/cache_election_basicpredictor_qwen2.5_1.5B_Instruct \
--train-batch-size 1 \
--eval-batch-size 1 \
--reasoner-max-news-items 10 \
--save-steps 200 \
--eval-steps 200 \
--reasoner-cache-dir ../cache/cache_election_basicpredictor_qwen2.5_1.5B_Instruct \
--epochs 10

CUDA_VISIBLE_DEVICES=4 python train_basicpredictor.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Politics_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Politics_dev.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--output-dir ../saves/saves_politics_basicpredictor_qwen2.5_1.5B_Instruct \
--model-name Qwen/Qwen2.5-1.5B-Instruct \
--cache-dir ../cache/cache_politics_basicpredictor_qwen2.5_1.5B_Instruct \
--reasoner-cache-dir ../cache/cache_politics_basicpredictor_qwen2.5_1.5B_Instruct \
--train-batch-size 1 \
--eval-batch-size 1 \
--reasoner-max-news-items 10 \
--save-steps 200 \
--eval-steps 200 \
--reasoner-cache-dir ../cache/cache_politics_basicpredictor_qwen2.5_1.5B_Instruct \
--epochs 10

CUDA_VISIBLE_DEVICES=7 python train_basicpredictor.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_dev.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--output-dir ../saves/saves_all_basicpredictor_qwen2.5_1.5B_Instruct \
--model-name Qwen/Qwen2.5-1.5B-Instruct \
--cache-dir ../cache/cache_all_basicpredictor \
--reasoner-cache-dir ../cache/cache_all_basicpredictor \
--train-batch-size 4 \
--eval-batch-size 4 \
--reasoner-max-news-items 10 \
--save-steps 200 \
--eval-steps 200 \
--gradient-checkpointing \
--use-qlora \
--epochs 10


CUDA_VISIBLE_DEVICES=0 python train_basicpredictor.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_dev.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--output-dir ../saves/saves_all_basicpredictor_qwen2.5_0.5B_Instruct \
--model-name /mnt/data/models/Qwen2.5-0.5B-Instruct \
--cache-dir ../cache/cache_all_basicpredictor \
--reasoner-cache-dir ../cache/cache_all_basicpredictor \
--train-batch-size 4 \
--eval-batch-size 4 \
--reasoner-max-news-items 10 \
--save-steps 200 \
--eval-steps 200 \
--gradient-checkpointing \
--epochs 10

CUDA_VISIBLE_DEVICES=1 python train_basicpredictor.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_dev.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--output-dir ../saves/saves_all_basicpredictor_qwen2.5_1.5B_Instruct \
--model-name /mnt/data/models/Qwen2.5-1.5B-Instruct \
--cache-dir ../cache/cache_all_basicpredictor \
--reasoner-cache-dir ../cache/cache_all_basicpredictor \
--train-batch-size 4 \
--eval-batch-size 4 \
--reasoner-max-news-items 10 \
--save-steps 200 \
--eval-steps 200 \
--gradient-checkpointing \
--epochs 10

CUDA_VISIBLE_DEVICES=2 python train_basicpredictor.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_dev.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--output-dir ../saves/saves_all_basicpredictor_qwen2.5_3B_Instruct \
--model-name /mnt/data/models/Qwen2.5-3B-Instruct \
--cache-dir ../cache/cache_all_basicpredictor \
--reasoner-cache-dir ../cache/cache_all_basicpredictor \
--train-batch-size 4 \
--eval-batch-size 4 \
--reasoner-max-news-items 10 \
--save-steps 200 \
--eval-steps 200 \
--gradient-checkpointing \
--epochs 10

CUDA_VISIBLE_DEVICES=3 python train_basicpredictor.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_dev.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--output-dir ../saves/saves_all_basicpredictor_qwen2.5_7B_Instruct_from_sft \
--model-name ../saves/saves_all_basicpredictor_qwen2.5_7B_Instruct/checkpoint-2000 \
--cache-dir ../cache/cache_all_basicpredictor \
--reasoner-cache-dir ../cache/cache_all_basicpredictor \
--train-batch-size 4 \
--eval-batch-size 4 \
--reasoner-max-news-items 10 \
--save-steps 200 \
--eval-steps 200 \
--gradient-checkpointing \
--epochs 10


CUDA_VISIBLE_DEVICES=4 python train_basicpredictor.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_dev.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--output-dir ../saves/saves_all_basicpredictor_qwen2.5_7B_Instruct_ref \
--model-name /mnt/data/models/Qwen2.5-7B-Instruct \
--cache-dir ../cache/cache_all_basicpredictor \
--reasoner-cache-dir ../cache/cache_all_basicpredictor \
--train-batch-size 4 \
--eval-batch-size 4 \
--reasoner-max-news-items 10 \
--save-steps 200 \
--eval-steps 200 \
--gradient-checkpointing \
--epochs 10


# for sanity check
CUDA_VISIBLE_DEVICES=4 python train_basicpredictor.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--output-dir ../saves/saves_crypto_basicpredictor_qwen2.5_1.5B_Instruct_sanitycheck \
--model-name Qwen/Qwen2.5-1.5B-Instruct \
--cache-dir ../cache/cache_crypto_basicpredictor_qwen2.5_1.5B_Instruct_sanitycheck \
--reasoner-cache-dir ../cache/cache_crypto_basicpredictor_qwen2.5_1.5B_Instruct_sanitycheck \
--epochs 200 \
--save-steps 10 \
--eval-steps 10 \
--warmup-steps 0 \
--reasoner-max-news-items 3 \
--sanity-check
