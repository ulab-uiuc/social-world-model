CUDA_VISIBLE_DEVICES=5 python test_basicpredictor.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--model-checkpoint ../saves/saves_crypto_basicpredictor/checkpoint-best \
--output-dir ../saves/saves_crypto_basicpredictor \
--cache-dir ../cache/cache_crypto_basicpredictor \
--test-batch-size 1 \
--reasoner-max-news-items 10

CUDA_VISIBLE_DEVICES=8 python test_basicpredictor.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Sports_test.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--model-checkpoint ../saves/saves_sports_basicpredictor/checkpoint-best \
--output-dir ../saves/saves_sports_basicpredictor \
--cache-dir ../cache/cache_sports_basicpredictor \
--test-batch-size 1 \
--reasoner-max-news-items 10

CUDA_VISIBLE_DEVICES=8 python test_basicpredictor.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Other_test.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--model-checkpoint ../saves/saves_other_basicpredictor/checkpoint-best \
--output-dir ../saves/saves_other_basicpredictor \
--cache-dir ../cache/cache_other_basicpredictor \
--test-batch-size 1 \
--reasoner-max-news-items 10

CUDA_VISIBLE_DEVICES=8 python test_basicpredictor.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Election_test.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--model-checkpoint ../saves/saves_election_basicpredictor/checkpoint-best \
--output-dir ../saves/saves_election_basicpredictor \
--cache-dir ../cache/cache_election_basicpredictor \
--test-batch-size 1 \
--reasoner-max-news-items 10

CUDA_VISIBLE_DEVICES=8 python test_basicpredictor.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Politics_test.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--model-checkpoint ../saves/saves_politics_basicpredictor/checkpoint-best \
--output-dir ../saves/saves_politics_basicpredictor \
--cache-dir ../cache/cache_politics_basicpredictor \
--test-batch-size 1 \
--reasoner-max-news-items 10

CUDA_VISIBLE_DEVICES=5 python test_basicpredictor.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_test.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--model-checkpoint ../saves/saves_all_basicpredictor_qwen2.5_0.5B_Instruct/checkpoint-best \
--output-dir ../saves/saves_all_basicpredictor_qwen2.5_0.5B_Instruct \
--cache-dir ../cache/cache_all_basicpredictor \
--test-batch-size 1 \
--reasoner-max-news-items 10 \


# sanity check
CUDA_VISIBLE_DEVICES=4 python test_basicpredictor.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--model-checkpoint ../saves/saves_crypto_basicpredictor_sanitycheck/checkpoint-best \
--output-dir ../saves/saves_crypto_basicpredictor_sanitycheck \
--cache-dir ../cache/cache_crypto_basicpredictor_sanitycheck \
--test-batch-size 40 \
--reasoner-max-news-items 3 \
--sanity-check


CUDA_VISIBLE_DEVICES=0 python test_basicpredictor.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--model-checkpoint ../saves/saves_all_basicpredictor_qwen2.5_0.5B_Instruct/checkpoint-best \
--output-dir ../saves/saves_crypto_basicpredictor_Qwen2.5-0.5B-Instruct \
--cache-dir ../cache/cache_crypto_basicpredictor \
--test-batch-size 1 \
--reasoner-max-news-items 10

CUDA_VISIBLE_DEVICES=0 python test_basicpredictor.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--model-checkpoint ../saves/saves_all_basicpredictor_qwen2.5_1.5B_Instruct/checkpoint-best \
--output-dir ../saves/saves_crypto_basicpredictor_Qwen2.5-1.5B-Instruct \
--cache-dir ../cache/cache_crypto_basicpredictor \
--test-batch-size 1 \
--reasoner-max-news-items 10

CUDA_VISIBLE_DEVICES=0 python test_basicpredictor.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--model-checkpoint ../saves/saves_all_basicpredictor_qwen2.5_3B_Instruct/checkpoint-3600 \
--output-dir ../saves/saves_crypto_basicpredictor_Qwen2.5-3B-Instruct_ckpt3600 \
--cache-dir ../cache/cache_crypto_basicpredictor \
--test-batch-size 1 \
--reasoner-max-news-items 10

CUDA_VISIBLE_DEVICES=0 python test_basicpredictor.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--model-checkpoint ../saves/saves_all_basicpredictor_qwen2.5_7B_Instruct/checkpoint-1600 \
--output-dir ../saves/saves_crypto_basicpredictor_Qwen2.5-7B-Instruct_ckpt1600 \
--cache-dir ../cache/cache_crypto_basicpredictor \
--test-batch-size 1 \
--reasoner-max-news-items 10



CUDA_VISIBLE_DEVICES=6 python test_basicpredictor.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--model-checkpoint ../saves/saves_all_basicpredictor_qwen2.5_7B_Instruct/checkpoint-2000 \
--output-dir ../saves/saves_crypto_basicpredictor_Qwen2.5-7B-Instruct \
--cache-dir ../cache/cache_crypto_basicpredictor \
--test-batch-size 1 \
--reasoner-max-news-items 10

CUDA_VISIBLE_DEVICES=5 python test_basicpredictor.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Politics_test.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--model-checkpoint ../saves/saves_all_basicpredictor_qwen2.5_7B_Instruct/checkpoint-2000 \
--output-dir ../saves/saves_politics_basicpredictor_Qwen2.5-7B-Instruct \
--cache-dir ../cache/cache_politics_basicpredictor \
--test-batch-size 1 \
--reasoner-max-news-items 10

CUDA_VISIBLE_DEVICES=2 python test_basicpredictor.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Sports_test.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--model-checkpoint ../saves/saves_all_basicpredictor_qwen2.5_7B_Instruct/checkpoint-2000 \
--output-dir ../saves/saves_sports_basicpredictor_Qwen2.5-7B-Instruct \
--cache-dir ../cache/cache_sports_basicpredictor \
--test-batch-size 1 \
--reasoner-max-news-items 10


CUDA_VISIBLE_DEVICES=1 python test_basicpredictor.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Election_test.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--model-checkpoint ../saves/saves_all_basicpredictor_qwen2.5_7B_Instruct/checkpoint-2000 \
--output-dir ../saves/saves_election_basicpredictor_Qwen2.5-7B-Instruct \
--cache-dir ../cache/cache_election_basicpredictor \
--test-batch-size 1 \
--reasoner-max-news-items 10


CUDA_VISIBLE_DEVICES=0 python test_basicpredictor.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Other_test.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--model-checkpoint ../saves/saves_all_basicpredictor_qwen2.5_7B_Instruct/checkpoint-2000 \
--output-dir ../saves/saves_other_basicpredictor_Qwen2.5-7B-Instruct \
--cache-dir ../cache/cache_other_basicpredictor \
--test-batch-size 1 \
--reasoner-max-news-items 10

CUDA_VISIBLE_DEVICES=3 python test_basicpredictor.py \
--test-data-path ../data/polymarket_data_processed_Other_test.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--model-checkpoint ../saves/saves_all_basicpredictor_qwen2.5_0.5B_Instruct_0730/checkpoint-best \
--output-dir ../saves/saves_other_basicpredictor_Qwen2.5-0.5B-Instruct_0730 \
--cache-dir ../cache/cache_other_basicpredictor \
--test-batch-size 1 \
--reasoner-max-news-items 10

CUDA_VISIBLE_DEVICES=3 python test_basicpredictor.py \
--test-data-path ../data/polymarket_data_processed_Other_test.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--model-checkpoint ../saves/saves_all_basicpredictor_qwen2.5_0.5B_Instruct_0730/checkpoint-best \
--output-dir ../saves/saves_other_basicpredictor_Qwen2.5-0.5B-Instruct_0730 \
--cache-dir ../cache/cache_other_basicpredictor \
--test-batch-size 1 \
--reasoner-max-news-items 10

CUDA_VISIBLE_DEVICES=3 python test_basicpredictor.py \
--test-data-path ../data/polymarket_data_processed_Crypto_test_filtered.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--model-checkpoint ../saves/saves_all_basicpredictor_qwen2.5_0.5B_Instruct_0730/checkpoint-best \
--output-dir ../saves/saves_crypto_filtered_basicpredictor_Qwen2.5-0.5B-Instruct_0730 \
--cache-dir ../cache/cache_crypto_basicpredictor \
--test-batch-size 1 \
--reasoner-max-news-items 10

CUDA_VISIBLE_DEVICES=3 python test_basicpredictor.py \
--test-data-path ../data/polymarket_data_processed_Sports_test.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--model-checkpoint ../saves/saves_all_basicpredictor_qwen2.5_0.5B_Instruct_0730/checkpoint-best \
--output-dir ../saves/saves_sports_basicpredictor_Qwen2.5-0.5B-Instruct_0730 \
--cache-dir ../cache/cache_sports_basicpredictor \
--test-batch-size 1 \
--reasoner-max-news-items 10

CUDA_VISIBLE_DEVICES=3 python test_basicpredictor.py \
--test-data-path ../data/polymarket_data_processed_Election_test_filtered.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--model-checkpoint ../saves/saves_all_basicpredictor_qwen2.5_0.5B_Instruct_0730/checkpoint-best \
--output-dir ../saves/saves_election_filtered_basicpredictor_Qwen2.5-0.5B-Instruct_0730 \
--cache-dir ../cache/cache_election_basicpredictor \
--test-batch-size 1 \
--reasoner-max-news-items 10

CUDA_VISIBLE_DEVICES=3 python test_basicpredictor.py \
--test-data-path ../data/polymarket_data_processed_Politics_test_filtered.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--model-checkpoint ../saves/saves_all_basicpredictor_qwen2.5_0.5B_Instruct_0730/checkpoint-best \
--output-dir ../saves/saves_politics_filtered_basicpredictor_Qwen2.5-0.5B-Instruct_0730 \
--cache-dir ../cache/cache_politics_basicpredictor \
--test-batch-size 1 \
--reasoner-max-news-items 10
