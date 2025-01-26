CUDA_VISIBLE_DEVICES=4 python test_ragswm.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--corpus-data-path ../data/splitted_polymarket/polymarket_data_processed_train.jsonl \
--model-checkpoint ../saves/saves_crypto_ragswm/checkpoint-11500 \
--output-dir ../saves/saves_crypto_ragswm \
--cache-dir ../cache/cache_crypto_ragswm \
--test-batch-size 20

CUDA_VISIBLE_DEVICES=4 python test_ragswm.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Sports_test.jsonl \
--corpus-data-path ../data/splitted_polymarket/polymarket_data_processed_train.jsonl \
--model-checkpoint ../saves/saves_sports_ragswm/checkpoint-best \
--output-dir ../saves/saves_sports_ragswm \
--cache-dir ../cache/cache_sports_ragswm \
--test-batch-size 20

CUDA_VISIBLE_DEVICES=4 python test_ragswm.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Other_test.jsonl \
--corpus-data-path ../data/splitted_polymarket/polymarket_data_processed_train.jsonl \
--model-checkpoint ../saves/saves_other_ragswm/checkpoint-best \
--output-dir ../saves/saves_other_ragswm \
--cache-dir ../cache/cache_other_ragswm \
--test-batch-size 20

CUDA_VISIBLE_DEVICES=4 python test_ragswm.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Election_test.jsonl \
--corpus-data-path ../data/splitted_polymarket/polymarket_data_processed_train.jsonl \
--model-checkpoint ../saves/saves_election_ragswm/checkpoint-best \
--output-dir ../saves/saves_election_ragswm \
--cache-dir ../cache/cache_election_ragswm \
--test-batch-size 20

CUDA_VISIBLE_DEVICES=4 python test_ragswm.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Politics_test.jsonl \
--corpus-data-path ../data/splitted_polymarket/polymarket_data_processed_train.jsonl \
--model-checkpoint ../saves/saves_politics_ragswm/checkpoint-best \
--output-dir ../saves/saves_politics_ragswm \
--cache-dir ../cache/cache_politics_ragswm \
--test-batch-size 20

CUDA_VISIBLE_DEVICES=4 python test_ragswm.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_test.jsonl \
--corpus-data-path ../data/splitted_polymarket/polymarket_data_processed_train.jsonl \
--model-checkpoint ../saves/saves_all_ragswm/checkpoint-best \
--output-dir ../saves/saves_all_ragswm \
--cache-dir ../cache/cache_all_ragswm \
--test-batch-size 20


# sanity check
CUDA_VISIBLE_DEVICES=0 python test_ragswm.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--corpus-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_train.jsonl \
--model-checkpoint ../saves/saves_crypto_ragswm_sanitycheck/checkpoint-best \
--output-dir ../saves/saves_crypto_ragswm_sanitycheck \
--cache-dir ../cache/cache_crypto_ragswm_sanitycheck \
--test-batch-size 40 \
--sanity-check
