CUDA_VISIBLE_DEVICES=2 python test_basicswm.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--model-checkpoint ../saves/saves_crypto_basicswm/checkpoint-best \
--output-dir ../saves/saves_crypto_basicswm \
--cache-dir ../cache/cache_crypto_basicswm \
--test-batch-size 20

CUDA_VISIBLE_DEVICES=2 python test_basicswm.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Sports_test.jsonl \
--model-checkpoint ../saves/saves_sports_basicswm/checkpoint-best \
--output-dir ../saves/saves_sports_basicswm \
--cache-dir ../cache/cache_sports_basicswm \
--test-batch-size 20

CUDA_VISIBLE_DEVICES=2 python test_basicswm.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Other_test.jsonl \
--model-checkpoint ../saves/saves_other_basicswm/checkpoint-best \
--output-dir ../saves/saves_other_basicswm \
--cache-dir ../cache/cache_other_basicswm \
--test-batch-size 20

CUDA_VISIBLE_DEVICES=2 python test_basicswm.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Election_test.jsonl \
--model-checkpoint ../saves/saves_election_basicswm/checkpoint-best \
--output-dir ../saves/saves_election_basicswm \
--cache-dir ../cache/cache_election_basicswm \
--test-batch-size 20

CUDA_VISIBLE_DEVICES=2 python test_basicswm.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Politics_test.jsonl \
--model-checkpoint ../saves/saves_politics_basicswm/checkpoint-best \
--output-dir ../saves/saves_politics_basicswm \
--cache-dir ../cache/cache_politics_basicswm \
--test-batch-size 20

CUDA_VISIBLE_DEVICES=2 python test_basicswm.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_test.jsonl \
--model-checkpoint ../saves/saves_all_basicswm/checkpoint-best \
--output-dir ../saves/saves_all_basicswm \
--cache-dir ../cache/cache_all_basicswm \
--test-batch-size 20

# sanity check
CUDA_VISIBLE_DEVICES=3 python test_basicswm.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--model-checkpoint ../saves/saves_crypto_basicswm_sanitycheck/checkpoint-best \
--output-dir ../saves/saves_crypto_basicswm_sanitycheck \
--cache-dir ../cache/cache_crypto_basicswm_sanitycheck \
--test-batch-size 40 \
--sanity-check
