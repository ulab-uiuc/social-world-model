CUDA_VISIBLE_DEVICES=9 python test_basicswm_zeroshot.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--output-dir ../saves/saves_crypto_basicswm_zeroshot \
--cache-dir ../cache/cache_crypto_basicswm_zeroshot \
--test-batch-size 20

CUDA_VISIBLE_DEVICES=8 python test_basicswm_zeroshot.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Sports_test.jsonl \
--output-dir ../saves/saves_sports_basicswm_zeroshot \
--cache-dir ../cache/cache_sports_basicswm_zeroshot \
--test-batch-size 20

CUDA_VISIBLE_DEVICES=8 python test_basicswm_zeroshot.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Other_test.jsonl \
--output-dir ../saves/saves_other_basicswm_zeroshot \
--cache-dir ../cache/cache_other_basicswm_zeroshot \
--test-batch-size 20

CUDA_VISIBLE_DEVICES=8 python test_basicswm_zeroshot.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Election_test.jsonl \
--output-dir ../saves/saves_election_basicswm_zeroshot \
--cache-dir ../cache/cache_election_basicswm_zeroshot \
--test-batch-size 20

CUDA_VISIBLE_DEVICES=8 python test_basicswm_zeroshot.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Politics_test.jsonl \
--output-dir ../saves/saves_politics_basicswm_zeroshot \
--cache-dir ../cache/cache_politics_basicswm_zeroshot \
--test-batch-size 20

CUDA_VISIBLE_DEVICES=2 python test_basicswm_zeroshot.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_test.jsonl \
--output-dir ../saves/saves_all_basicswm_zeroshot \
--cache-dir ../cache/cache_all_basicswm_zeroshot \
--test-batch-size 20

# sanity check
CUDA_VISIBLE_DEVICES=0 python test_basicswm_zeroshot.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--output-dir ../saves/saves_crypto_basicswm_zeroshot_sanitycheck \
--cache-dir ../cache/cache_crypto_basicswm_zeroshot_sanitycheck \
--test-batch-size 40 \
--sanity-check
