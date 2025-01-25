# sanity check
CUDA_VISIBLE_DEVICES=2 python test_ragswm.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--corpus-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_train.jsonl \
--model-checkpoint ../saves/saves_crypto_ragswm_sanitycheck/checkpoint-best \
--output-dir ../saves/saves_crypto_ragswm_sanitycheck \
--cache-dir ../cache/cache_crypto_ragswm_sanitycheck \
--test-batch-size 40 \
--sanity-check
