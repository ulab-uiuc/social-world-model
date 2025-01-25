# sanity check
CUDA_VISIBLE_DEVICES=3 python test_basicswm.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--model-checkpoint ../saves/saves_crypto_basicswm_sanitycheck/checkpoint-best \
--output-dir ../saves/saves_crypto_basicswm_sanitycheck \
--test-batch-size 40 \
--sanity-check
