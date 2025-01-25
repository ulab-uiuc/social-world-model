CUDA_VISIBLE_DEVICES=3 python test_basicswm.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--corpus-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--model-checkpoint ../saves/saves_crypto_basicswm/checkpoint-50 \
--output-dir ../saves/saves_crypto_basicswm \
--predictions-path ../saves/predictions_crypto_basicswm.csv \
--test-batch-size 40
