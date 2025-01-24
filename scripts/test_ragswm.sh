CUDA_VISIBLE_DEVICES=3 python test_ragswm.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--corpus-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--model-checkpoint ../saves/saves_crypto/checkpoint-10 \
--output-dir ../saves/saves_crypto \
--predictions-path ../saves/predictions_crypto.csv \
--batch-size 40
