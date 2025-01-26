# for sanity check
CUDA_VISIBLE_DEVICES=2 python train_basicpredictor.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--output-dir ../saves/saves_crypto_basicpredictor_sanitycheck \
--cache-dir ../cache/cache_crypto_basicpredictor_sanitycheck \
--epochs 160 \
--save-steps 10 \
--eval-steps 10 \
--warmup-steps 0 \
--sanity-check