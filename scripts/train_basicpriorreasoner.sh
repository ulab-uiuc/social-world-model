# for sanity check
CUDA_VISIBLE_DEVICES=3 python train_basicpriorreasoner.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--output-dir ../saves/saves_crypto_basicpriorreasoner_sanitycheck \
--cache-dir ../cache/cache_crypto_basicpriorreasoner_sanitycheck \
--reasoner-cache-dir ../cache/cache_crypto_basicpriorreasoner_sanitycheck \
--epochs 200 \
--save-steps 10 \
--eval-steps 10 \
--warmup-steps 0 \
--reasoner-max-news-items 3 \
--sanity-check
