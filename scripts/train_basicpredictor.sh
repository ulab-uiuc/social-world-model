CUDA_VISIBLE_DEVICES=4 python train_basicpredictor.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_train.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_dev.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--output-dir ../saves/saves_crypto_basicpredictor \
--cache-dir ../cache/cache_crypto_basicpredictor \
--reasoner-cache-dir ../cache/cache_crypto_basicpredictor \
--train-batch-size 4 \
--eval-batch-size 4 \
--reasoner-max-news-items 5 \
--reasoner-cache-dir ../cache/cache_crypto_basicpredictor \
--epochs 10

# for sanity check
CUDA_VISIBLE_DEVICES=4 python train_basicpredictor.py \
--train-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--valid-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--output-dir ../saves/saves_crypto_basicpredictor_sanitycheck \
--cache-dir ../cache/cache_crypto_basicpredictor_sanitycheck \
--reasoner-cache-dir ../cache/cache_crypto_basicpredictor_sanitycheck \
--epochs 200 \
--save-steps 10 \
--eval-steps 10 \
--warmup-steps 0 \
--reasoner-max-news-items 3 \
--sanity-check
