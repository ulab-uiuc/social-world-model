CUDA_VISIBLE_DEVICES=8 python test_basicpredictor.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--model-checkpoint ../saves/saves_crypto_basicpredictor/checkpoint-best \
--output-dir ../saves/saves_crypto_basicpredictor \
--cache-dir ../cache/cache_crypto_basicpredictor \
--test-batch-size 1 \
--reasoner-max-news-items 10


# sanity check
CUDA_VISIBLE_DEVICES=4 python test_basicpredictor.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--model-checkpoint ../saves/saves_crypto_basicpredictor_sanitycheck/checkpoint-best \
--output-dir ../saves/saves_crypto_basicpredictor_sanitycheck \
--cache-dir ../cache/cache_crypto_basicpredictor_sanitycheck \
--test-batch-size 40 \
--reasoner-max-news-items 3 \
--sanity-check
