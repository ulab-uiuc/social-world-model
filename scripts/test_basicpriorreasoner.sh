# sanity check
CUDA_VISIBLE_DEVICES=4 python test_basicpriorreasoner.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--model-checkpoint ../saves/saves_crypto_basicpriorreasoner_sanitycheck/checkpoint-best \
--output-dir ../saves/saves_crypto_basicpriorreasoner_sanitycheck \
--cache-dir ../cache/cache_crypto_basicpriorreasoner_sanitycheck \
--test-batch-size 40 \
--reasoner-max-news-items 3 \
--sanity-check
