CUDA_VISIBLE_DEVICES=8 python test_basicswmwithevent.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--model-checkpoint ../saves/saves_crypto_basicpredictor/checkpoint-best \
--prior-model-checkpoint ../saves/saves_crypto_basicpriorreasoner/checkpoint-best \
--output-dir ../saves/saves_crypto_basicswmwithevent \
--cache-dir ../cache/cache_crypto_basicswmwithevent \
--test-batch-size 1 \
--reasoner-max-news-items 10
