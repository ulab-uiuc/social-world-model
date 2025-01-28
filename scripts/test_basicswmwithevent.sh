# get news output from basic predictor (modify the posterior reasoner file)
CUDA_VISIBLE_DEVICES=8 python test_basicswmwithevent.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--prior-reasoner-checkpoint ../saves/saves_crypto_basicpriorreasoner/checkpoint-best \
--predictor-checkpoint ../saves/saves_crypto_basicpredictor/checkpoint-best \
--output-dir ../saves/saves_crypto_basicswmwithevent \
--prior-reasoner-cache-dir ../cache/cache_crypto_basicpriorreasoner \
--test-batch-size 2 \
--reasoner-max-news-items 10
