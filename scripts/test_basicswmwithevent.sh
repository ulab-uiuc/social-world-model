# get news output from basic predictor (modify the posterior reasoner file)
CUDA_VISIBLE_DEVICES=9 python test_basicswmwithevent.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--prior-reasoner-checkpoint ../saves/saves_crypto_basicpriorreasoner/checkpoint-best \
--predictor-checkpoint ../saves/saves_crypto_basicpredictor/checkpoint-best \
--output-dir ../saves/saves_crypto_basicswmwithevent \
--prior-reasoner-cache-dir ../cache/cache_crypto_basicpriorreasoner \
--test-batch-size 2 \
--reasoner-max-news-items 10

CUDA_VISIBLE_DEVICES=9 python test_basicswmwithevent.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Sports_test.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--prior-reasoner-checkpoint ../saves/saves_sports_basicpriorreasoner/checkpoint-best \
--predictor-checkpoint ../saves/saves_sports_basicpredictor/checkpoint-best \
--output-dir ../saves/saves_sports_basicswmwithevent \
--prior-reasoner-cache-dir ../cache/cache_sports_basicpriorreasoner \
--test-batch-size 2 \
--reasoner-max-news-items 10

CUDA_VISIBLE_DEVICES=9 python test_basicswmwithevent.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Other_test.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--prior-reasoner-checkpoint ../saves/saves_other_basicpriorreasoner/checkpoint-best \
--predictor-checkpoint ../saves/saves_other_basicpredictor/checkpoint-best \
--output-dir ../saves/saves_other_basicswmwithevent \
--prior-reasoner-cache-dir ../cache/cache_other_basicpriorreasoner \
--test-batch-size 2 \
--reasoner-max-news-items 10

CUDA_VISIBLE_DEVICES=9 python test_basicswmwithevent.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Election_test.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--prior-reasoner-checkpoint ../saves/saves_election_basicpriorreasoner/checkpoint-best \
--predictor-checkpoint ../saves/saves_election_basicpredictor/checkpoint-best \
--output-dir ../saves/saves_election_basicswmwithevent \
--prior-reasoner-cache-dir ../cache/cache_election_basicpriorreasoner \
--test-batch-size 2 \
--reasoner-max-news-items 10

CUDA_VISIBLE_DEVICES=9 python test_basicswmwithevent.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Politics_test.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--prior-reasoner-checkpoint ../saves/saves_politics_basicpriorreasoner/checkpoint-best \
--predictor-checkpoint ../saves/saves_politics_basicpredictor/checkpoint-best \
--output-dir ../saves/saves_politics_basicswmwithevent \
--prior-reasoner-cache-dir ../cache/cache_politics_basicpriorreasoner \
--test-batch-size 2 \
--reasoner-max-news-items 10
