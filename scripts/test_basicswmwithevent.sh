CUDA_VISIBLE_DEVICES=0 python test_basicswmwithevent.py \
--test-data-path ../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl \
--corpus-news-path ../data/processed_dailynews/dailynews_data_processed.jsonl \
--prior-reasoner-checkpoint /mnt/data/social-world-model/saves/saves_basicpriorreasoner_qwen2.5_0.5B_instruct/checkpoint-2000 \
--predictor-checkpoint /mnt/data/social-world-model/saves/saves_all_basicpredictor_qwen2.5_0.5B_Instruct/checkpoint-best \
--output-dir ../saves/saves_all_basicswmwithevent_qwen2.5_0.5B_instruct \
--prior-reasoner-cache-dir /mnt/data/social-world-model/cache/cache_all_basicpriorreasoner \
--test-batch-size 2 \
--reasoner-max-news-items 10
