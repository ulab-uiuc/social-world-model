from datetime import datetime, timedelta, timezone

import feedparser
import pymongo
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

client = pymongo.MongoClient('mongodb://localhost:27017/')
db = client['electionDB']
news_collection = db['news']


RSS_FEEDS = {
    'https://feeds.content.dowjones.io/public/rss/RSSOpinion': ['opinion'],
    'https://feeds.content.dowjones.io/public/rss/RSSWorldNews': ['world'],
    'https://feeds.content.dowjones.io/public/rss/WSJcomUSBusiness': ['business'],
    'https://feeds.content.dowjones.io/public/rss/RSSMarketsMain': [
        'finance',
        'markets',
    ],
    'https://feeds.content.dowjones.io/public/rss/RSSWSJD': ['technology'],
    'https://feeds.content.dowjones.io/public/rss/RSSLifestyle': ['lifestyle'],
    'https://feeds.content.dowjones.io/public/rss/RSSUSnews': ['us'],
    'https://feeds.content.dowjones.io/public/rss/socialpoliticsfeed': ['politics'],
    'https://feeds.content.dowjones.io/public/rss/socialeconomyfeed': ['economy'],
    'https://feeds.content.dowjones.io/public/rss/RSSArtsCulture': ['arts'],
    'https://feeds.content.dowjones.io/public/rss/latestnewsrealestate': [
        'real estate'
    ],
    'https://feeds.content.dowjones.io/public/rss/RSSPersonalFinance': [
        'personal finance'
    ],
    'https://feeds.content.dowjones.io/public/rss/socialhealth': ['health'],
    'https://feeds.content.dowjones.io/public/rss/RSSStyle': ['style'],
    'https://feeds.content.dowjones.io/public/rss/rsssportsfeed': ['sports'],
}


def delete_old_news():
    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
    result = news_collection.delete_many(
        {'timestamp': {'$lt': cutoff_time.isoformat()}}
    )
    print(f'Deleted {result.deleted_count} old news')


def is_today(published_time):
    try:
        news_time = datetime.strptime(
            published_time, '%a, %d %b %Y %H:%M:%S %Z'
        ).replace(tzinfo=timezone.utc)
        return news_time.date() == datetime.now(timezone.utc).date()
    except Exception as e:
        print(f'Time parsing error: {e}')
        return False


def fetch_news():
    delete_old_news()
    news_data = []

    for feed_url, tags in RSS_FEEDS.items():
        print(f'Fetching {feed_url}')
        feed = feedparser.parse(feed_url)

        for entry in feed.entries:
            if is_today(entry.published):
                title = entry.title
                timestamp = entry.published
                link = entry.link

                existing_news = news_collection.find_one(
                    {'$or': [{'title': title}, {'link': link}]}
                )
                if existing_news:
                    print(f'Skipped: {title}')
                    continue

                news_data.append(
                    {'title': title, 'timestamp': timestamp, 'link': link, 'tags': tags}
                )

    if news_data:
        news_collection.insert_many(news_data)
        print(f'Added {len(news_data)} news')
    else:
        print('No new news')


scheduler = BackgroundScheduler()
scheduler.add_job(fetch_news, 'interval', hours=6)
scheduler.start()


@app.route('/api/news', methods=['GET'])
def get_news():
    tag_filter = request.args.get('tag')
    query = {}

    if tag_filter:
        query['tags'] = tag_filter

    news = list(news_collection.find(query, {'_id': 0}).sort('timestamp', -1))

    return jsonify(news)


if __name__ == '__main__':
    print('Starting news scraper...')
    fetch_news()
    app.run(host='0.0.0.0', port=5001, debug=True)
