import os
import subprocess
import sys
import time
from multiprocessing import Process


def run_flask_app():
    print('Starting Flask application...')
    app_path = os.path.join(os.path.dirname(__file__), 'app.py')
    subprocess.run([sys.executable, app_path])


def run_news_scraper():
    print('Starting news scraper...')
    scraper_path = os.path.join(os.path.dirname(__file__), 'news_scraper.py')
    subprocess.run([sys.executable, scraper_path])


def run_polymarket_updater():
    print('Starting Polymarket updater...')
    updater_path = os.path.join(os.path.dirname(__file__), 'polymarket_updater.py')
    subprocess.run([sys.executable, updater_path])


def main():
    try:
        flask_process = Process(target=run_flask_app)
        news_process = Process(target=run_news_scraper)
        polymarket_process = Process(target=run_polymarket_updater)

        flask_process.start()
        news_process.start()
        polymarket_process.start()

        print('All services started successfully!')

        while True:
            if not flask_process.is_alive():
                print('Flask application stopped unexpectedly, restarting...')
                flask_process = Process(target=run_flask_app)
                flask_process.start()

            if not news_process.is_alive():
                print('News scraper stopped unexpectedly, restarting...')
                news_process = Process(target=run_news_scraper)
                news_process.start()

            if not polymarket_process.is_alive():
                print('Polymarket updater stopped unexpectedly, restarting...')
                polymarket_process = Process(target=run_polymarket_updater)
                polymarket_process.start()

            time.sleep(60)

    except KeyboardInterrupt:
        print('Shutting down all services...')
        if flask_process.is_alive():
            flask_process.terminate()
        if news_process.is_alive():
            news_process.terminate()
        if polymarket_process.is_alive():
            polymarket_process.terminate()

        flask_process.join()
        news_process.join()
        polymarket_process.join()

        print('All services stopped.')


if __name__ == '__main__':
    main()
