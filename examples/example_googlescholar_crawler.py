from swm.utils.crawler import GoogleScholarCrawler


def main():
    crawler = GoogleScholarCrawler(
        input_file='../data/high_impact_paper_bench.json',
        output_file='../data/all_paper_data.json',
        api_key='your_api_key',
    )
    crawler.crawl()


if __name__ == '__main__':
    main()
