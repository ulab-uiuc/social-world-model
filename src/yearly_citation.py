import logging
from scholarly import scholarly
from serpapi import GoogleSearch
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def retry_on_error(func, *args, retries=3, delay=2, **kwargs):
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise

def get_scholar_id_from_citedby(data):
    citedby_url = data.get('citedby_url', '')
    if 'cites=' in citedby_url:
        parts = citedby_url.split('cites=')
        if len(parts) > 1:
            id_and_rest = parts[1]
            return id_and_rest.split('&')[0]
    return None

def get_citation_id(title):
    def search_title():
        pubs = scholarly.search_pubs(title)
        for pub in pubs:
            if title.lower() == pub['bib']['title'].lower():
                scholar_id = get_scholar_id_from_citedby(pub)
                pub_year = int(pub['bib']['pub_year'])
                return pub, scholar_id, pub_year
        return None, None, None

    return retry_on_error(search_title)

def fetch_citation_count(citation_id, year_from=None, year_to=None):
    def fetch_count():
        params = {
            "engine": "google_scholar",
            "cites": citation_id,
            "api_key": "62761aa86b00accf57227b5b644bede3059bcb4593a66fb6387501d7a7aeea9b"
        }
        if year_from:
            params["as_ylo"] = year_from
        if year_to:
            params["as_yhi"] = year_to

        search = GoogleSearch(params)
        results = search.get_dict()
        return results.get("search_information", {}).get("total_results")

    return retry_on_error(fetch_count)

def get_paper_data(title):
    try:
        pub, citation_id, pub_year = get_citation_id(title)
        if citation_id:
            total_citations = fetch_citation_count(citation_id)
            citations = {
                year: fetch_citation_count(citation_id, year_from=year, year_to=year)
                for year in range(pub_year, 2025)
            }
            return {
                'title': title,
                'citation_id': citation_id,
                'total_citations': total_citations,
                'citations': citations,
                'raw_data': pub 
            }
        else:
            logger.warning("Failed to retrieve Citation ID.")
            return None
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return None


if __name__ == "__main__":
    title = "Attention is all you need"
    paper_data = get_paper_data(title)
    if paper_data:
        print(paper_data)
    else:
        print("Failed to retrieve paper data.")
