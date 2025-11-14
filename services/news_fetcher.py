import logging
import requests

def fetch_news(topic, api_key):
    url = f"https://newsapi.org/v2/everything?q={topic}&apiKey={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json().get("articles", [])
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching news: {e}")
        return []

def fetch_top_headlines(category, country, api_key, page_size=20):
    url = "https://newsapi.org/v2/top-headlines"
    params = {"country": country, "pageSize": page_size, "apiKey": api_key}
    if category:
        params["category"] = category
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json().get("articles", [])
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching headlines: {e}")
        return []

def fetch_everything_query(query, api_key, sort_by="publishedAt", page_size=20):
    url = "https://newsapi.org/v2/everything"
    params = {"q": query, "sortBy": sort_by, "pageSize": page_size, "apiKey": api_key}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json().get("articles", [])
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching news: {e}")
        return []