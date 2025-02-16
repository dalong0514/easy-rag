import os

from dotenv import load_dotenv, find_dotenv

def load_env():
    _ = load_dotenv(find_dotenv())


def get_api_key():
    load_env()
    api_key = os.getenv("API_KEY")
    return api_key

def get_api_key_openai():
    load_env()
    api_key = os.getenv("OPENAI_API_KEY")
    return api_key

def get_api_key_google():
    load_env()
    api_key = os.getenv("GOOGLE_API_KEY")
    return api_key

def get_api_key_deepseek():
    load_env()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    return api_key

def get_api_key_fireworks():
    load_env()
    api_key = os.getenv("FIREWORKs_API_KEY")
    return api_key

def get_api_key_weaviate():
    load_env()
    api_key = os.getenv("WCD_API_KEY")
    return api_key


def get_wcd_url_weaviate():
    load_env()
    wcd_url = os.getenv("WCD_URL")
    return wcd_url