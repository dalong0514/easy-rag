import os

from dotenv import load_dotenv, find_dotenv

def load_env():
    _ = load_dotenv(find_dotenv())


def get_api_key():
    load_env()
    api_key = os.getenv("API_KEY")
    return api_key