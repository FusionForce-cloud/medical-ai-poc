from tavily import TavilyClient
from dotenv import load_dotenv
import os
from logging_config import logger

load_dotenv()
client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def web_search(query: str) -> str:
    logger.info(f"Web search started for query: {query}")
    try:
        results = client.search(query, max_results=3)
        formatted = []
        for r in results["results"]:
            formatted.append(
                f"{r['title']}: {r['content']}\nSource: {r['url']}"
            )
        logger.info(f"Web search completed for query: {query}")
        return "\n\n".join(formatted)
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return "Sorry, I couldn't fetch web results at the moment."
