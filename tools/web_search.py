
from tavily import TavilyClient
import config


def search_web(query: str) -> list[dict]:
    """
    Searches the web using Tavily and returns structured results.
    Tavily is purpose-built for LLM agents — returns clean, relevant content.

    Args:
        query: Search query string

    Returns:
        List of dicts with 'content' and 'url' keys
    """
    if not config.TAVILY_API_KEY:
        print("[WebSearch] No Tavily API key found. Skipping web search.")
        return []

    try:
        client = TavilyClient(api_key=config.TAVILY_API_KEY)
        response = client.search(
            query=query,
            max_results=config.MAX_SEARCH_RESULTS,
            search_depth="advanced"   # deeper, higher quality results
        )

        results = []
        for item in response.get("results", []):
            content = item.get("content", "").strip()
            url = item.get("url", "")
            if content:
                results.append({"content": content, "url": url})

        print(f"[WebSearch] Found {len(results)} results for: '{query}'")
        return results

    except Exception as e:
        print(f"[WebSearch] Error during search: {e}")
        return []
