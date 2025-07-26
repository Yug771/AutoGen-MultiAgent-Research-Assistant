"""
Web search functionality for the Research Agent
"""
from typing import List, Dict, Any
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
import logging

# Try to import newspaper, but don't fail if there are issues
try:
    from newspaper import Article
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False
    logging.warning("Newspaper3k not available, using fallback extraction only")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebSearcher:
    """Web search functionality using DuckDuckGo"""
    
    def __init__(self, max_results: int = 5):
        self.max_results = max_results
        self.ddgs = DDGS()
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search the web using DuckDuckGo
        
        Args:
            query: Search query string
            
        Returns:
            List of search results with title, link, and snippet
        """
        try:
            results = []
            search_results = self.ddgs.text(query, max_results=self.max_results)
            
            for result in search_results:
                results.append({
                    "title": result.get("title", ""),
                    "link": result.get("link", ""),
                    "snippet": result.get("body", ""),
                })
            
            logger.info(f"Found {len(results)} search results for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Error during web search: {str(e)}")
            return []
    
    def extract_article_content(self, url: str) -> Dict[str, Any]:
        """
        Extract content from a webpage using newspaper3k
        
        Args:
            url: URL of the webpage
            
        Returns:
            Dictionary with article content and metadata
        """
        if NEWSPAPER_AVAILABLE:
            try:
                article = Article(url)
                article.download()
                article.parse()
                article.nlp()
                
                return {
                    "title": article.title,
                    "authors": article.authors,
                    "publish_date": str(article.publish_date) if article.publish_date else None,
                    "text": article.text,
                    "summary": article.summary,
                    "keywords": article.keywords,
                }
                
            except Exception as e:
                logger.error(f"Error extracting article content from {url}: {str(e)}")
                # Fallback to basic BeautifulSoup extraction
                return self._fallback_extract(url)
        else:
            # Use fallback extraction if newspaper is not available
            return self._fallback_extract(url)
    
    def _fallback_extract(self, url: str) -> Dict[str, Any]:
        """
        Fallback content extraction using BeautifulSoup
        
        Args:
            url: URL of the webpage
            
        Returns:
            Dictionary with basic content
        """
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Get text
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return {
                "title": soup.title.string if soup.title else "",
                "text": text[:5000],  # Limit text length
                "summary": text[:500],  # Simple summary
            }
            
        except Exception as e:
            logger.error(f"Fallback extraction failed for {url}: {str(e)}")
            return {"title": "", "text": "", "summary": ""} 