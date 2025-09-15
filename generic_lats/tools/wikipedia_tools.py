"""
Wikipedia tools for Generic LATS framework, ported from original implementation.
"""

import logging
import time
import requests
from typing import Optional, Any, List, Dict
from bs4 import BeautifulSoup
from langchain.tools import BaseTool
from pydantic import Field

logger = logging.getLogger(__name__)


def clean_str(p: str) -> str:
    """Clean string encoding issues from Wikipedia content."""
    try:
        return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")
    except UnicodeDecodeError:
        return p


class WikipediaSearchTool(BaseTool):
    """
    Tool for searching Wikipedia articles.
    
    Searches for entities on Wikipedia and returns the first paragraph if found,
    or similar entities if not found.
    """
    
    name: str = "search"
    description: str = """Search for an entity on Wikipedia and return the first paragraph if it exists. 
    If not found, returns similar entities to search. Usage: Search[entity_name]"""
    
    # Declare fields with proper Pydantic syntax
    search_time: float = Field(default=0.0, init=False)
    num_searches: int = Field(default=0, init=False)
    cache: Dict[str, str] = Field(default_factory=dict, init=False)
    
    def _run(self, entity: str, run_manager: Optional[Any] = None) -> str:
        """
        Execute Wikipedia search.
        
        Args:
            entity: Entity name to search for
            run_manager: Optional run manager
            
        Returns:
            Search result as string
        """
        if not entity:
            return "Error: Empty search entity"
        
        # Check cache first
        if entity in self.cache:
            logger.debug(f"Cache hit for entity: {entity}")
            return self.cache[entity]
        
        try:
            result = self._search_wikipedia(entity)
            # Cache the result
            self.cache[entity] = result
            return result
        except Exception as e:
            logger.error(f"Error searching Wikipedia for '{entity}': {e}")
            return f"Error searching for {entity}: {str(e)}"
    
    async def _arun(self, entity: str, run_manager: Optional[Any] = None) -> str:
        """Async version of _run."""
        return self._run(entity, run_manager)
    
    def _search_wikipedia(self, entity: str) -> str:
        """
        Perform the actual Wikipedia search.
        
        Args:
            entity: Entity to search for
            
        Returns:
            Search result string
        """
        entity_encoded = entity.replace(" ", "+")
        search_url = f"https://en.wikipedia.org/w/index.php?search={entity_encoded}"
        
        start_time = time.time()
        
        try:
            response = requests.get(search_url, timeout=10)
            response.raise_for_status()
            
            self.search_time += time.time() - start_time
            self.num_searches += 1
            
            soup = BeautifulSoup(response.text, features="html.parser")
            
            # Check if we got search results (no direct match)
            result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
            
            if result_divs:
                # No direct match found, return similar results
                similar_titles = [clean_str(div.get_text().strip()) for div in result_divs]
                return f"Could not find {entity}. Similar: {similar_titles[:5]}."
            
            else:
                # Direct match found, extract page content
                paragraphs = soup.find_all("p") + soup.find_all("ul")
                page_content = [p.get_text().strip() for p in paragraphs]
                
                # Check if this is a disambiguation page
                if any("may refer to:" in p for p in page_content):
                    # Retry with bracketed entity name
                    return self._search_wikipedia(f"[{entity}]")
                
                # Extract main content
                page_text = ""
                for p in page_content:
                    if len(p.split(" ")) > 2:  # Filter out very short content
                        page_text += clean_str(p)
                        if not p.endswith("\n"):
                            page_text += "\n"
                
                if page_text.strip():
                    # Return first 5 sentences as observation
                    return self._get_page_obs(page_text)
                else:
                    return f"Found {entity} but no content could be extracted."
        
        except requests.exceptions.Timeout:
            return f"Search timeout for {entity}"
        except requests.exceptions.RequestException as e:
            return f"Search error for {entity}: {str(e)}"
    
    @staticmethod
    def _get_page_obs(page: str) -> str:
        """
        Extract observation from page content (first few sentences).
        
        Args:
            page: Full page content
            
        Returns:
            Truncated page content for observation
        """
        # Split into paragraphs
        paragraphs = page.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Split into sentences
        sentences = []
        for p in paragraphs:
            sentences += p.split('. ')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        
        # Return first 5 sentences
        return ' '.join(sentences[:5])
    
    def get_search_stats(self) -> dict:
        """Get search statistics."""
        speed = self.search_time / self.num_searches if self.num_searches > 0 else 0
        return {
            "call_speed": speed,
            "call_time": self.search_time,
            "num_calls": self.num_searches,
            "cache_size": len(self.cache)
        }
    
    def clear_cache(self) -> None:
        """Clear the search cache."""
        self.cache.clear()


class WikipediaLookupTool(BaseTool):
    """
    Tool for looking up keywords within the current Wikipedia page.
    
    Searches for sentences containing a specific keyword within the most recently
    retrieved Wikipedia page content.
    """
    
    name: str = "lookup"
    description: str = """Look up a keyword in the current Wikipedia page and return the next sentence 
    containing that keyword. Usage: Lookup[keyword]"""
    
    # Declare fields with proper Pydantic syntax
    current_page: Optional[str] = Field(default=None, init=False)
    lookup_keyword: Optional[str] = Field(default=None, init=False)
    lookup_list: Optional[List[str]] = Field(default=None, init=False)
    lookup_cnt: Optional[int] = Field(default=None, init=False)
    
    def _run(self, keyword: str, run_manager: Optional[Any] = None) -> str:
        """
        Execute keyword lookup.
        
        Args:
            keyword: Keyword to look up
            run_manager: Optional run manager
            
        Returns:
            Lookup result as string
        """
        if not keyword:
            return "Error: Empty lookup keyword"
        
        if not self.current_page:
            return "Error: No Wikipedia page loaded. Search for an entity first."
        
        try:
            return self._perform_lookup(keyword)
        except Exception as e:
            logger.error(f"Error performing lookup for '{keyword}': {e}")
            return f"Error looking up {keyword}: {str(e)}"
    
    async def _arun(self, keyword: str, run_manager: Optional[Any] = None) -> str:
        """Async version of _run."""
        return self._run(keyword, run_manager)
    
    def _perform_lookup(self, keyword: str) -> str:
        """
        Perform the actual keyword lookup.
        
        Args:
            keyword: Keyword to search for
            
        Returns:
            Lookup result string
        """
        # Reset lookup if different keyword
        if self.lookup_keyword != keyword:
            self.lookup_keyword = keyword
            self.lookup_list = self._construct_lookup_list(keyword)
            self.lookup_cnt = 0
        
        # Check if we have more results
        if self.lookup_cnt >= len(self.lookup_list):
            return "No more results."
        
        # Return next result
        result = self.lookup_list[self.lookup_cnt]
        result_text = f"(Result {self.lookup_cnt + 1} / {len(self.lookup_list)}) {result}"
        self.lookup_cnt += 1
        
        return result_text
    
    def _construct_lookup_list(self, keyword: str) -> List[str]:
        """
        Construct list of sentences containing the keyword.
        
        Args:
            keyword: Keyword to search for
            
        Returns:
            List of sentences containing the keyword
        """
        if not self.current_page:
            return []
        
        # Split into paragraphs
        paragraphs = self.current_page.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Split into sentences
        sentences = []
        for p in paragraphs:
            sentences += p.split('. ')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        
        # Filter sentences containing keyword (case-insensitive)
        matching_sentences = [
            s for s in sentences 
            if keyword.lower() in s.lower()
        ]
        
        return matching_sentences
    
    def set_current_page(self, page_content: str) -> None:
        """
        Set the current Wikipedia page content for lookup.
        
        Args:
            page_content: Full Wikipedia page content
        """
        self.current_page = page_content
        # Reset lookup state
        self.lookup_keyword = None
        self.lookup_list = None
        self.lookup_cnt = None
        
        logger.debug("Set new Wikipedia page for lookup")
    
    def get_lookup_stats(self) -> dict:
        """Get lookup statistics."""
        return {
            "current_keyword": self.lookup_keyword,
            "total_results": len(self.lookup_list) if self.lookup_list else 0,
            "current_position": self.lookup_cnt if self.lookup_cnt is not None else 0,
            "has_page": self.current_page is not None
        }


class WikipediaEnvironment:
    """
    Wrapper class that coordinates Wikipedia search and lookup tools.
    
    This class manages the state between search and lookup operations,
    ensuring that lookup operates on the most recently searched page.
    """
    
    def __init__(self):
        self.search_tool = WikipediaSearchTool()
        self.lookup_tool = WikipediaLookupTool()
        self.current_page_content = None
        self.steps = 0
        
    def search(self, entity: str) -> str:
        """
        Search for a Wikipedia entity and prepare it for lookup.
        
        Args:
            entity: Entity to search for
            
        Returns:
            Search result
        """
        result = self.search_tool._run(entity)
        
        # If search was successful (not a "Could not find" message), 
        # extract full page content for lookup
        if not result.startswith("Could not find") and not result.startswith("Error"):
            # For now, use the search result as page content
            # In a full implementation, we'd fetch the complete page
            self.current_page_content = result
            self.lookup_tool.set_current_page(result)
        
        self.steps += 1
        return result
    
    def lookup(self, keyword: str) -> str:
        """
        Look up a keyword in the current page.
        
        Args:
            keyword: Keyword to look up
            
        Returns:
            Lookup result
        """
        result = self.lookup_tool._run(keyword)
        self.steps += 1
        return result
    
    def get_stats(self) -> dict:
        """Get combined statistics from both tools."""
        return {
            "steps": self.steps,
            "search_stats": self.search_tool.get_search_stats(),
            "lookup_stats": self.lookup_tool.get_lookup_stats()
        }
    
    def reset(self) -> None:
        """Reset the environment state."""
        self.current_page_content = None
        self.steps = 0
        self.lookup_tool.set_current_page("")