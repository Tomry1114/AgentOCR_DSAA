import json
import logging
import requests
import uuid
import time
import threading
from typing import Tuple, Optional, Any, Dict
from urllib.parse import urlparse

from agent_system.environments.env_package.search.third_party.skyrl_gym.tools.core import tool, ToolGroup

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_TIMEOUT = 30
MAX_RETRIES = 10
INITIAL_RETRY_DELAY = 1


def _call_search_api_once(
    session: requests.Session,
    retrieval_service_url: str,
    payload: Dict[str, Any],
    timeout: int,
):
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    return session.post(
        retrieval_service_url,
        headers=headers,
        json=payload,
        timeout=timeout,
    )


def call_search_api(
    retrieval_service_url: str,
    query: str,
    topk: int = 3,
    return_scores: bool = True,
    timeout: int = DEFAULT_TIMEOUT,
    log_requests: bool = True,
    session: Optional[requests.Session] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Calls the search API with a single query.

    Args:
        retrieval_service_url: The URL of the search API.
        query: The query to search for.
        topk: The number of results to return.
        return_scores: Whether to return scores for the results.
        timeout: The timeout for the request.
        log_requests: Whether to log requests.
        session: The session to use for the request. If none is provided, a new session will be created.

    Returns:
        response: The response from the search API (json if successful, None otherwise)
        error_msg: The error message if the request failed.
    """
    request_id = str(uuid.uuid4())
    log_prefix = f"[Search Request ID: {request_id}] "

    payload = {"query": query, "topk": topk, "return_scores": return_scores}

    # Use provided session or create a new one for this request
    if session is None:
        session = requests.Session()
        should_close_session = True
    else:
        should_close_session = False

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            if log_requests:
                logger.info(
                    f"{log_prefix}Attempt {attempt + 1}/{MAX_RETRIES}: Calling search API at {retrieval_service_url}"
                )
            response = _call_search_api_once(
                session=session,
                retrieval_service_url=retrieval_service_url,
                payload=payload,
                timeout=timeout,
            )

            # Check for Gateway Timeout (504) and other server errors for retrying
            if response.status_code in [500, 502, 503, 504]:
                last_error = f"{log_prefix}API Request Error: Server Error ({response.status_code}) on attempt {attempt + 1}/{MAX_RETRIES}"
                logger.warning(last_error)
                if attempt < MAX_RETRIES - 1:
                    delay = INITIAL_RETRY_DELAY * (attempt + 1)
                    logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                    time.sleep(delay)
                continue

            # Check for other HTTP errors (e.g., 4xx)
            response.raise_for_status()

            # If successful (status code 2xx)
            if log_requests:
                logger.info(f"{log_prefix}Search API call successful on attempt {attempt + 1}")

            # Close session if we created it
            if should_close_session:
                session.close()

            return response.json(), None

        except requests.exceptions.ConnectionError as e:
            last_error = f"{log_prefix}Connection Error: {e}"
            logger.warning(last_error)
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                time.sleep(delay)
            continue
        except requests.exceptions.Timeout as e:
            last_error = f"{log_prefix}Timeout Error: {e}"
            logger.warning(last_error)
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                time.sleep(delay)
            continue
        except requests.exceptions.RequestException as e:
            last_error = f"{log_prefix}API Request Error: {e}"
            break  # Exit retry loop on other request errors
        except json.JSONDecodeError as e:
            raw_response_text = response.text if "response" in locals() else "N/A"
            last_error = f"{log_prefix}API Response JSON Decode Error: {e}, Response: {raw_response_text[:200]}"
            break  # Exit retry loop on JSON decode errors
        except Exception as e:
            last_error = f"{log_prefix}Unexpected Error: {e}"
            break  # Exit retry loop on other unexpected errors

    # If we reach here, all attempts failed
    logger.error(f"{log_prefix}API Request Failed after {MAX_RETRIES} attempts: {last_error}")

    # Close session if we created it
    if should_close_session:
        session.close()

    return None, last_error


def call_search_api_batch(
    retrieval_service_url: str,
    query_list: list[str],
    topk: int = 3,
    return_scores: bool = True,
    timeout: int = DEFAULT_TIMEOUT,
    log_requests: bool = True,
    session: Optional[requests.Session] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Calls the search API with a batch of queries.
    """
    request_id = str(uuid.uuid4())
    log_prefix = f"[Search Batch Request ID: {request_id}] "

    if session is None:
        session = requests.Session()
        should_close_session = True
    else:
        should_close_session = False

    payload = {"queries": query_list, "topk": topk, "return_scores": return_scores}
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            if log_requests:
                logger.info(
                    f"{log_prefix}Attempt {attempt + 1}/{MAX_RETRIES}: Calling batch search API at {retrieval_service_url} for {len(query_list)} queries"
                )
            response = _call_search_api_once(
                session=session,
                retrieval_service_url=retrieval_service_url,
                payload=payload,
                timeout=timeout,
            )

            if response.status_code in [500, 502, 503, 504]:
                last_error = f"{log_prefix}API Request Error: Server Error ({response.status_code}) on attempt {attempt + 1}/{MAX_RETRIES}"
                logger.warning(last_error)
                if attempt < MAX_RETRIES - 1:
                    delay = INITIAL_RETRY_DELAY * (attempt + 1)
                    logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                    time.sleep(delay)
                continue

            if response.status_code in [400, 404, 405, 422] and query_list:
                if log_requests:
                    logger.warning(
                        f"{log_prefix}Batch search payload was rejected with status {response.status_code}; "
                        "falling back to single-query compatibility mode."
                    )
                combined_results = []
                for query in query_list:
                    single_response, single_error = call_search_api(
                        retrieval_service_url=retrieval_service_url,
                        query=query,
                        topk=topk,
                        return_scores=return_scores,
                        timeout=timeout,
                        log_requests=log_requests,
                        session=session,
                    )
                    if single_error:
                        if should_close_session:
                            session.close()
                        return None, single_error
                    single_results = [] if single_response is None else single_response.get("result", [])
                    if not isinstance(single_results, list):
                        if should_close_session:
                            session.close()
                        return None, f"{log_prefix}Unexpected single-query response format: {single_response}"
                    combined_results.extend(single_results)
                if should_close_session:
                    session.close()
                return {"result": combined_results}, None

            response.raise_for_status()

            if log_requests:
                logger.info(
                    f"{log_prefix}Batch search API call successful on attempt {attempt + 1}"
                )

            if should_close_session:
                session.close()

            return response.json(), None

        except requests.exceptions.ConnectionError as e:
            last_error = f"{log_prefix}Connection Error: {e}"
            logger.warning(last_error)
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                time.sleep(delay)
            continue
        except requests.exceptions.Timeout as e:
            last_error = f"{log_prefix}Timeout Error: {e}"
            logger.warning(last_error)
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                time.sleep(delay)
            continue
        except requests.exceptions.RequestException as e:
            last_error = f"{log_prefix}API Request Error: {e}"
            break
        except json.JSONDecodeError as e:
            raw_response_text = response.text if "response" in locals() else "N/A"
            last_error = f"{log_prefix}API Response JSON Decode Error: {e}, Response: {raw_response_text[:200]}"
            break
        except Exception as e:
            last_error = f"{log_prefix}Unexpected Error: {e}"
            break

    logger.error(f"{log_prefix}Batch API Request Failed after {MAX_RETRIES} attempts: {last_error}")

    if should_close_session:
        session.close()

    return None, last_error


def _passages2string(retrieval_result):
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"].strip()
        format_reference += f"Doc {idx+1}: {content}\n"
    return format_reference


def build_search_result_text(retrieval_result: Any) -> Tuple[str, int, Optional[str]]:
    if retrieval_result:
        formatted_result = _passages2string(retrieval_result)
        total_results = len(retrieval_result) if isinstance(retrieval_result, list) else 1
        return json.dumps({"result": formatted_result}), total_results, formatted_result
    return json.dumps({"result": "No search results found."}), 0, None


class SearchToolGroup(ToolGroup):
    # Class-level session pool shared across all instances
    _session_pool = {}
    _session_lock = threading.Lock()
    _rate_limit_pool = {}
    _rate_limit_lock = threading.Lock()

    @classmethod
    def _get_shared_session(cls, base_url: str) -> requests.Session:
        """Get or create a shared session for the given base URL"""
        with cls._session_lock:
            if base_url not in cls._session_pool:
                session = requests.Session()
                # Configure connection pooling
                adapter = requests.adapters.HTTPAdapter(
                    pool_connections=512,  # Number of connection pools
                    pool_maxsize=512,  # Max connections per pool
                    max_retries=0,  # We handle retries ourselves
                    pool_block=False,  # Don't block if pool is full
                )
                session.mount("http://", adapter)
                session.mount("https://", adapter)
                cls._session_pool[base_url] = session
                logger.info(f"Created shared session pool for {base_url}")
            return cls._session_pool[base_url]

    @classmethod
    def _get_shared_rate_limiter(cls, base_url: str, max_concurrent_requests: int) -> threading.BoundedSemaphore:
        """Get or create a shared per-URL concurrency limiter."""
        normalized_limit = max(1, int(max_concurrent_requests))
        with cls._rate_limit_lock:
            limiter = cls._rate_limit_pool.get(base_url)
            if limiter is None:
                limiter = threading.BoundedSemaphore(normalized_limit)
                cls._rate_limit_pool[base_url] = limiter
                logger.info(
                    "Created shared search concurrency limiter for %s with max_concurrent_requests=%s",
                    base_url,
                    normalized_limit,
                )
            return limiter

    def __init__(
        self,
        search_url="http://127.0.0.1:8000/retrieve",
        topk=3,
        timeout=DEFAULT_TIMEOUT,
        log_requests=True,
        max_concurrent_requests=8,
    ):
        self.search_url = search_url
        self.topk = topk
        self.timeout = timeout
        self.log_requests = log_requests
        self.max_concurrent_requests = max(1, int(max_concurrent_requests))

        # Extract base URL for session sharing
        parsed_url = urlparse(self.search_url)
        self.base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

        # Get shared session for this base URL
        self.session = self._get_shared_session(self.base_url)
        self.concurrent_semaphore = self._get_shared_rate_limiter(
            self.base_url,
            self.max_concurrent_requests,
        )
        if self.log_requests:
            logger.info(
                "SearchToolGroup initialized using shared session pool for %s with max_concurrent_requests=%s",
                self.base_url,
                self.max_concurrent_requests,
            )

        super().__init__(name="SearchToolGroup")

    @tool
    def search(self, query: str) -> str:
        # NOTE(shu): add warning messages here?
        if query is None:
            return ""

        query = query.strip()

        try:
            with self.concurrent_semaphore:
                api_response, error_msg = call_search_api(
                    retrieval_service_url=self.search_url,
                    query=query,
                    topk=self.topk,
                    timeout=self.timeout,
                    log_requests=self.log_requests,
                    session=self.session,  # Pass our shared session for connection reuse
                )
        except Exception as e:
            error_msg = f"API Request Exception during batch search: {e}"
            logger.error(f"Batch search: {error_msg}")

        metadata = {
            "query": query,
            "api_request_error": error_msg,
            "api_response": None,
            "status": "unknown",
            "total_results": 0,
            "formatted_result": None,
        }

        result_text = "Search request failed or timed out after retries."

        if error_msg:
            metadata["status"] = "api_error"
            result_text = f"Search error: {error_msg}"
            logger.error(f"Batch search: API error occurred: {error_msg}")
        elif api_response:
            logger.debug(f"Batch search: API Response: {api_response}")
            metadata["api_response"] = api_response

            try:
                raw_results = api_response.get("result", [])
                if raw_results:
                    pretty_results = []
                    total_results = 0
                    for retrieval in raw_results:
                        _, retrieval_count, formatted = build_search_result_text(retrieval)
                        if formatted:
                            pretty_results.append(formatted)
                        total_results += retrieval_count

                    final_result = "\n---\n".join(pretty_results) if pretty_results else "No search results found."
                    result_text = final_result
                    metadata["status"] = "success"
                    metadata["total_results"] = total_results
                    metadata["formatted_result"] = final_result
                    if self.log_requests:
                        logger.info(f"Batch search: Successful, got {total_results} total results")
                else:
                    result_text = "No search results found."
                    metadata["status"] = "no_results"
                    metadata["total_results"] = 0
                    if self.log_requests:
                        logger.info("Batch search: No results found")
            except Exception as e:
                error_msg = f"Error processing search results: {e}"
                result_text = error_msg
                metadata["status"] = "processing_error"
                logger.error(f"Batch search: {error_msg}")
        else:
            metadata["status"] = "unknown_api_state"
            result_text = "Unknown API state (no response and no error message)."
            logger.error("Batch search: Unknown API state.")

        return result_text
