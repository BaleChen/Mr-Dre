"""
Citation-based evaluation module.
Evaluates reports by checking if claims are supported by citations.
"""

import json
import logging
import re
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from dr_eval.eval_prompts import (
    get_citation_claim_extraction_prompts,
    get_citation_supported_judge_prompts,
    get_content_summarization_prompt,
)
from dr_eval.crawler.jina import fetch_webpage_content_jina
from utils import model_name

logger = logging.getLogger(__name__)

VALID_JUDGE_RESULTS = {"supported", "insufficient", "contradictory", "no_url"}


def _split_report_into_sections(report: str) -> List[str]:
    """
    Split the report into sections using double newlines. This excludes the references/sources section.
    Adaptively splits into 4 parts, with each part containing at least 2 \n\n-separated sections.
    
    From ResearcherBench.
    """
    reference_titles_hash = ['References', 'references', 'Key Citations']
    reference_titles_nohash = ['**References**', '参考资料', '**Sources:**', '**Works cited**', '**引用的著作**', '<div>⁂</div>', '<div style="text-align: center">⁂</div>']

    hash_pattern = '|'.join(re.escape(title) for title in reference_titles_hash)
    nohash_pattern = '|'.join(re.escape(title) for title in reference_titles_nohash)
    reference_pattern = fr'(?i)(?:^#+\s*(?:{hash_pattern})|^#*\s*(?:{nohash_pattern}))'
    
    # Remove references section if it exists
    report = re.split(reference_pattern, report, flags=re.MULTILINE)[0]
    
    sections = re.split(r'\n\n', report)
    sections = [s for s in sections if s.strip() != ""]

    # Adaptively choose step to split into 4 parts, with minimum 2 sections per part
    step = max(2, -(-len(sections) // 4))  # ceiling division, min 2
    
    processed_sections = []
    for i in range(0, len(sections), step):
        processed_sections.append("\n\n".join(sections[i:i+step]))
    
    if not processed_sections:
        return [report.strip()]
        
    return processed_sections


def extract_claims(
    report: str,
    extraction_model: Any,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Extract claims from a report.
    
    Args:
        report: The generated report text
        extraction_model: Callable LLM used for claim extraction
        **kwargs: Additional arguments
    
    Returns:
        List of claim dictionaries with structure:
        [
            {
                "claim": str,
                "url": List[str]  # list of citation URLs (empty list if no citation)
            }
        ]
    """
    # Get all sections ready
    sections = _split_report_into_sections(report)
    if not sections:
        sections = [report]
    
    # Get all prompts
    prompts = get_citation_claim_extraction_prompts(report, sections)
    
    # Multi-threaded LLM calls
    call_llm = lambda prompt: json.loads(extraction_model(prompt))
    results = []
    with ThreadPoolExecutor(max_workers=min(len(prompts), 20)) as executor:
        future_to_prompt = {executor.submit(call_llm, prompt): idx
                           for idx, prompt in enumerate(prompts)}

        results = [None] * len(prompts)
        with tqdm(total=len(prompts), desc="Extracting claims", disable=len(prompts) < 4, leave=False) as pbar:
            for future in as_completed(future_to_prompt):
                idx = future_to_prompt[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = []
                    logger.error(f"Claim extraction {idx} failed: {e}")
                pbar.update(1)

    # Flatten and normalize url field to always be a list
    claims = sum(results, [])
    for claim in claims:
        url = claim.get("url", [])
        if isinstance(url, str):
            claim["url"] = [url] if url else []
    return claims


def judge_claims(
    claims: List[Dict[str, Any]],
    judge_model: Any,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Judge whether claims are supported by their combined URL content.
    
    Args:
        claims: List of claim dictionaries, where each dict has:
            - "id": int
            - "claim": str
            - "url": List[str]
            - "combined_content": str (concatenated content/summaries from all URLs)
        judge_model: Model to use for judgment
        **kwargs: Additional arguments
    
    Returns:
        Same list with "result" field added to each claim
    """
    if not claims:
        return claims
    
    # Filter claims that have content to judge
    claims_to_judge = [(i, c) for i, c in enumerate(claims) if c.get("combined_content")]
    
    if not claims_to_judge:
        for claim in claims:
            claim["result"] = "no_url" if not claim.get("url") else "insufficient"
        return claims
    
    # Build prompts - one per claim
    prompts = get_citation_supported_judge_prompts([c for _, c in claims_to_judge])
    
    call_llm = lambda prompt: json.loads(judge_model(prompt))
    responses = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=min(len(prompts), 10)) as executor:
        future_to_idx = {executor.submit(call_llm, p): i for i, p in enumerate(prompts)}
        with tqdm(total=len(prompts), desc="Judging claims", disable=len(prompts) < 5, leave=False) as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    responses[idx] = future.result()
                except Exception as e:
                    logger.error(f"Claim judgment {idx} failed: {e}")
                pbar.update(1)
    
    # Parse results
    for (orig_idx, _), response in zip(claims_to_judge, responses):
        if response and isinstance(response, dict):
            result = response.get("result", "error").lower()
            claims[orig_idx]["result"] = result if result in VALID_JUDGE_RESULTS else "error"
        else:
            claims[orig_idx]["result"] = "error"
    
    # Handle claims without URLs
    for claim in claims:
        if "result" not in claim:
            claim["result"] = "no_url" if not claim.get("url") else "insufficient"
    
    return claims


def crawl_urls(urls: List[str]) -> Dict[str, str]:
    """
    Crawl a list of URLs and return a mapping of URL -> content.
    Uses caching so duplicate URLs are only fetched once.
    
    Args:
        urls: List of URLs to crawl
    
    Returns:
        Dict mapping URL to its content
    """
    if not urls:
        return {}
    
    def crawl_url(url: str) -> Tuple[str, str]:
        try:
            result = fetch_webpage_content_jina(url)
            return (url, result.get("content", ""))
        except Exception as e:
            logger.warning(f"Crawl failed: {url[:50]}...")
            return (url, "")
    
    url_to_content = {}
    with ThreadPoolExecutor(max_workers=min(len(urls), 20)) as executor:
        futures = {executor.submit(crawl_url, url): url for url in urls}
        with tqdm(total=len(urls), desc="Crawling URLs", disable=len(urls) < 3, leave=False) as pbar:
            for future in as_completed(futures):
                url, content = future.result()
                url_to_content[url] = content
                pbar.update(1)
    
    return url_to_content


def summarize_url_content(
    url_to_content: Dict[str, str],
    url_to_claims: Dict[str, List[Dict[str, Any]]],
    summarization_model: Any,
) -> Dict[str, str]:
    """
    Summarize URL content to extract parts relevant to the claims citing each URL.
    
    Args:
        url_to_content: Mapping of URL to raw content
        url_to_claims: Mapping of URL to list of claims that cite it
        summarization_model: Model to use for summarization
    
    Returns:
        Mapping of URL to summarized content
    """
    if not url_to_content or summarization_model is None:
        return url_to_content
    
    # Filter URLs that have content to summarize
    urls_with_content = [url for url, content in url_to_content.items() if content]
    
    if not urls_with_content:
        return url_to_content
    
    def summarize(url: str) -> Tuple[str, str]:
        try:
            content = url_to_content[url]
            claims = url_to_claims.get(url, [])
            prompt = get_content_summarization_prompt(content, claims)
            summary = summarization_model(prompt)
            return (url, summary)
        except Exception as e:
            logger.error(f"Summarization failed: {url[:50]}...")
            return (url, url_to_content[url])  # Return original on error
    
    url_to_summary = dict(url_to_content)  # Start with original content
    with ThreadPoolExecutor(max_workers=min(len(urls_with_content), 10)) as executor:
        futures = {executor.submit(summarize, url): url for url in urls_with_content}
        with tqdm(total=len(futures), desc="Summarizing", disable=len(urls_with_content) < 3, leave=False) as pbar:
            for future in as_completed(futures):
                url, summary = future.result()
                url_to_summary[url] = summary
                pbar.update(1)
    
    return url_to_summary


def evaluate_citations(
    report_obj: Dict[str, Any],
    claim_extraction_model: Any,
    supported_judge_model: Any,
    webpage_summarization_model: Any = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Evaluate a report's citations for faithfulness and groundedness.
    
    Args:
        report_obj: Generated report object
        claim_extraction_model: Model for claim extraction
        supported_judge_model: Model for judging claim support
        webpage_summarization_model: Optional model for summarizing webpage content
        **kwargs: Additional arguments
    
    Returns:
        Dict with structure:
        {
            "id": int,
            "claim_extraction_model": str,
            "supported_judge_model": str,
            "faithfulness_score": float,
            "groundedness_score": float,
            "total_claims": int,
            "supported_claims": int,
            "claims_details": List[Dict]
        }
    """
    report = report_obj.get("full_response", {}).get("report", "") or ""

    # Extract claims (each claim has "url" as a list)
    extracted_claims = extract_claims(
        report=report,
        extraction_model=claim_extraction_model,
        **kwargs,
    )
    # Collect all unique URLs and build URL -> claims mapping for summarization
    all_urls = set()
    url_to_claims = {}  # For summarization: URL -> list of claims citing it
    for claim in extracted_claims:
        for url in claim.get("url", []):
            all_urls.add(url)
            if url not in url_to_claims:
                url_to_claims[url] = []
            url_to_claims[url].append({"id": len(url_to_claims[url]), "claim": claim["claim"]})

    # Crawl all URLs (caching handles duplicates)
    logger.debug(f"Crawling {len(all_urls)} URLs")
    url_to_content = crawl_urls(list(all_urls))

    # Optionally summarize each URL's content
    if webpage_summarization_model is not None:
        logger.debug(f"Summarizing {len(all_urls)} URLs")
        url_to_content = summarize_url_content(url_to_content, url_to_claims, webpage_summarization_model)

    # Build claims with combined content for judging
    claims_for_judging = []
    for i, claim in enumerate(extracted_claims):
        urls = claim.get("url", [])
        # Concatenate content/summaries from all URLs for this claim
        content_parts = [url_to_content.get(url, "").strip() for url in urls]
        combined_content = "\n\n---\n\n".join([c for c in content_parts if c])
        
        claims_for_judging.append({
            "id": i,
            "claim": claim["claim"].strip(),
            "url": urls,
            "combined_content": combined_content,
        })

    # Judge claims
    claims_with_judgments = judge_claims(
        claims=claims_for_judging,
        judge_model=supported_judge_model,
        **kwargs,
    )

    # Build claims_details for output (save summaries instead of raw url_content)
    claims_details = []
    for claim in claims_with_judgments:
        urls = claim.get("url", [])
        url_summaries = [url_to_content.get(url, "") for url in urls]
        claims_details.append({
            "claim": claim["claim"],
            "url": urls,
            "url_summaries": url_summaries,
            "result": claim.get("result", "no_url"),
        })

    # exclude claims with result "error" for metrics calculation since it comes from judgement model errors
    claims_without_errors = [item for item in claims_details if item.get("result") != "error"]

    total_claims = len(claims_without_errors)
    supported_claims = sum(1 for item in claims_without_errors if item.get("result") == "supported")
    claims_with_citations = [item for item in claims_without_errors if item.get("url")]

    faithfulness_score = (supported_claims / len(claims_with_citations)) if claims_with_citations else 0.0
    groundedness_score = (supported_claims / total_claims) if total_claims > 0 else 0.0
    return {
        "id": report_obj.get("id"),
        "claim_extraction_model": model_name(claim_extraction_model),
        "supported_judge_model": model_name(supported_judge_model),
        "faithfulness_score": round(faithfulness_score, 4),
        "groundedness_score": round(groundedness_score, 4),
        "total_claims": total_claims,
        "supported_claims": supported_claims,
        "claims_details": claims_details,
    }


def aggregate_citation_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate citation metrics across multiple samples.
    
    Args:
        metrics: List of sample-level citation metric objects
    
    Returns:
        Dict with aggregated metrics
    """
    if not metrics:
        return {}
    
    total_samples = len(metrics)
    avg_faithfulness = sum(metric.get("faithfulness_score", 0.0) for metric in metrics) / total_samples
    avg_groundedness = sum(metric.get("groundedness_score", 0.0) for metric in metrics) / total_samples
    total_claims = sum(metric.get("total_claims", 0) for metric in metrics)
    supported_claims = sum(metric.get("supported_claims", 0) for metric in metrics)

    return {
        "avg_faithfulness_score": round(avg_faithfulness, 4),
        "avg_groundedness_score": round(avg_groundedness, 4),
        "total_samples": total_samples,
        "total_claims": total_claims,
        "total_supported_claims": supported_claims,
    }

