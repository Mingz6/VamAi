import os
import requests
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import time


def search_duckduckgo(query, num_results=3):
    """
    Search DuckDuckGo for the given query and return the specified number of results.
    """
    ddgs = DDGS()
    results = list(ddgs.text(query, max_results=num_results))
    return results


def get_url_content(url):
    """
    Fetch the content of a URL and extract the main text using BeautifulSoup.
    Ignores paragraphs with fewer than 10 words.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()

        # Get text
        text = soup.get_text(separator="\n")

        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))

        # Filter out paragraphs with fewer than 10 words
        filtered_chunks = []
        for chunk in chunks:
            if chunk and len(chunk.split()) >= 10:
                filtered_chunks.append(chunk)

        text = "\n".join(filtered_chunks)

        return text
    except Exception as e:
        return f"Error fetching content: {str(e)}"


def save_content(content, url, results_dir):
    """
    Save the content to a file in the results directory.
    """
    # Create a filename based on the URL
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    path = parsed_url.path.replace("/", "_")
    if path == "" or path == "_":
        path = "_index"
    filename = f"{domain}{path}.txt"

    # Ensure the filename is valid
    filename = "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)

    filepath = os.path.join(results_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"Source URL: {url}\n\n")
        f.write(content)

    return filepath, filename


def save_summary(content, filename, results_dir):
    """
    Save the summarized content to a file in the results directory.
    """
    summary_filename = f"summarized_{filename}"
    summary_filepath = os.path.join(results_dir, summary_filename)

    with open(summary_filepath, "w", encoding="utf-8") as f:
        f.write(content)

    return summary_filepath


def main():
    # Delete and recreate results directory
    results_dir = "results"
    if os.path.exists(results_dir):
        import shutil

        shutil.rmtree(results_dir)
    os.makedirs(results_dir)

    # Import SummarizerAgent
    from agents import SummarizerAgent, InsightAgent

    summarizer = SummarizerAgent()

    # Get user query
    query = "State of tariffs"

    # Search DuckDuckGo
    print(f"Searching DuckDuckGo for: {query}")
    search_results = search_duckduckgo(query)

    if not search_results:
        print("No results found.")
        return

    # Process each result
    saved_files = []
    summarized_files = []
    for i, result in enumerate(search_results):
        url = result.get("href")
        if not url:
            continue

        print(f"\nProcessing result {i+1}/{len(search_results)}: {url}")

        # Get content
        content = get_url_content(url)

        # Save content
        filepath, filename = save_content(content, url, results_dir)
        saved_files.append(filepath)

        print(f"Content saved to: {filepath}")

        # Prepare content for summarization (limit to 500 words)
        words = content.split()
        if len(words) > 500:
            content_for_summary = " ".join(words[:500])
        else:
            content_for_summary = content

        # Generate summary
        print(f"Generating summary...")
        summary = summarizer.process(content_for_summary)

        # Save summary
        summary_filepath = save_summary(summary, filename, results_dir)
        summarized_files.append(summary_filepath)

        print(f"Summary saved to: {summary_filepath}")

        # Add a small delay to be nice to servers
        if i < len(search_results) - 1:
            time.sleep(1)

    print(
        f"\nSearch complete. {len(saved_files)} results saved to the '{results_dir}' directory."
    )
    print(f"Summaries created for {len(summarized_files)} files.")

    # Generate insights from all summaries
    print("\nGenerating insights from all summaries...")
    insight_agent = InsightAgent()
    insights = insight_agent.process(summarized_files)

    # Save insights to file
    insights_filepath = os.path.join(results_dir, "insights.txt")
    with open(insights_filepath, "w", encoding="utf-8") as f:
        f.write(insights)

    print(f"Insights saved to: {insights_filepath}")


if __name__ == "__main__":
    main()
