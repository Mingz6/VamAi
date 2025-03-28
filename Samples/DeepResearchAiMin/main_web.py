from flask import Flask, render_template, request, jsonify
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from agents import SummarizerAgent, InsightAgent, RecommenderAgent

app = Flask(__name__)
summarizer = SummarizerAgent()
insight_agent = InsightAgent()
recommender_agent = RecommenderAgent()


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


@app.route("/", methods=["GET", "POST"])
def index():
    search_results = []
    query = ""
    insights = None

    if request.method == "POST":
        query = request.form.get("query", "")
        if query:
            # Use DuckDuckGo search to get results
            try:
                with DDGS() as ddgs:
                    search_results = list(ddgs.text(query, max_results=3))
            except Exception as e:
                search_results = [
                    {"title": "Error", "body": f"Search error: {str(e)}", "href": "#"}
                ]

    return render_template(
        "index.html", results=search_results, query=query, insights=insights
    )


@app.route("/summarize", methods=["POST"])
def summarize():
    url = request.json.get("url")
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        # Get content from URL
        content = get_url_content(url)

        # Prepare content for summarization (limit to 500 words)
        words = content.split()
        if len(words) > 500:
            content_for_summary = " ".join(words[:500])
        else:
            content_for_summary = content

        # Generate summary
        summary = summarizer.process(content_for_summary)

        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": f"Error generating summary: {str(e)}"}), 500


@app.route("/summarize_all", methods=["POST"])
def summarize_all():
    urls = request.json.get("urls", [])
    if not urls or len(urls) == 0:
        return jsonify({"error": "No URLs provided"}), 400

    try:
        summaries = {}
        for url in urls:
            # Get content from URL
            content = get_url_content(url)

            # Prepare content for summarization (limit to 500 words)
            words = content.split()
            if len(words) > 500:
                content_for_summary = " ".join(words[:500])
            else:
                content_for_summary = content

            # Generate summary
            summary = summarizer.process(content_for_summary)
            summaries[url] = summary

        # Generate insights from all summaries
        insights = insight_agent.process_text(list(summaries.values()))

        return jsonify({"summaries": summaries, "insights": insights})
    except Exception as e:
        return (
            jsonify({"error": f"Error generating summaries and insights: {str(e)}"}),
            500,
        )


@app.route("/generate_recommendations", methods=["POST"])
def generate_recommendations():
    data = request.json
    insights = data.get("insights", "")
    summaries = data.get("summaries", [])
    user_goal = data.get("goal", "")
    persona = data.get("persona", "")

    if not insights or not summaries or not user_goal:
        return jsonify({"error": "Missing required data"}), 400

    try:
        recommendations = recommender_agent.process(
            insights, summaries, user_goal, persona
        )
        next_query = recommender_agent.suggest_next_query(
            insights, summaries, user_goal, persona
        )
        return jsonify({"recommendations": recommendations, "next_query": next_query})
    except Exception as e:
        return jsonify({"error": f"Error generating recommendations: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5008, host="0.0.0.0", use_reloader=False)
