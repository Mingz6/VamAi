# Prompt: create a flask app with templates/index.html that gives us a search bar, call it deep research agent and when the user searches for something on the search bar display 3 results use duckduckgo_search for this
# pip install flask duckduckgo_searc

from flask import Flask, render_template, request
from duckduckgo_search import DDGS
import time
import random

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    search_results = []
    query = ""
    
    if request.method == "POST":
        query = request.form.get("query", "")
        
        if query:
            # Use DuckDuckGo search to get results
            retries = 5  # Increased retries
            for attempt in range(retries):
                try:
                    with DDGS() as ddgs:
                        search_results = list(ddgs.text(query, max_results=3))
                    break  # Success - exit the retry loop
                except Exception as e:
                    if "Ratelimit" in str(e) and attempt < retries - 1:
                        # Exponential backoff with jitter
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        print(f"Rate limited. Waiting {wait_time:.2f} seconds before retry {attempt+1}/{retries}")
                        time.sleep(wait_time)
                    else:
                        error_message = f"Search error: {str(e)}"
                        print(error_message)
                        search_results = [
                            {"title": "Error", "body": error_message, "href": "#"}
                        ]
                        break
    
    return render_template("index.html", results=search_results, query=query)

if __name__ == "__main__":
    app.run(debug=True, port=5008, host="0.0.0.0", use_reloader=False)