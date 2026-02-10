# Koios
---

Based on this [Medium article](https://medium.com/@sahin.samia/how-to-build-a-interactive-personal-ai-research-agent-with-llama-3-2-b2a390eed63e)[1], this project seeks to create an AI research agent that takes the research question as input and either generates an answer based on its knowledge or queries the DuckDuckGo web API for more context. If it cannot find any additional information, it will return a message indicating that it does not have enough information to answer the question.

The major difference between this and the article demonstration is that this project is structured in a object-oriented manner, and will include other methods of information querying using Wikipedia and the Google search API.

## TODO:
1. Add UI interface to research agent.
2. Add additional methods of information querying when the first web query fails to get any context.
3. If any links were provided in web search, have output provide Markdown links to open.

## Getting Started:
It is assumed that you are running local model using the developer server on LM Studio.

1. Create virtual environment `python3 -m venv venv`
2. Activate virtual environment `source venv/bin/activate`
3. Install dependencies `pip install -r requirements.txt`
4. Run the program either by:
    - Running the webapp `python3 -m src app"`
    - Directly query the model `python3 -m src "<research question to ask>"`

## Source
[1] https://medium.com/@sahin.samia/how-to-build-a-interactive-personal-ai-research-agent-with-llama-3-2-b2a390eed63e