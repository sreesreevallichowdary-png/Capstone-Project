# Zepto Policy Support Assistant

## Architecture
Ingestion: `docs/doc_01.txt`–`doc_08.txt` are read as one chunk each. Embedding: `get_collection()` embeds them locally with `all-MiniLM-L6-v2`. Retrieval: ChromaDB collection `zepto_policy_chunks` returns the three nearest chunks in `retrieve_and_answer`. Generation: the same LangGraph node returns the mock `Based on the retrieved context: ...` answer. `classify_intent` conditionally routes policy queries to retrieval and general questions to `direct_answer`.

`MOCK_LLM=1` is the default graded path: no LLM API calls are made. In both answer nodes the generation step branches on `MOCK_LLM`; `MOCK_LLM=0` is reserved for an optional real-LLM extension. The Pydantic response always contains `answer`, `sources`, and `confidence`.

## Run
`pip install -r requirements.txt` then `uvicorn main:app --host 0.0.0.0 --port 7860`.

## Docker
`docker build -t zepto-support-assistant .` then `docker run --rm -p 7860:7860 zepto-support-assistant`.

Example mock responses are printed in the executed notebook for a policy query and an unrelated general query.
