from __future__ import annotations
import os
from pathlib import Path
from typing import Literal, TypedDict
import chromadb
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from langgraph.graph import END, StateGraph
ROOT=Path(__file__).parent; DOCS_DIR=ROOT/'docs'; CHROMA_DIR=ROOT/'chroma_db'; COLLECTION='zepto_policy_chunks'
MOCK_LLM=os.getenv('MOCK_LLM','1') != '0'
KEYWORDS=('delivery','return','refund','membership','tracking','cancel','gift card','support hours')
class AskRequest(BaseModel): query:str=Field(min_length=1,max_length=1000)
class AskResponse(BaseModel): answer:str; sources:list[str]; confidence:float=Field(ge=0,le=1)
class State(TypedDict,total=False): query:str; intent:Literal['policy_question','general_question']; response:AskResponse
embedder=None; coll=None
def get_collection():
 global embedder,coll
 if coll is not None: return coll
 embedder=SentenceTransformer('all-MiniLM-L6-v2')
 client=chromadb.PersistentClient(path=str(CHROMA_DIR))
 try: client.delete_collection(COLLECTION)
 except Exception: pass
 coll=client.create_collection(COLLECTION,metadata={'hnsw:space':'cosine'})
 paths=sorted(DOCS_DIR.glob('doc_*.txt'))
 if len(paths)!=8: raise RuntimeError('Expected 8 corpus files')
 texts=[p.read_text(encoding='utf-8') for p in paths]; ids=[p.stem for p in paths]
 coll.add(ids=ids,documents=texts,metadatas=[{'document_id':i} for i in ids],embeddings=embedder.encode(texts,normalize_embeddings=True).tolist())
 return coll
def classify_intent(state):
 q=state['query'].lower(); return {'intent':'policy_question' if any(k in q for k in KEYWORDS) else 'general_question'}
def retrieve_and_answer(state):
 c=get_collection(); r=c.query(query_embeddings=embedder.encode([state['query']],normalize_embeddings=True).tolist(),n_results=3,include=['documents','metadatas'])
 ids=[m['document_id'] for m in r['metadatas'][0]]; top=r['documents'][0][0]
 if MOCK_LLM: response=AskResponse(answer=f'Based on the retrieved context: {top[:200]}',sources=ids,confidence=1.0)
 else: raise RuntimeError('Optional MOCK_LLM=0 real LLM extension is not configured.')
 return {'response':response}
def direct_answer(state):
 if MOCK_LLM: response=AskResponse(answer='I can only answer questions about Zepto policies right now.',sources=[],confidence=1.0)
 else: raise RuntimeError('Optional MOCK_LLM=0 real LLM extension is not configured.')
 return {'response':response}
def route(state): return 'retrieve_and_answer' if state['intent']=='policy_question' else 'direct_answer'
g=StateGraph(State); g.add_node('classify_intent',classify_intent); g.add_node('retrieve_and_answer',retrieve_and_answer); g.add_node('direct_answer',direct_answer); g.set_entry_point('classify_intent'); g.add_conditional_edges('classify_intent',route,{'retrieve_and_answer':'retrieve_and_answer','direct_answer':'direct_answer'}); g.add_edge('retrieve_and_answer',END); g.add_edge('direct_answer',END); graph=g.compile()
app=FastAPI(title='Zepto Policy Support Assistant')
@app.post('/ask',response_model=AskResponse)
def ask(request:AskRequest): return graph.invoke({'query':request.query})['response']
