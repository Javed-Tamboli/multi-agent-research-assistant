# Multi-Agent AI Research Assistant

A production-style multi-agent AI system built with LangChain and OpenAI GPT-4o. 
Specialized sub-agents handle web search, document retrieval, and synthesis — 
orchestrated via a planner-executor pattern with persistent memory and structured outputs.

---

## Architecture

Planner Agent
    ├── Web Search Agent      → fetches real-time sources
    ├── Retriever Agent       → queries FAISS vector store
    └── Synthesizer Agent     → compiles cited research summary

---

## Tech Stack

| Layer       | Tools                              |
|-------------|------------------------------------|
| LLM         | OpenAI GPT-4o, function calling    |
| Orchestration | LangChain, Planner-Executor      |
| Memory      | FAISS vector store, sliding window |
| Deployment  | AWS Lambda                         |
| Output      | Structured JSON with citations     |

---

## Key Features

- Multi-agent orchestration with specialized tool-use per agent
- Persistent memory via FAISS — context retained across sessions
- Sub-2s latency on AWS Lambda for cited summaries
- Structured output validation using Pydantic

---

## Setup

git clone https://github.com/javedtamboli/multi-agent-research-assistant
cd multi-agent-research-assistant
pip install -r requirements.txt

Add your API key:
OPENAI_API_KEY=your_key_here

Run:
python main.py

---

## Results

| Metric                  | Value        |
|-------------------------|--------------|
| Response latency        | < 2 seconds  |
| Memory retention turns  | 15+ turns    |
| Deployment              | AWS Lambda   |
