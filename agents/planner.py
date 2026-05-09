
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from agents.retriever import RetrieverAgent
from agents.synthesizer import SynthesizerAgent
from tools.web_search import search_web
from memory.vector_store import VectorStore
import config
import json


PLANNER_SYSTEM_PROMPT = """You are a research planner. Given a user query, your job is to:
1. Decide if the query needs web search, memory retrieval, or both.
2. Break the query into 2-3 focused sub-questions if needed.
3. Return a JSON plan with this exact format:

{
  "needs_web_search": true or false,
  "needs_memory": true or false,
  "sub_questions": ["question 1", "question 2"]
}

Only return the JSON. No explanation."""


class PlannerAgent:
    """
    Orchestrates the full research pipeline:
    Plan → Search/Retrieve → Synthesize → Return cited summary
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            api_key=config.OPENAI_API_KEY,
            temperature=0
        )
        self.retriever = RetrieverAgent()
        self.synthesizer = SynthesizerAgent()
        self.vector_store = VectorStore()
        self.conversation_history = []  # sliding window memory

    def _build_plan(self, query: str) -> dict:
        """Ask the LLM to create a search/retrieval plan for the query."""
        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=query)
        ]
        response = self.llm.invoke(messages)
        try:
            plan = json.loads(response.content)
        except json.JSONDecodeError:
            # Fallback: do both search and retrieval
            plan = {
                "needs_web_search": True,
                "needs_memory": True,
                "sub_questions": [query]
            }
        return plan

    def _update_memory(self, query: str, summary: str):
        """Store this exchange in the sliding window + vector store."""
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": summary})

        # Keep only the last MAX_MEMORY_TURNS turns (sliding window)
        if len(self.conversation_history) > config.MAX_MEMORY_TURNS * 2:
            self.conversation_history = self.conversation_history[-(config.MAX_MEMORY_TURNS * 2):]

        # Persist to FAISS for future retrieval
        self.vector_store.add_text(f"Q: {query}\nA: {summary}")

    def run(self, query: str) -> dict:
        """
        Full pipeline:
        1. Plan the approach
        2. Run web search and/or memory retrieval
        3. Synthesize into a cited summary
        """
        # Step 1: Plan
        plan = self._build_plan(query)
        print(f"[Planner] Plan: web_search={plan['needs_web_search']}, memory={plan['needs_memory']}")
        print(f"[Planner] Sub-questions: {plan['sub_questions']}")

        gathered_info = []
        sources = []

        # Step 2a: Web search for each sub-question
        if plan["needs_web_search"]:
            for sub_q in plan["sub_questions"]:
                print(f"\n[Search Agent] Searching: '{sub_q}'")
                search_results = search_web(sub_q)
                for r in search_results:
                    gathered_info.append(r["content"])
                    sources.append(r["url"])

        # Step 2b: Memory/vector retrieval
        if plan["needs_memory"]:
            print(f"\n[Retriever Agent] Querying memory for: '{query}'")
            retrieved_docs = self.retriever.retrieve(query)
            for doc in retrieved_docs:
                gathered_info.append(doc)
                sources.append("(from memory)")

        # Step 3: Synthesize
        print("\n[Synthesizer] Generating cited summary...")
        summary = self.synthesizer.synthesize(
            query=query,
            context=gathered_info,
            conversation_history=self.conversation_history
        )

        # Update memory
        self._update_memory(query, summary)

        # Deduplicate sources
        unique_sources = list(dict.fromkeys(s for s in sources if s != "(from memory)"))
        if not unique_sources:
            unique_sources = ["Generated from memory context"]

        return {
            "summary": summary,
            "sources": unique_sources
        }
