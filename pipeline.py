"""End-to-end question answering: leave agent -> rewrite -> hybrid search -> guardrail -> grounded answer."""
import re
import time

from app import config, db
from app.agent import tools
from app.rag import citations, generator

ESCALATION_TEXT = (
    "I couldn't find this in the HR policies, so I've forwarded your question to the HR team "
    "(ticket #{tid}). You can see the reply on the Tickets page.\n\n"
    "मुझे यह जानकारी HR नीतियों में नहीं मिली, इसलिए आपका सवाल HR टीम को भेज दिया गया है (टिकट #{tid})."
)


def _escalate(user, question, create_ticket, started, query_en, reason):
    tid = db.create_ticket(user["id"], question) if create_ticket else None
    return {"answer": ESCALATION_TEXT.format(tid=tid or "-"), "sources": [], "route": "escalated",
            "ticket_id": tid, "latency_ms": int((time.time() - started) * 1000), "query_en": query_en,
            "reason": reason}


def ask(user: dict, question: str, history: list[tuple[str, str]], retriever, create_ticket: bool = True) -> dict:
    started = time.time()
    level = config.ROLE_LEVEL.get(user["role"], 1)

    # 1) Personal leave questions / actions go to the tool-calling agent
    if tools.looks_personal(question):
        reply = tools.run_agent(user, question, history)
        if reply:
            return {"answer": reply, "sources": [], "route": "agent", "ticket_id": None,
                    "latency_ms": int((time.time() - started) * 1000), "query_en": question}

    # 2) Policy questions go through RAG
    query_en = generator.rewrite_query(question, history)
    hits = retriever.search(query_en, level=level)
    if not hits or hits[0]["rerank_score"] < config.MIN_RERANK_SCORE:
        return _escalate(user, question, create_ticket, started, query_en, "low_retrieval_score")

    text = generator.answer(question, hits, history)
    if "NOT_FOUND" in text:
        return _escalate(user, question, create_ticket, started, query_en, "llm_not_found")

    used = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 0 < int(n) <= len(hits)}) or \
        list(range(1, len(hits) + 1))
    sources = []
    for n in used:
        c = hits[n - 1]
        sentence = citations.best_sentence(c["text"], query_en, text)
        sources.append({"n": n, "source": c["source"], "page": c["page"], "heading": c["heading"],
                        "text": citations.highlight(c["text"], sentence),
                        "score": round(c["rerank_score"], 2)})
    return {"answer": text, "sources": sources, "route": "rag", "ticket_id": None,
            "latency_ms": int((time.time() - started) * 1000), "query_en": query_en}
