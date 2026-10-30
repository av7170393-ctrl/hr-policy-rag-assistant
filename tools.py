"""Leave assistant: business rules + LLM tool calling (check balance, apply for leave)."""
import json
import re
from datetime import date, datetime, timedelta

from app import db

LEAVE_TYPES = {"casual", "sick", "earned"}
_LEAVE_WORD = re.compile(r"(leave|chutti|chhutti|छुट्टी|casual|sick|earned)", re.I)
_BALANCE = re.compile(r"(balance|remaining|\bleft\b|bache|bacha|baki|बची|बाकी|शेष)", re.I)
_ACTION = re.compile(r"(\bapply\b|request|\bbook\b|take (a )?(day|leave)|(want|need).{0,20}leave|"
                     r"(chutti|छुट्टी) (chahiye|lena|लेनी|चाहिए))", re.I)


def looks_personal(question: str) -> bool:
    """Cheap pre-check so general policy questions skip the tool-calling step."""
    return bool(_LEAVE_WORD.search(question) and (_BALANCE.search(question) or _ACTION.search(question)))


def working_days(start: date, end: date) -> int:
    days, d = 0, start
    while d <= end:
        days += d.weekday() < 5
        d += timedelta(days=1)
    return days


def apply_leave(user: dict, leave_type: str, start_date: str, end_date: str, reason: str = "") -> dict:
    lt = (leave_type or "").lower().strip()
    if lt not in LEAVE_TYPES:
        return {"ok": False, "error": "Leave type must be casual, sick or earned."}
    try:
        start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
    except (TypeError, ValueError):
        return {"ok": False, "error": "Dates must be in YYYY-MM-DD format."}
    if end < start:
        return {"ok": False, "error": "End date is before start date."}
    days = working_days(start, end)
    if days == 0:
        return {"ok": False, "error": "The selected dates have no working days (weekend only)."}
    if lt == "casual" and days > 3:
        return {"ok": False, "error": "Policy: at most 3 casual leaves can be taken consecutively."}
    if lt == "earned" and (start - date.today()).days < 7:
        return {"ok": False, "error": "Policy: earned leave must be applied at least 7 days in advance."}
    balance = db.get_balances(user["id"]).get(lt, 0)
    if days > balance:
        return {"ok": False, "error": f"Not enough {lt} leave: you need {days} day(s) but have {balance:g}."}
    note = "Medical certificate must be submitted to HR within 3 days of returning." \
        if lt == "sick" and days > 2 else ""
    rid = db.add_leave_request(user["id"], lt, start_date, end_date, days, reason)
    db.adjust_balance(user["id"], lt, -days)
    return {"ok": True, "request_id": rid, "days": days, "status": "pending",
            "balance_left": balance - days, "note": note}


def decide_leave(request_id: int, approve: bool) -> dict:
    req = db.get_leave_request(request_id)
    if not req:
        return {"ok": False, "error": "Request not found."}
    if req["status"] != "pending":
        return {"ok": False, "error": f"Request is already {req['status']}."}
    db.set_leave_status(request_id, "approved" if approve else "rejected")
    if not approve:  # give the days back
        db.adjust_balance(req["user_id"], req["leave_type"], req["days"])
    return {"ok": True, "status": "approved" if approve else "rejected"}


TOOLS = [
    {"type": "function", "function": {
        "name": "get_leave_balance",
        "description": "Get the signed-in employee's remaining casual, sick and earned leave days.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "apply_leave",
        "description": "Submit a leave request for the signed-in employee. Only call when type and both dates are known.",
        "parameters": {"type": "object", "properties": {
            "leave_type": {"type": "string", "enum": ["casual", "sick", "earned"]},
            "start_date": {"type": "string", "description": "YYYY-MM-DD"},
            "end_date": {"type": "string", "description": "YYYY-MM-DD"},
            "reason": {"type": "string"}},
            "required": ["leave_type", "start_date", "end_date"]}}},
]


def _execute(user: dict, name: str, args: dict) -> dict:
    if name == "get_leave_balance":
        return db.get_balances(user["id"])
    if name == "apply_leave":
        try:
            return apply_leave(user, **args)
        except TypeError:
            return {"ok": False, "error": "Invalid arguments for apply_leave."}
    return {"error": f"Unknown tool {name}"}


def run_agent(user: dict, question: str, history: list[tuple[str, str]]) -> str | None:
    """Returns a reply, or None when the question is not about the user's own leave."""
    from app.rag import generator
    system = (f"You are the HR assistant for Nexora Technologies. Today is {date.today().isoformat()} "
              f"({datetime.now().strftime('%A')}). Use tools only for the signed-in employee's own leave "
              "balance or for applying leave. If the request is about anything else, reply with exactly NO_TOOL. "
              "If details needed to apply for leave (type, start date, end date) are missing, ask one short "
              "question. Reply in the employee's language.")
    messages = [{"role": "system", "content": system}]
    messages += [{"role": r, "content": t} for r, t in history[-4:]]
    messages.append({"role": "user", "content": question})

    msg = generator.chat(messages, tools=TOOLS, tool_choice="auto", max_tokens=900).choices[0].message
    if not msg.tool_calls:
        text = (msg.content or "").strip()
        return None if (not text or "NO_TOOL" in text) else text

    messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
        {"id": tc.id, "type": "function",
         "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in msg.tool_calls]})
    for tc in msg.tool_calls:
        result = _execute(user, tc.function.name, json.loads(tc.function.arguments or "{}"))
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
    return (generator.chat(messages, max_tokens=700).choices[0].message.content or "").strip()
