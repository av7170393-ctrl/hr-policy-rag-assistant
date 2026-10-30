"""Streamlit frontend. Run: streamlit run ui/streamlit_app.py  (API must be running)."""
import hashlib
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="Nexora HR Assistant", page_icon="💼", layout="wide")


def call(method, path, **kw):
    headers = {"Authorization": f"Bearer {st.session_state['token']}"} if st.session_state.get("token") else {}
    try:
        r = requests.request(method, API + path, headers=headers, timeout=180, **kw)
    except requests.RequestException as e:
        st.error(f"Cannot reach the API at {API}: {e}")
        st.stop()
    if r.status_code == 401 and st.session_state.get("token"):
        st.session_state.clear()
        st.rerun()
    return r


def detail(r):
    try:
        return r.json().get("detail", r.text)
    except Exception:
        return r.text


def pretty(name):
    return Path(name).stem.replace("_", " ").title()


# ---------------- login ----------------
def login_view():
    st.title("💼 Nexora HR Assistant")
    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        r = call("POST", "/auth/login", json={"username": username, "password": password})
        if r.status_code == 200:
            data = r.json()
            st.session_state.update(token=data["token"], user=data["user"])
            st.rerun()
        else:
            st.error(detail(r))
    with st.expander("Demo accounts"):
        st.code("employee / emp123\nmanager  / mgr123\nhr       / hr123")


# ---------------- chat ----------------
BADGES = {"rag": "📚 Policy answer", "agent": "🤖 Leave assistant", "escalated": "🎫 Escalated to HR"}


def render_turn(m):
    with st.chat_message("user"):
        st.write(m["question"])
    with st.chat_message("assistant"):
        st.caption(BADGES.get(m["route"], m["route"]))
        st.markdown(m["answer"])
        if m["sources"]:
            with st.expander(f"Sources ({len(m['sources'])})"):
                for s in m["sources"]:
                    st.markdown(f"**[{s['n']}] {pretty(s['source'])} › {s['heading']}** · page {s['page']}")
                    st.markdown(s["text"])
        c1, c2, c3, _ = st.columns([1, 1, 2, 6])
        if c1.button("👍", key=f"up{m['id']}"):
            call("POST", "/feedback", json={"message_id": m["id"], "rating": 1})
            st.toast("Thanks for the feedback!")
        if c2.button("👎", key=f"down{m['id']}"):
            call("POST", "/feedback", json={"message_id": m["id"], "rating": -1})
            st.toast("Thanks, HR will review this answer.")
        if c3.button("🔊 Listen", key=f"tts{m['id']}"):
            r = call("POST", "/voice/speak", json={"text": m["answer"]})
            if r.status_code == 200:
                st.audio(r.content, format="audio/mp3")
            else:
                st.error(detail(r))


def chat_view():
    st.header("💬 Ask HR")
    if "chat" not in st.session_state:
        st.session_state["chat"] = call("GET", "/history?limit=30").json()
    if not st.session_state["chat"]:
        st.info("Ask about leave, WFH, attendance, benefits… in English, Hindi or Hinglish. "
                "Try: “How many casual leaves do I have left?” or “Notice period kitna hai?”")
    for m in st.session_state["chat"]:
        render_turn(m)

    audio = st.audio_input("🎤 Ask by voice")
    question = st.chat_input("Ask an HR policy question…")
    if audio is not None:
        data = audio.getvalue()
        digest = hashlib.md5(data).hexdigest()
        if st.session_state.get("last_audio") != digest:
            st.session_state["last_audio"] = digest
            with st.spinner("Transcribing…"):
                r = call("POST", "/voice/transcribe", files={"file": ("voice.wav", data, "audio/wav")})
            if r.status_code == 200:
                question = r.json()["text"] or None
            else:
                st.error(detail(r))
    if question:
        with st.spinner("Searching policies…"):
            r = call("POST", "/chat", json={"question": question})
        if r.status_code == 200:
            d = r.json()
            st.session_state["chat"].append({"id": d["message_id"], "question": question, "answer": d["answer"],
                                             "sources": d["sources"], "route": d["route"]})
            st.rerun()
        else:
            st.error(detail(r))


# ---------------- leave ----------------
def leave_view():
    st.header("🏖️ My leave")
    balances = call("GET", "/leave/balance").json()
    for col, (kind, days) in zip(st.columns(max(len(balances), 1)), sorted(balances.items())):
        col.metric(f"{kind.title()} leave", f"{days:g} days")
    with st.form("apply"):
        st.subheader("Apply for leave")
        kind = st.selectbox("Type", ["casual", "sick", "earned"])
        start, end = st.columns(2)
        from_date = start.date_input("From")
        to_date = end.date_input("To")
        reason = st.text_input("Reason")
        if st.form_submit_button("Submit request"):
            r = call("POST", "/leave/apply", json={"leave_type": kind, "start_date": from_date.isoformat(),
                                                   "end_date": to_date.isoformat(), "reason": reason})
            if r.status_code == 200:
                d = r.json()
                st.success(f"Request #{d['request_id']} submitted for {d['days']:g} day(s). {d['note']}")
            else:
                st.error(detail(r))
    requests_ = call("GET", "/leave/requests").json()
    if requests_:
        st.subheader("My requests")
        st.dataframe(pd.DataFrame(requests_)[["id", "leave_type", "start_date", "end_date", "days", "status"]],
                     hide_index=True, use_container_width=True)


def approvals_view():
    st.header("✅ Leave approvals")
    pending = call("GET", "/leave/pending").json()
    if not pending:
        st.success("No pending requests.")
    for r in pending:
        with st.container(border=True):
            st.markdown(f"**{r['full_name']}** · {r['leave_type']} · {r['start_date']} → {r['end_date']} "
                        f"({r['days']:g} days)\n\n{r['reason'] or '_No reason given_'}")
            a, b, _ = st.columns([1, 1, 6])
            if a.button("Approve", key=f"ap{r['id']}"):
                call("POST", f"/leave/{r['id']}/decision", json={"approve": True}); st.rerun()
            if b.button("Reject", key=f"rj{r['id']}"):
                call("POST", f"/leave/{r['id']}/decision", json={"approve": False}); st.rerun()


# ---------------- tickets ----------------
def tickets_view():
    st.header("🎫 HR tickets")
    is_hr = st.session_state["user"]["role"] == "hr"
    show_all = is_hr and st.toggle("Show all employees’ tickets", value=True)
    tickets = call("GET", f"/tickets?all={str(show_all).lower()}").json()
    if not tickets:
        st.info("No tickets yet. Questions the assistant can’t answer are escalated here automatically.")
    for t in tickets:
        with st.container(border=True):
            st.markdown(f"**#{t['id']}** · {t['full_name']} · `{t['status']}`\n\n{t['question']}")
            if t["reply"]:
                st.success(f"HR reply: {t['reply']}")
            elif is_hr:
                reply = st.text_input("Reply", key=f"reply{t['id']}")
                if st.button("Resolve", key=f"res{t['id']}") and reply:
                    call("POST", f"/tickets/{t['id']}/resolve", json={"reply": reply}); st.rerun()


# ---------------- admin ----------------
def admin_view():
    st.header("🛠️ HR admin")
    tab_a, tab_d, tab_f = st.tabs(["Analytics", "Documents", "Weak answers"])
    with tab_a:
        a = call("GET", "/admin/analytics").json()
        c = st.columns(4)
        c[0].metric("Questions", a["total_questions"])
        c[1].metric("Escalation rate", f"{a['escalation_rate']:.0%}")
        c[2].metric("Helpful rate", "n/a" if a["helpful_rate"] is None else f"{a['helpful_rate']:.0%}")
        c[3].metric("Avg latency", f"{a['avg_latency_ms']} ms")
        if a["top_questions"]:
            st.subheader("Most asked questions")
            st.bar_chart(pd.DataFrame(a["top_questions"]).set_index("question")["count"])
        if a["unanswered"]:
            st.subheader("Recently unanswered (policy gaps)")
            st.dataframe(pd.DataFrame(a["unanswered"]), hide_index=True, use_container_width=True)
    with tab_d:
        st.write("Upload a policy document. It is indexed automatically and only visible to the chosen role and above.")
        up = st.file_uploader("Policy file", type=["pdf", "md", "txt"])
        role = st.selectbox("Visible to", ["employee", "manager", "hr"])
        if st.button("Upload and index") and up:
            r = call("POST", "/admin/upload", files={"file": (up.name, up.getvalue())}, data={"min_role": role})
            st.success(r.json()["message"]) if r.status_code == 200 else st.error(detail(r))
        if st.button("Re-index all documents"):
            st.success(call("POST", "/admin/reindex").json()["message"])
    with tab_f:
        rows = call("GET", "/admin/feedback").json()
        if rows:
            st.dataframe(pd.DataFrame(rows)[["created_at", "question", "answer", "route", "comment"]],
                         hide_index=True, use_container_width=True)
        else:
            st.info("No 👎 feedback yet.")


# ---------------- main ----------------
if not st.session_state.get("token"):
    login_view()
    st.stop()

user = st.session_state["user"]
pages = {"💬 Ask HR": chat_view, "🏖️ My leave": leave_view, "🎫 Tickets": tickets_view}
if user["role"] in ("manager", "hr"):
    pages["✅ Approvals"] = approvals_view
if user["role"] == "hr":
    pages["🛠️ Admin"] = admin_view
with st.sidebar:
    st.markdown(f"**{user['full_name']}**  \n`{user['role']}`")
    page = st.radio("Go to", list(pages))
    if st.button("Sign out"):
        st.session_state.clear()
        st.rerun()
pages[page]()
