"""Blinded human evaluation of extracted design decisions (Streamlit).

Each rater sees, for every sampled commit, the commit context and the developers' ground-truth
decision, followed by the systems' outputs labelled only "Output A-D" and in a random order, and
rates each output 0-3 on completeness and conciseness, with an optional comment. Commit order and
output order are shuffled per rater (deterministically, so a rater sees the same order when they
come back). The system behind each slot is stored with the rating but never sent to the browser.

Responses go to a Google Sheet when `gcp_service_account` and `sheet_url` are set in Streamlit
secrets, and to responses.csv next to this file otherwise (for local testing).

    streamlit run app.py
"""

import csv
import html
import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
ITEMS = json.loads((HERE / "data/eval_items.json").read_text())
EVALS = ITEMS["evals"]
COMMITS = {item["issue_key"]: item for item in ITEMS["items"]}
CSV_PATH = HERE / "responses.csv"

SCORES = [0, 1, 2, 3]
RUBRIC = {
    "completeness": ("Completeness", "How well the output covers the ground-truth decision.", {
        0: "Major components missing or incorrect",
        1: "Partial coverage; key information missing",
        2: "Most components captured; minor omissions",
        3: "Complete and accurate coverage",
    }),
    "conciseness": ("Conciseness", "How well the output stays at the architectural level.", {
        0: "Dominated by implementation details",
        1: "Mix of architectural and implementation details",
        2: "Mostly architectural but somewhat verbose",
        3: "Clear and concise architectural abstraction",
    }),
}
COLUMNS = ["timestamp", "rater", "eval", "issue_key", "slot", "system",
           "completeness", "conciseness", "comment", "seconds_on_page"]


# ---------- storage ----------

def _secret(name: str):
    try:
        return st.secrets.get(name)
    except Exception:  # no secrets.toml at all (local testing)
        return None


def _gsheets_configured() -> bool:
    return bool(_secret("gcp_service_account") and _secret("sheet_url"))


@st.cache_resource
def _worksheet():
    import gspread
    client = gspread.service_account_from_dict(dict(st.secrets["gcp_service_account"]))
    ws = client.open_by_url(st.secrets["sheet_url"]).sheet1
    if ws.row_values(1) != COLUMNS:
        if ws.row_values(1):
            raise RuntimeError("The first row of the sheet is not the expected header; use an empty sheet.")
        ws.append_row(COLUMNS)
    return ws


def load_rows(rater: str) -> list[dict]:
    if _gsheets_configured():
        rows = _worksheet().get_all_records(expected_headers=COLUMNS)
    elif CSV_PATH.exists():
        with CSV_PATH.open(newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        rows = []
    return [r for r in rows if str(r["rater"]) == rater]


def append_rows(rows: list[dict]):
    if _gsheets_configured():
        # RAW so a comment starting with "=" is stored as text, not run as a formula.
        _worksheet().append_rows([[r[c] for c in COLUMNS] for r in rows], value_input_option="RAW")
        return
    new = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            writer.writeheader()
        writer.writerows(rows)


# ---------- per-rater blinding and order ----------

def commit_order(rater: str, eval_id: str) -> list[str]:
    keys = sorted(COMMITS)
    random.Random(f"{ITEMS['seed']}|{rater}|{eval_id}").shuffle(keys)
    return keys


def slot_assignment(rater: str, eval_id: str, issue_key: str) -> dict[str, str]:
    """Slot letter -> system id, shuffled per (rater, eval, commit)."""
    systems = list(EVALS[eval_id]["systems"])
    random.Random(f"{ITEMS['seed']}|{rater}|{eval_id}|{issue_key}").shuffle(systems)
    return {chr(ord("A") + i): s for i, s in enumerate(systems)}


def render_output(text: str) -> str:
    """Make model Markdown safe to show inside a card: demote headings, stop $...$ rendering as LaTeX."""
    text = re.sub(r"^#{1,3}(?=\s)", "####", text, flags=re.M)
    return text.replace("$", r"\$")


# ---------- session ----------

def saved_ratings(rater: str) -> dict:
    """(eval, issue_key, system) -> latest rating row for this rater; later rows override earlier ones."""
    if "saved" not in st.session_state:
        st.session_state.saved = {(r["eval"], r["issue_key"], r["system"]): r for r in load_rows(rater)}
    return st.session_state.saved


def is_done(saved: dict, eval_id: str, issue_key: str) -> bool:
    return all((eval_id, issue_key, s) in saved for s in EVALS[eval_id]["systems"])


# ---------- look and feel ----------

CSS = """
<style>
.block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1560px; }
header[data-testid="stHeader"] { background: transparent; }

/* hero / context */
.hero { background: linear-gradient(120deg, #4F46E5 0%, #7C3AED 55%, #A855F7 100%); color: #fff;
        border-radius: 18px; padding: 22px 28px; margin-bottom: 14px; box-shadow: 0 10px 30px rgba(79,70,229,.25); }
.hero .eyebrow { font-size: .78rem; letter-spacing: .08em; text-transform: uppercase; opacity: .85; font-weight: 600; }
.hero h1 { font-size: 1.55rem; line-height: 1.3; margin: 6px 0 0 0; padding: 0; color: #fff; font-weight: 700; }
.label { font-size: .74rem; letter-spacing: .08em; text-transform: uppercase; font-weight: 700; margin-bottom: 8px; }
.commit { background: #0F172A; border-radius: 14px; padding: 16px 20px; margin-bottom: 14px; }
.commit .label { color: #94A3B8; }
.commit .msg { margin: 0; padding: 0; white-space: pre-wrap; word-break: break-word;
              color: #E2E8F0 !important; background: transparent !important; border: 0 !important;
              font-family: 'Source Code Pro', ui-monospace, monospace; font-size: .85rem; line-height: 1.6; }
.gt { background: #ECFDF5; border: 1px solid #A7F3D0; border-left: 6px solid #10B981; border-radius: 14px;
      padding: 16px 22px; margin-bottom: 8px; }
.gt .label { color: #047857; }
.gt .body { color: #064E3B; white-space: pre-wrap; line-height: 1.6; font-size: .98rem; }
.hint { color: #6B7280; font-size: .9rem; margin: 6px 2px 14px 2px; }

/* output cards (containers keyed card_A ... card_D) */
div[class*="st-key-card_"] { background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 18px;
    padding: 20px 22px 16px 22px; box-shadow: 0 1px 2px rgba(16,24,40,.04), 0 8px 24px rgba(16,24,40,.06); }
div[class*="st-key-card_"] h4 { font-size: 1.02rem; margin-top: .6rem; }
.slot { display: flex; align-items: center; gap: 12px; margin-bottom: 4px; }
.slot .badge { width: 38px; height: 38px; border-radius: 12px; background: #EEF2FF; color: #4338CA;
               display: inline-flex; align-items: center; justify-content: center; font-weight: 700; font-size: 1.1rem; }
.slot .name { font-weight: 700; font-size: 1.1rem; color: #111827; }
.empty { background: #F9FAFB; border: 1px dashed #D1D5DB; color: #6B7280; border-radius: 12px; padding: 28px;
         text-align: center; font-style: italic; }
div[class*="st-key-rate_"] { border-top: 1px solid #EEF2F7; padding-top: 14px; margin-top: 6px; }
div[class*="st-key-rate_"] button[data-variant="segmented_control"][aria-checked="false"]:not([data-hovered]) {
    background: #FFFFFF; }

/* status line between the arrows */
.status { text-align: center; color: #374151; font-size: .92rem; }
.status b { color: #111827; }
.pill-ok { background: #DCFCE7; color: #166534; border-radius: 999px; padding: 2px 10px; font-weight: 600; font-size: .8rem; }
.pill-todo { background: #FEF3C7; color: #92400E; border-radius: 999px; padding: 2px 10px; font-weight: 600; font-size: .8rem; }

/* sidebar */
section[data-testid="stSidebar"] { background: #F5F6FB; border-right: 1px solid #E5E7EB; }
.who { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
.who .avatar { width: 36px; height: 36px; border-radius: 50%; background: linear-gradient(135deg,#6366F1,#A855F7);
               color: #fff; display: inline-flex; align-items: center; justify-content: center; font-weight: 700; }
.who .meta { font-size: .78rem; color: #6B7280; line-height: 1.2; }
.who .meta b { display: block; color: #111827; font-size: .95rem; }
.rubric h5 { margin: 10px 0 6px 0; font-size: .9rem; }
.rubric .q { color: #6B7280; font-size: .8rem; margin-bottom: 6px; }
.rubric .row { display: flex; gap: 8px; align-items: flex-start; margin: 4px 0; font-size: .82rem; color: #374151; }
.rubric .n { flex: 0 0 22px; height: 22px; border-radius: 6px; background: #EEF2FF; color: #4338CA; font-weight: 700;
             display: inline-flex; align-items: center; justify-content: center; font-size: .78rem; }

/* login */
.login-hero { text-align: center; margin: 5vh 0 18px 0; }
.login-hero .logo { font-size: 2.6rem; }
.login-hero h1 { font-size: 1.8rem; margin: 6px 0 4px 0; }
.login-hero p { color: #6B7280; }
</style>
"""


def esc(text: str) -> str:
    return html.escape(str(text))


def login():
    # Personal links (?rater=<name>) sign in directly, unless an access code is required.
    linked = st.query_params.get("rater", "").strip()
    if linked and not _secret("access_code"):
        st.session_state.rater = linked.lower()
        st.rerun()
    _, mid, _ = st.columns([1, 1.3, 1])
    with mid:
        st.markdown("<div class='login-hero'><div class='logo'>🧭</div><h1>Design decision evaluation</h1>"
                    "<p>Rate how well each system's output captures the architectural decision behind a commit.</p>"
                    "</div>", unsafe_allow_html=True)
        with st.form("login", border=True):
            name = st.text_input("Rater name / ID", value=linked, placeholder="e.g. rater01")
            code = st.text_input("Access code", type="password") if _secret("access_code") else None
            st.caption("Use the same name each time so your progress is restored.")
            if st.form_submit_button("Start evaluating →", type="primary", use_container_width=True):
                if not name.strip():
                    st.error("Please enter your name or ID.")
                elif code is not None and code != _secret("access_code"):
                    st.error("Wrong access code.")
                else:
                    st.session_state.rater = name.strip().lower()
                    st.rerun()


def show_rubric():
    parts = ["<div class='rubric'>"]
    for label, question, levels in RUBRIC.values():
        parts.append(f"<h5>{label}</h5><div class='q'>{question}</div>")
        parts += [f"<div class='row'><span class='n'>{k}</span><span>{v}</span></div>" for k, v in levels.items()]
    st.markdown("".join(parts) + "</div>", unsafe_allow_html=True)


def sidebar(rater: str, saved: dict) -> str:
    with st.sidebar:
        st.markdown(f"<div class='who'><span class='avatar'>{esc(rater[:1].upper())}</span>"
                    f"<span class='meta'>Signed in as<b>{esc(rater)}</b></span></div>", unsafe_allow_html=True)
        eval_id = st.segmented_control(
            "Evaluation", list(EVALS), key="eval_id", default=list(EVALS)[0], selection_mode="single",
            format_func=lambda e: EVALS[e]["title"].split(":")[0], width="stretch") or list(EVALS)[0]
        st.caption(EVALS[eval_id]["title"].split(": ", 1)[-1])
        for eid in EVALS:
            done = sum(is_done(saved, eid, k) for k in COMMITS)
            st.progress(done / len(COMMITS), text=f"{EVALS[eid]['title']} · **{done}/{len(COMMITS)}**")

        order = commit_order(rater, eval_id)
        pos_key = f"pos_{eval_id}"
        if f"next_{eval_id}" in st.session_state:  # set after a save; applied before the widget exists
            st.session_state[pos_key] = st.session_state.pop(f"next_{eval_id}")
        if pos_key not in st.session_state:
            st.session_state[pos_key] = next(
                (i for i, k in enumerate(order) if not is_done(saved, eval_id, k)), 0)
        st.selectbox(
            "Go to commit", range(len(order)), key=pos_key,
            format_func=lambda i: f"{'✅' if is_done(saved, eval_id, order[i]) else '⬜'}  Commit {i + 1}")
        with st.expander("📏 Rating rubric", expanded=True):
            show_rubric()
        if st.button("Sign out", use_container_width=True):
            st.session_state.clear()
            st.query_params.clear()
            st.rerun()
    return eval_id


def _go_to(pos_key: str, pos: int):
    # Runs as a callback, i.e. before the sidebar's "Go to commit" selectbox is drawn, so it may set its state.
    st.session_state[pos_key] = pos


def nav_buttons(eval_id: str, pos: int, n: int, saved: dict, order: list[str], where: str):
    """← Previous / Next → around a status line. Moving away discards unsaved ratings on the page."""
    pos_key = f"pos_{eval_id}"
    left, mid, right = st.columns([1, 4, 1], vertical_alignment="center")
    hint = "Unsaved ratings on this page are discarded; use Save and continue to keep them."
    left.button("← Previous", key=f"prev_{where}", disabled=pos == 0, use_container_width=True,
                on_click=_go_to, args=(pos_key, pos - 1), help=hint)
    right.button("Next →", key=f"next_{where}", disabled=pos == n - 1, use_container_width=True,
                 on_click=_go_to, args=(pos_key, pos + 1), help=hint)
    pill = ("<span class='pill-ok'>Saved</span>" if is_done(saved, eval_id, order[pos])
            else "<span class='pill-todo'>Not yet saved</span>")
    mid.markdown(f"<div class='status'>Commit <b>{pos + 1}</b> of <b>{n}</b> &nbsp;·&nbsp; {pill}</div>",
                 unsafe_allow_html=True)


def rate_commit(rater: str, eval_id: str, saved: dict):
    order = commit_order(rater, eval_id)
    pos_key = f"pos_{eval_id}"
    pos = st.session_state[pos_key]
    issue_key = order[pos]
    item = COMMITS[issue_key]
    slots = slot_assignment(rater, eval_id, issue_key)

    timer_key = f"start_{eval_id}_{issue_key}"
    st.session_state.setdefault(timer_key, time.time())

    nav_buttons(eval_id, pos, len(order), saved, order, "top")
    st.markdown(
        f"<div class='hero'><div class='eyebrow'>{esc(issue_key)}</div><h1>{esc(item['title'])}</h1></div>",
        unsafe_allow_html=True)
    messages = "\n\n".join(item["commit_messages"]) or "(no commit message)"
    st.markdown(f"<div class='commit'><div class='label'>Commit message</div><div class='msg'>{esc(messages)}</div></div>",
                unsafe_allow_html=True)
    st.markdown(f"<div class='gt'><div class='label'>Ground truth · reference decision</div>"
                f"<div class='body'>{esc(item['ground_truth'])}</div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='hint'>Rate each of the {len(slots)} outputs below against the ground truth. "
                "They are in random order; neither position nor length indicates quality.</div>",
                unsafe_allow_html=True)

    with st.form(f"form_{eval_id}_{issue_key}", border=False):
        answers = {}
        items = list(slots.items())
        # 2x2 grid: one row of two cards per pair of slots; cards grow to fit their content.
        cells = [cell for i in range(0, len(items), 2) for cell in st.columns(2, gap="medium")]
        for cell, (slot, system) in zip(cells, items):
            prev = saved.get((eval_id, issue_key, system), {})
            wkey = f"{eval_id}|{issue_key}|{slot}"
            with cell, st.container(key=f"card_{slot}"):
                st.markdown(f"<div class='slot'><span class='badge'>{slot}</span>"
                            f"<span class='name'>Output {slot}</span></div>", unsafe_allow_html=True)
                text = item["outputs"][system]
                if text.strip():
                    st.markdown(render_output(text))
                else:
                    st.markdown("<div class='empty'>No output was produced.</div>", unsafe_allow_html=True)
                with st.container(key=f"rate_{slot}"):
                    cols = st.columns(2)
                    for col, metric in zip(cols, RUBRIC):
                        label, _, levels = RUBRIC[metric]
                        prev_score = prev.get(metric)
                        answers[(slot, metric)] = col.segmented_control(
                            label, SCORES, key=f"{wkey}|{metric}", selection_mode="single", width="stretch",
                            default=int(prev_score) if prev_score not in (None, "") else None,
                            help="  \n".join(f"**{k}**: {v}" for k, v in levels.items()))
                    answers[(slot, "comment")] = st.text_area(
                        "Comment (optional)", value=str(prev.get("comment", "")), key=f"{wkey}|comment",
                        height=68, placeholder="Anything notable about this output?")
        st.write("")
        submitted = st.form_submit_button("Save and continue  →", type="primary", use_container_width=True)
    nav_buttons(eval_id, pos, len(order), saved, order, "bottom")

    if submitted:
        missing = [f"Output {s} {RUBRIC[m][0].lower()}" for (s, m), v in answers.items() if m in RUBRIC and v is None]
        if missing:
            st.error("Please rate: " + ", ".join(missing))
            return
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        elapsed = round(time.time() - st.session_state[timer_key])
        rows = [{
            "timestamp": now, "rater": rater, "eval": eval_id, "issue_key": issue_key,
            "slot": slot, "system": system,
            "completeness": answers[(slot, "completeness")], "conciseness": answers[(slot, "conciseness")],
            "comment": answers[(slot, "comment")].strip(), "seconds_on_page": elapsed,
        } for slot, system in slots.items()]
        try:
            append_rows(rows)
        except Exception as exc:  # surface storage errors to the rater instead of losing the ratings silently
            st.error(f"Could not save your ratings, please try again. ({exc})")
            return
        for r in rows:
            saved[(eval_id, issue_key, r["system"])] = r
        st.session_state.pop(timer_key, None)
        remaining = [i for i, k in enumerate(order) if not is_done(saved, eval_id, k)]
        nxt = next((i for i in remaining if i > pos), remaining[0] if remaining else None)
        if nxt is None:
            st.session_state.finished = eval_id
        else:
            st.session_state[f"next_{eval_id}"] = nxt
            st.toast("Ratings saved", icon="✅")
        st.rerun()


def main():
    st.set_page_config(page_title="Design decision evaluation", page_icon="🧭", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    if "rater" not in st.session_state:
        login()
        return
    rater = st.session_state.rater
    saved = saved_ratings(rater)
    eval_id = sidebar(rater, saved)

    if st.session_state.pop("finished", None) == eval_id:
        st.balloons()
        st.success(f"You have rated every commit in {EVALS[eval_id]['title']}. Thank you! "
                   "You can switch evaluations in the sidebar, or revisit any commit to change your ratings.")
    rate_commit(rater, eval_id, saved)


main()
