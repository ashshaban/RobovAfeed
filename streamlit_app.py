import json
import re
from html import unescape

import feedparser
import requests
import streamlit as st
from groq import Groq

DEFAULT_SYSTEM_PROMPT = """
أنت KAIROS، عقله الاستراتيجي. هذا النظام ليس أداة نشر فقط، بل سلاح استراتيجي.
**المهمة:** إعادة صياغة خبر واحد كمنشور فيسبوك باللهجة المصرية.
**الهدف:** إثبات القيادة التقنية وربط الخبر برؤية مصر 2030 والسيادة الرقمية.
**الهيكل المطلوب:**
1) Hook: عنوان جذاب وقصير.
2) Insight: شرح الخبر ببساطة وبُعد تقني واضح.
3) Impact: ربط مباشر بمصر 2030/تأهيل الشباب/السيادة الرقمية.
4) Hashtags: #RoboVAI #DigitalEgypt #MohamedShaban 🇪🇬 🦾
**النبرة:** Techno-Statesman — سياسي/تقني/رؤيوي/مودرن.
""".strip()

DEFAULT_FEEDS = [
    {"name": "WEF - Industry 4.0", "url": "http://feeds.feedburner.com/wef-manufacturing"},
    {"name": "MIT Tech Review", "url": "https://www.technologyreview.com/feed/"},
    {"name": "TechCrunch", "url": "https://techcrunch.com/feed/"},
    {"name": "Egypt Today (Business)", "url": "https://www.egypttoday.com/Rss/24/Business"},
]

MODEL_OPTIONS = ["llama3-70b-8192", "mixtral-8x7b-32768"]


def strip_html(text: str) -> str:
    clean = re.sub(r"<[^>]+>", " ", text or "")
    clean = re.sub(r"\s+", " ", clean).strip()
    return unescape(clean)


def build_user_prompt(entry: dict) -> str:
    title = strip_html(entry.get("title", "(No title)"))
    summary = strip_html(entry.get("summary", "(No summary available)"))
    link = entry.get("link", "")
    published = entry.get("published", "")
    return (
        "أعد صياغة الخبر التالي إلى منشور فيسبوك وفق التعليمات.\n\n"
        f"العنوان: {title}\n\n"
        f"الملخص: {summary}\n\n"
        f"تاريخ النشر: {published}\n\n"
        f"الرابط: {link}"
    )


def call_groq(api_key: str, model: str, system_prompt: str, user_prompt: str, params: dict) -> str:
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=params["temperature"],
        max_tokens=params["max_tokens"],
        top_p=params["top_p"],
        frequency_penalty=params["frequency_penalty"],
        presence_penalty=params["presence_penalty"],
        stop=params["stop"],
    )
    return response.choices[0].message.content.strip()


def call_nvidia(api_key: str, base_url: str, model: str, system_prompt: str, user_prompt: str, params: dict) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": params["temperature"],
        "max_tokens": params["max_tokens"],
        "top_p": params["top_p"],
        "frequency_penalty": params["frequency_penalty"],
        "presence_penalty": params["presence_penalty"],
        "stop": params["stop"],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = ""
        try:
            detail = response.text.strip()
        except Exception:
            detail = ""
        raise requests.HTTPError(
            f"NVIDIA request failed: {response.status_code} for {url}\n{detail}"
        ) from exc
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def init_state() -> None:
    if "feeds" not in st.session_state:
        st.session_state.feeds = DEFAULT_FEEDS.copy()
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT


def main() -> None:
    st.set_page_config(page_title="NEXUS Intelligence Engine", page_icon="📡", layout="wide")
    init_state()

    st.title("📡 NEXUS Intelligence Engine")
    st.caption("Personal Intelligence Radar — مصمم ليبقيك القائد الأول في التحول الرقمي.")

    with st.sidebar:
        st.header("Control Room")
        provider = st.selectbox("Provider", ["Groq", "NVIDIA (OpenAI Compatible)"])
        api_key = st.text_input("API Key", type="password")

        model_block = st.container()
        with model_block:
            use_custom_model = st.checkbox("Use custom model", value=False)
            if use_custom_model:
                model = st.text_input("Model name", value="")
            else:
                model = st.selectbox("Model", MODEL_OPTIONS, index=0)

        base_url = ""
        if provider.startswith("NVIDIA"):
            base_url = st.text_input("NVIDIA Base URL", value="https://integrate.api.nvidia.com/v1")
            test_col, _ = st.columns([1, 2])
            with test_col:
                if st.button("Test NVIDIA Connection"):
                    if not api_key:
                        st.warning("Please enter your API key first.")
                    else:
                        try:
                            models_url = base_url.rstrip("/") + "/models"
                            resp = requests.get(
                                models_url,
                                headers={"Authorization": f"Bearer {api_key}"},
                                timeout=30,
                            )
                            resp.raise_for_status()
                            st.success("NVIDIA connection OK.")
                        except Exception as exc:
                            st.error(f"NVIDIA connection failed: {exc}")

        st.divider()
        st.subheader("Groq / NVIDIA Parameters")
        temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
        top_p = st.slider("Top P", 0.0, 1.0, 1.0, 0.05)
        max_tokens = st.slider("Max tokens", 128, 2048, 700, 64)
        frequency_penalty = st.slider("Frequency penalty", -2.0, 2.0, 0.0, 0.1)
        presence_penalty = st.slider("Presence penalty", -2.0, 2.0, 0.0, 0.1)
        stop_text = st.text_area("Stop sequences (one per line)", value="", height=80)

        st.divider()
        st.subheader("System Prompt")
        st.session_state.system_prompt = st.text_area(
            "Edit system prompt",
            value=st.session_state.system_prompt,
            height=260,
        )

        st.divider()
        st.subheader("Feed Manager")
        with st.expander("Add / Remove Feeds", expanded=True):
            for idx, feed in enumerate(list(st.session_state.feeds)):
                col_name, col_del = st.columns([4, 1])
                col_name.markdown(f"**{feed['name']}**")
                if col_del.button("Delete", key=f"delete_feed_{idx}"):
                    st.session_state.feeds.pop(idx)
                    st.rerun()

            st.markdown("---")
            new_name = st.text_input("Feed name", value="")
            new_url = st.text_input("Feed URL", value="")
            if st.button("Add feed"):
                if new_name and new_url:
                    st.session_state.feeds.append({"name": new_name.strip(), "url": new_url.strip()})
                    st.rerun()
                else:
                    st.warning("Please provide both name and URL.")

    feeds = st.session_state.feeds
    if not feeds:
        st.warning("No feeds available. Add at least one feed in the sidebar.")
        st.stop()

    feed_names = [feed["name"] for feed in feeds]
    selected_feed = st.selectbox("Select Feed", feed_names)
    feed_url = next(feed["url"] for feed in feeds if feed["name"] == selected_feed)

    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a:
        items_count = st.number_input("Articles", min_value=1, max_value=10, value=3, step=1)
    with col_b:
        show_raw = st.checkbox("Show original summaries", value=False)

    params = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "stop": [line.strip() for line in stop_text.splitlines() if line.strip()] or None,
    }

    if st.button("📡 Scan & Analyze"):
        if not api_key:
            st.error("Please enter your API key.")
            st.stop()
        if not model:
            st.error("Please select or enter a model.")
            st.stop()

        with st.spinner("Fetching latest articles..."):
            feed = feedparser.parse(feed_url)

        if hasattr(feed, "status") and feed.status and feed.status >= 400:
            st.error(f"Feed fetch failed: {feed.status} for {feed_url}")
            st.stop()

        if feed.bozo:
            st.warning("The RSS feed may be malformed. Attempting to continue.")

        entries = feed.entries[: int(items_count)]
        if not entries:
            st.info("No articles found in the selected feed.")
            st.stop()

        for idx, entry in enumerate(entries, start=1):
            title = strip_html(entry.get("title", f"Article {idx}"))
            st.subheader(title)

            if show_raw:
                with st.expander("Original Summary", expanded=False):
                    st.write(strip_html(entry.get("summary", "No summary available.")))
                    link = entry.get("link")
                    if link:
                        st.markdown(f"[Read more]({link})")

            user_prompt = build_user_prompt(entry)
            with st.spinner("Generating strategic post..."):
                try:
                    if provider == "Groq":
                        generated = call_groq(api_key, model, st.session_state.system_prompt, user_prompt, params)
                    else:
                        generated = call_nvidia(
                            api_key,
                            base_url,
                            model,
                            st.session_state.system_prompt,
                            user_prompt,
                            params,
                        )
                except Exception as exc:
                    st.error(f"AI request failed: {exc}")
                    continue

            st.markdown("**AI Generated Post**")
            st.text_area(
                "",
                value=generated,
                height=240,
                key=f"generated_{idx}",
            )

            copy_col, _ = st.columns([1, 5])
            with copy_col:
                if st.button("Copy", key=f"copy_{idx}"):
                    st.components.v1.html(
                        f"""
                        <script>
                        navigator.clipboard.writeText({json.dumps(generated)});
                        </script>
                        """,
                        height=0,
                    )
                    st.toast("Copied to clipboard.")

            st.divider()


if __name__ == "__main__":
    main()
