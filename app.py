import os
import re
from typing import List, Optional
import json
import pandas as pd
import streamlit as st
from googleapiclient.discovery import build
from langchain import prompts
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# ---------------------- Helper Functions ----------------------

def extract_video_id(url: str) -> Optional[str]:
    if not url:
        return None
    watch_match = re.search(r"[?&]v=([\w-]{11})", url)
    if watch_match:
        return watch_match.group(1)
    short_match = re.search(r"youtu\.be/([\w-]{11})", url)
    if short_match:
        return short_match.group(1)
    shorts_match = re.search(r"/shorts/([\w-]{11})", url)
    if shorts_match:
        return shorts_match.group(1)
    embed_match = re.search(r"/embed/([\w-]{11})", url)
    if embed_match:
        return embed_match.group(1)
    return None


def fetch_top_comments(video_id: str, api_key: str, max_comments: int) -> List[str]:
    youtube = build("youtube", "v3", developerKey=api_key)
    comments: List[str] = []
    next_page_token: Optional[str] = None

    while len(comments) < max_comments:
        remaining = max_comments - len(comments)
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, remaining),
            order="relevance",
            textFormat="plainText",
            pageToken=next_page_token,
        )
        response = request.execute()

        for item in response.get("items", []):
            snippet = item.get("snippet", {})
            top_comment = snippet.get("topLevelComment", {}).get("snippet", {})
            text = top_comment.get("textOriginal")
            if text:
                comments.append(text)
                if len(comments) >= max_comments:
                    break

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return comments


def sanitize_comment(text: str, max_len: int = 600) -> str:
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        text = text[:max_len] + "…"
    return text


def build_chain():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        model_kwargs={"response_mime_type": "application/json"},
    )

    template = (
        "You are analyzing YouTube comments. Only use the comments provided below.\n"
        "If a sentiment/category is not present, return \"none\" (or 0% if you choose percentages).\n"
        "Respond with ONLY strict JSON, no extra text, no code fences.\n\n"
        "Schema (use exactly these keys):\n"
        "{{\n"
        "  \"comment_summary\": \"...\",\n"
        "  \"sentiments\": {{\n"
        "    \"happy\": \"...\",\n"
        "    \"sad\": \"...\",\n"
        "    \"excited\": \"...\",\n"
        "    \"sarcastic\": \"...\",\n"
        "    \"comedic\": \"...\",\n"
        "    \"loving\": \"...\",\n"
        "    \"angry\": \"...\",\n"
        "    \"disappointed\": \"...\",\n"
        "    \"inspirational\": \"...\",\n"
        "    \"other\": \"...\"\n"
        "  }}\n"
        "}}\n\n"
        "Top {n} comments to analyze (bullet list):\n{comments}\n"
    )

    prompt = prompts.PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain

# ---------------------- Main App ----------------------

def main() -> None:
    st.set_page_config(page_title="YouTube Comment Analyzer", layout="centered")
    st.title("YouTube Comment Analyzer 🎥💬")

    url = st.text_input("YouTube video URL")
    n = st.slider("Number of top comments to analyze", min_value=50, max_value=200, value=100, step=10)

    if st.button("Analyze"):
        video_id = extract_video_id(url)
        if not video_id:
            st.error("❌ Could not extract a valid YouTube video ID from the URL.")
            return

        yt_api_key = os.environ.get("YOUTUBE_API_KEY") or st.secrets.get("YOUTUBE_API_KEY")
        if not yt_api_key:
            st.error("❌ YouTube API key not found. Set 'YOUTUBE_API_KEY' in your .env or Streamlit secrets.")
            return

        if not os.environ.get("GOOGLE_API_KEY") and not st.secrets.get("GOOGLE_API_KEY"):
            st.error("❌ Google AI API key not found. Set 'GOOGLE_API_KEY' in your .env or Streamlit secrets.")
            return

        with st.spinner("📥 Fetching comments..."):
            try:
                comments = fetch_top_comments(video_id, yt_api_key, n)
            except Exception as e:
                st.error(f"Failed to fetch comments: {e}")
                return

        if not comments:
            st.warning("⚠ No comments found for this video.")
            return

        bullets = [f"- {sanitize_comment(c)}" for c in comments]
        comments_block = "\n".join(bullets)

        with st.spinner("🤖 Analyzing with Gemini 2.5 Flash..."):
            try:
                chain = build_chain()
                result_str = chain.invoke({"comments": comments_block, "n": len(comments)})

                start_index = result_str.find('{')
                end_index = result_str.rfind('}')
                if start_index != -1 and end_index != -1 and end_index > start_index:
                    json_str = result_str[start_index : end_index + 1]
                    result_json = json.loads(json_str)
                else:
                    raise ValueError("Could not find a valid JSON object in the model's output.")

            except (ValueError, json.JSONDecodeError) as e:
                st.error(f"❌ Failed to parse JSON from model. Error: {e}")
                st.subheader("Raw output from model:")
                st.text(result_str)
                return
            except Exception as e:
                st.error(f"❌ Model invocation failed: {e}")
                return

        # ---------- Display Results Nicely ----------
        st.subheader("Analysis Results")

        st.markdown("### 📝 Comment Summary")
        st.info(result_json.get("comment_summary", "No summary available."))

        st.markdown("### 🎭 Sentiment Breakdown")
        sentiments = result_json.get("sentiments", {})

        if sentiments:
            emoji_map = {
                "happy": "😊",
                "sad": "😢",
                "excited": "🤩",
                "sarcastic": "🙃",
                "comedic": "😂",
                "loving": "❤️",
                "angry": "😡",
                "disappointed": "😞",
                "inspirational": "🌟",
                "other": "🔹"
            }

            sentiment_rows = []
            for key, value in sentiments.items():
                sentiment_rows.append({
                    "Sentiment": f"{emoji_map.get(key, '')} {key.capitalize()}",
                    "Detected": value
                })

            df = pd.DataFrame(sentiment_rows)
            st.table(df)

            # Optional progress bars if percentages
            if all(str(v).endswith("%") for v in sentiments.values()):
                st.markdown("### 📊 Sentiment Percentage Chart")
                for k, v in sentiments.items():
                    try:
                        percent = float(v.strip('%'))
                        st.progress(int(percent))
                        st.caption(f"{emoji_map.get(k, '')} {k.capitalize()}: {percent}%")
                    except:
                        pass
        else:
            st.warning("⚠ No sentiments detected.")

        with st.expander("🔍 Raw JSON Output"):
            st.json(result_json)

        with st.expander("📌 Details"):
            st.write("Video ID:", video_id)
            st.write("Comments analyzed:", len(comments))


if __name__ == "__main__":
    main()
