# ============================================================
# app_chatbot.py – Chatbot Phân tích Phản hồi Sinh viên
# ============================================================
import streamlit as st
import pandas as pd
import json
from datetime import datetime
import re
import os
from typing import List, Dict, Any

# Visualization
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px

# NLP
try:
    from underthesea import sentiment, word_tokenize
except ImportError:
    st.error("Vui lòng cài underthesea: pip install underthesea")
    sentiment = None
    word_tokenize = None

# ============================================================
# CONSTANTS & SESSION STATE
# ============================================================
EMOJI_MAP = {"positive": "😊", "negative": "😟", "neutral": "😐"}

STOPWORDS_PATH = "stopwords_vi.txt"

DEFAULT_STOPWORDS = {
    "và", "của", "là", "các", "cho", "có", "không", "được", "với", "trong",
    "một", "này", "như", "để", "tôi", "bạn", "rất", "nhưng", "cũng", "thì",
    "lại", "nên", "hay", "đang", "sẽ", "đã", "mà", "nếu", "vì", "tại", "khi",
    "thì", "ở", "ra", "vào", "lên", "xuống", "đi", "về", "học", "sinh", "viên"
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================
@st.cache_resource
def load_stopwords(path: str = STOPWORDS_PATH) -> set:
    """Load stopwords từ file hoặc dùng default."""
    stopwords = set(DEFAULT_STOPWORDS)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    word = line.strip().lower()
                    if word:
                        stopwords.add(word)
        except Exception:
            pass
    return stopwords


STOPWORDS = load_stopwords()


def clean_text(text: str) -> str:
    """Xử lý ký tự đặc biệt, emoji-only, quá ngắn."""
    if not text or len(text.strip()) < 2:
        return ""
    
    # Giữ lại chữ cái, số, dấu tiếng Việt và khoảng trắng
    text = re.sub(r'[^a-zA-Z0-9\sÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠ-ỹ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def is_emoji_only(text: str) -> bool:
    emoji_pattern = re.compile(
        "["u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" u"\U0001F680-\U0001F6FF" 
        u"\U0001F1E0-\U0001F1FF" "]+", flags=re.UNICODE)
    cleaned = re.sub(emoji_pattern, '', text).strip()
    return len(cleaned) == 0


def detect_language(text: str) -> str:
    """TODO 9: Detect ngôn ngữ đơn giản (ưu tiên tiếng Việt)."""
    text = text.lower()
    vi_chars = len(re.findall(r'[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]', text))
    if vi_chars > len(text) * 0.05 or any(word in text for word in ["của", "là", "không", "rất", "được"]):
        return "vi"
    return "en"  # fallback


@st.cache_resource
def get_sentiment_model():
    """Cache underthesea model."""
    return sentiment


def analyze_feedback(text: str) -> Dict[str, Any]:
    """Phân tích cảm xúc + từ khóa."""
    if not text or len(text.strip()) < 3:
        return {
            "sentiment": "neutral",
            "keywords": [],
            "confidence": 0.5,
            "message": "Phản hồi quá ngắn hoặc không hợp lệ."
        }

    if is_emoji_only(text):
        return {
            "sentiment": "positive" if any(c in text for c in "😊👍❤️") else "negative",
            "keywords": ["emoji"],
            "confidence": 0.8,
            "message": "Phát hiện emoji-only."
        }

    cleaned = clean_text(text)
    if not cleaned:
        return {"sentiment": "neutral", "keywords": [], "confidence": 0.4, "message": "Không có nội dung sau khi làm sạch."}

    try:
        # Sentiment
        sent_result = get_sentiment_model()(cleaned)
        raw_sent = str(sent_result[0]) if isinstance(sent_result, (list, tuple)) else str(sent_result)
        
        if "POSITIVE" in raw_sent.upper() or "TÍCH CỰC" in raw_sent:
            sentiment_label = "positive"
        elif "NEGATIVE" in raw_sent.upper() or "TIÊU CỰC" in raw_sent:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"

        # Từ khóa
        tokens = word_tokenize(cleaned, format="text").split()
        keywords = [w.lower() for w in tokens if w.lower() not in STOPWORDS and len(w) > 2]
        
        # Confidence giả lập (càng dài + nhiều từ khóa → confidence cao)
        confidence = min(0.95, 0.6 + (len(keywords) * 0.05) + (len(cleaned) / 500))

        return {
            "sentiment": sentiment_label,
            "keywords": keywords[:20],  # giới hạn
            "confidence": round(confidence, 2),
            "raw_sentiment": raw_sent,
            "message": ""
        }
    except Exception as e:
        return {
            "sentiment": "neutral",
            "keywords": [],
            "confidence": 0.3,
            "message": f"Lỗi phân tích: {str(e)}"
        }


def render_analysis(result: Dict) -> str:
    """Tạo markdown đẹp cho chat bubble."""
    emoji = EMOJI_MAP.get(result["sentiment"], "😐")
    conf = result.get("confidence", 0) * 100
    
    md = f"""
**{emoji} Cảm xúc:** {result['sentiment'].upper()}  
**Độ tin cậy:** {conf:.1f}%  
"""
    if result.get("message"):
        md += f"**Ghi chú:** {result['message']}\n\n"
    
    if result["keywords"]:
        kw_str = ", ".join(result["keywords"][:10])
        md += f"**Từ khóa chính:** {kw_str}\n"
    
    return md.strip()


# ============================================================
# FILE HANDLING
# ============================================================
def handle_file_upload() -> List[str]:
    """Upload CSV/Excel và trả về list phản hồi."""
    uploaded = st.sidebar.file_uploader("📁 Upload file phản hồi (CSV/Excel)", type=["csv", "xlsx", "xls"])
    if not uploaded:
        return []
    
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
        
        # Tìm cột chứa phản hồi (ưu tiên 'feedback', 'text', 'phản hồi', cột đầu tiên)
        text_col = None
        for col in ["feedback", "text", "phản_hồi", "phan_hoi", "comment", df.columns[0]]:
            if col in df.columns:
                text_col = col
                break
        
        if text_col:
            feedbacks = df[text_col].dropna().astype(str).tolist()
            st.sidebar.success(f"Đã tải {len(feedbacks)} phản hồi từ file.")
            return feedbacks
        else:
            st.sidebar.error("Không tìm thấy cột phản hồi phù hợp.")
            return []
    except Exception as e:
        st.sidebar.error(f"Lỗi đọc file: {e}")
        return []


def export_history(history: List[Dict]) -> bytes:
    """Export lịch sử ra CSV bytes."""
    if not history:
        return b""
    
    df = pd.DataFrame([
        {
            "timestamp": item["timestamp"],
            "feedback": item["feedback"],
            "sentiment": item["result"]["sentiment"],
            "confidence": item["result"].get("confidence", 0),
            "keywords": ", ".join(item["result"].get("keywords", []))
        }
        for item in history
    ])
    return df.to_csv(index=False).encode("utf-8")


# ============================================================
# VISUALIZATION
# ============================================================
def render_wordcloud(keywords: List[str]):
    """Vẽ Word Cloud."""
    if not keywords:
        st.info("Chưa có từ khóa để vẽ Word Cloud.")
        return
    
    text = " ".join(keywords)
    try:
        wc = WordCloud(
            width=800, height=400,
            background_color="white",
            colormap="viridis",
            stopwords=STOPWORDS,
            max_words=100
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
    except Exception:
        st.warning("Không thể vẽ Word Cloud.")


def render_sentiment_timeline(history: List[Dict]):
    """Biểu đồ xu hướng cảm xúc theo thời gian."""
    if len(history) < 2:
        return
    
    df = pd.DataFrame(history)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    
    sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
    df["score"] = df["result"].apply(lambda x: sentiment_map.get(x.get("sentiment", "neutral"), 0))
    
    daily = df.groupby("date")["score"].mean().reset_index()
    
    fig = px.line(
        daily, x="date", y="score",
        title="📈 Xu hướng cảm xúc theo thời gian",
        labels={"score": "Điểm cảm xúc (-1 → 1)", "date": "Ngày"},
        markers=True
    )
    fig.update_yaxes(range=[-1.1, 1.1])
    st.plotly_chart(fig, use_container_width=True)


def render_sidebar_stats(history: List[Dict]):
    """Thống kê sidebar."""
    st.sidebar.header("📊 Thống kê tổng hợp")
    
    if not history:
        st.sidebar.info("Chưa có dữ liệu phân tích.")
        return
    
    sentiments = [h["result"]["sentiment"] for h in history]
    pos = sentiments.count("positive")
    neg = sentiments.count("negative")
    neu = sentiments.count("neutral")
    
    col1, col2, col3 = st.sidebar.columns(3)
    col1.metric("😊 Tích cực", pos)
    col2.metric("😟 Tiêu cực", neg)
    col3.metric("😐 Trung lập", neu)
    
    # Word Cloud
    all_keywords = []
    for h in history:
        all_keywords.extend(h["result"].get("keywords", []))
    
    if all_keywords:
        st.sidebar.subheader("☁️ Word Cloud")
        render_wordcloud(all_keywords)
    
    # Timeline
    if len(history) > 3:
        st.sidebar.subheader("📅 Xu hướng")
        render_sentiment_timeline(history)
    
    # So sánh (TODO 12 - cơ bản)
    if st.sidebar.checkbox("🔄 Chế độ so sánh 2 nhóm"):
        st.sidebar.info("Tính năng so sánh 2 nhóm (trước/sau) sẽ được mở rộng ở phiên bản sau.")


# ============================================================
# HISTORY MANAGEMENT
# ============================================================
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history" not in st.session_state:
        st.session_state.history = load_history()


def save_history(history: List[Dict], path: str = "history.json"):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def load_history(path: str = "history.json") -> List[Dict]:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def delete_feedback(index: int):
    """Xóa phản hồi theo index."""
    if 0 <= index < len(st.session_state.history):
        del st.session_state.history[index]
        # Cũng xóa message tương ứng (phức tạp một chút)
        if len(st.session_state.messages) > index * 2 + 1:  # user + assistant
            del st.session_state.messages[index * 2 : index * 2 + 2]
        save_history(st.session_state.history)


# ============================================================
# HELP PAGE
# ============================================================
def render_help_page():
    with st.expander("❓ Hướng dẫn sử dụng", expanded=False):
        st.markdown("""
        **Cách sử dụng:**
        - Nhập phản hồi sinh viên vào ô chat (có thể nhập nhiều dòng).
        - Upload file CSV/Excel chứa cột phản hồi.
        - Xem kết quả phân tích cảm xúc + từ khóa.
        - Sidebar hiển thị thống kê, Word Cloud và timeline.
        - Nút **Xóa** bên cạnh mỗi phản hồi để chỉnh sửa lịch sử.
        
        **Chỉ số:**
        - **Cảm xúc**: positive / negative / neutral
        - **Độ tin cậy**: Confidence score (0-1)
        - **Từ khóa**: Các từ quan trọng sau khi loại stopwords
        """)

# ============================================================
# MAIN
# ============================================================
def main():
    st.set_page_config(
        page_title="Phân tích Phản hồi SV",
        page_icon="🤖",
        layout="wide"
    )
    
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        render_sidebar_stats(st.session_state.history)
        render_help_page()
        
        # Upload file
        uploaded_feedbacks = handle_file_upload()
        for fb in uploaded_feedbacks:
            if fb.strip():
                result = analyze_feedback(fb)
                st.session_state.history.append({
                    "timestamp": datetime.now().isoformat(),
                    "feedback": fb,
                    "result": result
                })
                st.session_state.messages.append({"role": "user", "content": fb})
                st.session_state.messages.append({"role": "assistant", "content": render_analysis(result)})
        
        # Export
        if st.session_state.history and st.sidebar.button("📥 Export lịch sử CSV"):
            csv_bytes = export_history(st.session_state.history)
            st.sidebar.download_button(
                label="Tải file CSV",
                data=csv_bytes,
                file_name=f"phan_hoi_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
    
    # Main UI
    st.title("🤖 Chatbot Phân tích Phản hồi Sinh viên")
    st.caption("Hỗ trợ phân tích cảm xúc & từ khóa từ phản hồi của sinh viên")
    
    # Hiển thị lịch sử chat
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Nút xóa cho mỗi cặp user-assistant
            if msg["role"] == "assistant" and (i % 2 == 1) and (i-1)//2 < len(st.session_state.history):
                idx = (i-1)//2
                if st.button("🗑️ Xóa", key=f"del_{idx}"):
                    delete_feedback(idx)
                    st.rerun()
    
    # Chat input
    if prompt := st.chat_input("Nhập phản hồi sinh viên (có thể nhiều dòng)..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        lines = [line.strip() for line in prompt.splitlines() if line.strip()]
        
        for line in lines:
            result = analyze_feedback(line)
            
            analysis_md = render_analysis(result)
            
            with st.chat_message("assistant"):
                st.markdown(analysis_md)
            
            # Lưu history
            st.session_state.history.append({
                "timestamp": datetime.now().isoformat(),
                "feedback": line,
                "result": result
            })
            
            st.session_state.messages.append({"role": "assistant", "content": analysis_md})
        
        save_history(st.session_state.history)
        st.rerun()


if __name__ == "__main__":
    main()
