import streamlit as st
import pdfplumber
from PyPDF2 import PdfReader
import re
from datetime import datetime
from langdetect import detect
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import yake
import jieba
import tempfile
import os

# ---------------------------- 页面配置与自定义样式 ----------------------------
st.set_page_config(page_title="文献伴侣 · 对话版", page_icon="📚", layout="centered")

# 强制覆盖所有文字颜色为深色，修复白色字体问题
fix_white_text_style = """
    <style>
        /* 全局文字颜色 */
        html, body, .stApp, .stApp * {
            color: #1e293b !important;
        }
        /* 输入框文字颜色 */
        .stTextInput input, .stTextArea textarea, .stSelectbox select, .stMultiselect span {
            color: #1e293b !important;
            background-color: #ffffff !important;
        }
        /* 按钮文字颜色 */
        .stButton button {
            color: #1e293b !important;
            background-color: #f0f2f6 !important;
            border: 1px solid #cbd5e1 !important;
        }
        .stButton button:hover {
            background-color: #e2e8f0 !important;
        }
        /* 隐藏默认UI元素 */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .stAppHeader {display: none;}
        .stDeployButton {display: none;}
        .stActionButton {display: none;}
        .stStatusWidget {display: none;}
        .viewerBadge_link__qRIco {display: none;}
        [data-testid="stToolbar"] {display: none;}
        [data-testid="stDecoration"] {display: none;}
        [data-testid="stStatusWidget"] {display: none;}
        /* 聊天消息样式 */
        .chat-message-user {
            background-color: #dbeafe;
            padding: 12px;
            border-radius: 20px;
            margin-bottom: 12px;
            max-width: 80%;
            align-self: flex-end;
            color: #1e293b !important;
        }
        .chat-message-assistant {
            background-color: #f1f5f9;
            padding: 12px;
            border-radius: 20px;
            margin-bottom: 12px;
            max-width: 80%;
            align-self: flex-start;
            color: #1e293b !important;
        }
        /* 链接颜色（若有） */
        a {
            color: #2563eb !important;
        }
        /* 标题颜色 */
        h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #0f172a !important;
        }
        /* 文件上传器文字 */
        .stFileUploader label, .stFileUploader span {
            color: #1e293b !important;
        }
    </style>
"""
st.markdown(fix_white_text_style, unsafe_allow_html=True)

st.title("📚 文献伴侣 · 对话版")
st.caption("华师大·学习智能体 | 总结论文 | 提取关键词 | 生成引用")

# ---------------------------- 初始化会话状态 ----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "step" not in st.session_state:
    st.session_state.step = "start"
if "paper_text" not in st.session_state:
    st.session_state.paper_text = ""
if "paper_lang" not in st.session_state:
    st.session_state.paper_lang = "en"
if "summary_core" not in st.session_state:
    st.session_state.summary_core = ""
if "summary_detail" not in st.session_state:
    st.session_state.summary_detail = ""
if "keywords" not in st.session_state:
    st.session_state.keywords = []
if "citation_meta" not in st.session_state:
    st.session_state.citation_meta = {}

# ---------------------------- 辅助函数 ----------------------------
def add_message(role, content):
    st.session_state.chat_history.append({"role": role, "content": content})

def detect_language(text):
    try:
        lang = detect(text[:500])
        return 'zh' if lang.startswith('zh') else 'en'
    except:
        return 'en'

def extract_text_from_pdf(uploaded_file):
    text = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if not text.strip():
            reader = PdfReader(tmp_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        st.error(f"PDF读取错误: {e}")
    os.unlink(tmp_path)
    return text.strip()

def get_summary(text, lang, sentence_count=4):
    try:
        if lang == 'zh':
            sentences = re.split(r'[。！？!?]', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            return sentences[:sentence_count] if sentences else []
        else:
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            stemmer = Stemmer("english")
            summarizer = LsaSummarizer(stemmer)
            summarizer.stop_words = get_stop_words("english")
            summary = summarizer(parser.document, sentence_count)
            return [str(s) for s in summary]
    except:
        return []

def extract_keywords(text, lang, num_keywords=3):
    try:
        lan = 'zh' if lang == 'zh' else 'en'
        kw_extractor = yake.KeywordExtractor(lan=lan, top=num_keywords, dedupLim=0.9)
        keywords = kw_extractor.extract_keywords(text)
        return [kw[0] for kw in keywords]
    except:
        return []

def format_citations(authors, title, year, journal, volume, pages):
    author_list = []
    for item in authors.split(';'):
        name = item.strip()
        if not name:
            continue
        if ',' in name:
            last, first = name.split(',', 1)
            author_list.append((last.strip(), first.strip()))
        else:
            parts = name.split()
            if len(parts) == 1:
                author_list.append((parts[0], ''))
            else:
                author_list.append((parts[-1], ' '.join(parts[:-1])))
    apa_authors = []
    for last, first in author_list:
        if first:
            initials = ''.join([n[0]+'.' for n in first.split() if n])
            apa_authors.append(f"{last}, {initials}")
        else:
            apa_authors.append(last)
    if len(apa_authors) == 1:
        apa_auth_str = apa_authors[0]
    elif len(apa_authors) == 2:
        apa_auth_str = f"{apa_authors[0]} & {apa_authors[1]}"
    else:
        apa_auth_str = ", ".join(apa_authors[:-1]) + ", & " + apa_authors[-1]

    if len(author_list) == 1:
        last, first = author_list[0]
        mla_auth_str = f"{last}, {first}" if first else last
    elif len(author_list) == 2:
        last1, first1 = author_list[0]
        last2, first2 = author_list[1]
        if first1 and first2:
            mla_auth_str = f"{last1}, {first1}, and {first2} {last2}"
        else:
            mla_auth_str = f"{last1} and {last2}"
    else:
        last1, first1 = author_list[0]
        mla_auth_str = f"{last1}, {first1 if first1 else ''} et al."

    apa_cite = f"{apa_auth_str} ({year}). {title}. {journal}"
    if volume:
        apa_cite += f", {volume}"
    if pages:
        apa_cite += f", {pages}"
    apa_cite += "."
    mla_cite = f'{mla_auth_str}. "{title}." {journal}'
    if volume:
        mla_cite += f", vol. {volume}"
    if pages:
        mla_cite += f", pp. {pages}"
    mla_cite += f", {year}."
    return apa_cite, mla_cite

# ---------------------------- 对话逻辑 ----------------------------
def process_user_input(user_input):
    step = st.session_state.step

    if step == "start":
        if user_input == "1":
            st.session_state.step = "await_text"
            return "请直接粘贴论文文本（支持多行），粘贴完成后在下方输入框内单行输入：**END** 并发送。"
        elif user_input == "2":
            st.session_state.step = "await_pdf"
            return "请使用下方的文件上传器上传 PDF 文件（仅支持文字型PDF）。"
        else:
            return "请输入数字 1 或 2：\n1️⃣ 粘贴文本\n2️⃣ 上传 PDF 文件"

    elif step == "await_text":
        if user_input.strip().upper() == "END":
            text = st.session_state.pending_text
            if not text:
                return "文本内容为空。请重新选择模式（1或2）。"
            st.session_state.paper_text = text
            st.session_state.step = "analyzing"
            return analyze_paper(text)
        else:
            if "pending_text" not in st.session_state:
                st.session_state.pending_text = ""
            st.session_state.pending_text += user_input + "\n"
            return f"已接收文本片段（当前共 {len(st.session_state.pending_text)} 字符），继续粘贴，输入 END 结束。"

    elif step == "await_pdf":
        return None

    elif step == "await_metadata":
        meta = st.session_state.citation_meta
        if "meta_field" not in st.session_state:
            st.session_state.meta_field = "title"
        field = st.session_state.meta_field
        meta[field] = user_input
        if field == "title":
            st.session_state.meta_field = "authors"
            return "请输入作者（多个作者用英文分号 ; 分隔，例如：Zhang, Wei; Li, Ming）"
        elif field == "authors":
            st.session_state.meta_field = "year"
            return "请输入发表年份（如 2024）"
        elif field == "year":
            st.session_state.meta_field = "journal"
            return "请输入期刊或会议名称"
        elif field == "journal":
            st.session_state.meta_field = "volume"
            return "请输入卷号（如果没有，直接回复 无）"
        elif field == "volume":
            vol = user_input.strip()
            if vol.lower() in ["无", "none", ""]:
                meta["volume"] = ""
            else:
                meta["volume"] = vol
            st.session_state.meta_field = "pages"
            return "请输入页码（例如 123-130，如果没有，回复 无）"
        elif field == "pages":
            pages = user_input.strip()
            if pages.lower() in ["无", "none", ""]:
                meta["pages"] = ""
            else:
                meta["pages"] = pages
            apa, mla = format_citations(
                meta.get("authors", ""),
                meta.get("title", ""),
                meta.get("year", ""),
                meta.get("journal", ""),
                meta.get("volume", ""),
                meta.get("pages", "")
            )
            result = (
                f"✅ 分析完成！\n\n"
                f"📌 核心观点：\n{st.session_state.summary_core}\n\n"
                f"🔑 关键发现/论点：\n" + "\n".join([f"{i+1}. {kw}" for i, kw in enumerate(st.session_state.keywords)]) + "\n\n"
                f"📄 详细摘要：\n{st.session_state.summary_detail}\n\n"
                f"📖 参考文献：\nAPA: {apa}\nMLA: {mla}\n\n"
                f"你可以继续上传新的论文，或输入「新对话」重置。"
            )
            st.session_state.step = "done"
            return result

    elif step == "done":
        if user_input.strip() == "新对话":
            for key in ["step", "paper_text", "paper_lang", "summary_core", "summary_detail", "keywords", "citation_meta", "pending_text", "meta_field"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.step = "start"
            st.session_state.chat_history = []
            return "✨ 已重置。请选择输入方式：\n1️⃣ 粘贴文本\n2️⃣ 上传 PDF 文件"
        else:
            return "请回复「新对话」开始处理下一篇论文。"

    return "我有点困惑，请重新开始（刷新页面）"

def analyze_paper(text):
    lang = detect_language(text)
    st.session_state.paper_lang = lang
    sentences = get_summary(text, lang)
    core = sentences[0] if sentences else "无法生成摘要"
    detail = " ".join(sentences[1:4]) if len(sentences) > 1 else ""
    keywords = extract_keywords(text, lang)
    st.session_state.summary_core = core
    st.session_state.summary_detail = detail
    st.session_state.keywords = keywords
    st.session_state.step = "await_metadata"
    st.session_state.citation_meta = {}
    st.session_state.meta_field = "title"
    return (
        f"🔍 语言检测：{'中文' if lang=='zh' else '英文'}\n"
        f"摘要与关键词已生成。\n\n"
        f"接下来需要您补充论文的发表信息，以便生成标准引用。\n"
        f"请输入论文标题："
    )

# ---------------------------- 渲染聊天界面 ----------------------------
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"<div style='display: flex; justify-content: flex-end;'><div class='chat-message-user'>🧑‍🎓 {msg['content']}</div></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='display: flex; justify-content: flex-start;'><div class='chat-message-assistant'>🤖 {msg['content']}</div></div>", unsafe_allow_html=True)

if st.session_state.step == "await_pdf":
    uploaded_file = st.file_uploader("📄 上传 PDF 文件", type="pdf", key="pdf_uploader")
    if uploaded_file:
        with st.spinner("正在提取 PDF 文本..."):
            text = extract_text_from_pdf(uploaded_file)
            if text:
                add_message("assistant", f"✅ PDF 已读取，共 {len(text)} 字符。正在分析...")
                response = analyze_paper(text)
                add_message("assistant", response)
                st.rerun()
            else:
                st.error("PDF 无法提取文字，请尝试文本模式。")

with st.container():
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input("", placeholder="在这里输入...", key="user_input", label_visibility="collapsed")
    with col2:
        send_btn = st.button("发送")

if send_btn and user_input.strip():
    add_message("user", user_input)
    assistant_reply = process_user_input(user_input.strip())
    if assistant_reply:
        add_message("assistant", assistant_reply)
    st.rerun()

if len(st.session_state.chat_history) == 0:
    welcome = "你好！我是文献伴侣智能体，你可以粘贴论文文本或上传PDF，我会为你生成摘要、关键词和引用格式。\n\n请选择输入方式：\n1️⃣ 粘贴文本\n2️⃣ 上传 PDF 文件"
    add_message("assistant", welcome)
    st.rerun()
