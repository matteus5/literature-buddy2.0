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

# ---------- 页面配置 ----------
st.set_page_config(page_title="文献伴侣", page_icon="📚", layout="centered")

# 深色文字样式
st.markdown("""
<style>
    html, body, .stApp, .stApp * {
        color: #1e293b !important;
    }
    .stTextInput input, .stTextArea textarea, .stSelectbox select {
        color: #1e293b !important;
        background-color: #ffffff !important;
    }
    .stButton button {
        color: #1e293b !important;
        background-color: #f0f2f6 !important;
        border: 1px solid #cbd5e1 !important;
    }
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    [data-testid="stToolbar"] {display: none;}
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
    h1, h2, h3 {
        color: #0f172a !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("📚 文献伴侣")
st.caption("华师大·学习智能体 | 总结论文 | 提取关键词 | 生成引用")

# ---------- 会话状态 ----------
if "history" not in st.session_state:
    st.session_state.history = []
if "step" not in st.session_state:
    st.session_state.step = "menu"          # menu, wait_text, wait_pdf, wait_metadata, done
if "paper_text" not in st.session_state:
    st.session_state.paper_text = ""
if "core" not in st.session_state:
    st.session_state.core = ""
if "detail" not in st.session_state:
    st.session_state.detail = ""
if "keywords" not in st.session_state:
    st.session_state.keywords = []
if "meta" not in st.session_state:
    st.session_state.meta = {}
if "meta_index" not in st.session_state:    # 0~5 对应字段顺序
    st.session_state.meta_index = 0

# ---------- 辅助函数 ----------
def add_message(role, content):
    st.session_state.history.append({"role": role, "content": content})

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
    # 简化版 APA/MLA
    apa = f"{authors} ({year}). {title}. {journal}"
    if volume:
        apa += f", {volume}"
    if pages:
        apa += f", {pages}"
    apa += "."
    mla = f'{authors}. "{title}." {journal}'
    if volume:
        mla += f", vol. {volume}"
    if pages:
        mla += f", pp. {pages}"
    mla += f", {year}."
    return apa, mla

def analyze_paper(text):
    lang = detect_language(text)
    sentences = get_summary(text, lang)
    core = sentences[0] if sentences else "（无法生成摘要）"
    detail = " ".join(sentences[1:4]) if len(sentences) > 1 else ""
    kw = extract_keywords(text, lang)
    return core, detail, kw, lang

# ---------- 响应逻辑 ----------
def process_menu_choice(choice):
    if choice == "1":
        st.session_state.step = "wait_text"
        return "请在下方的文本框中输入论文全文，然后点击「提交文本」。"
    elif choice == "2":
        st.session_state.step = "wait_pdf"
        return "请上传 PDF 文件（仅支持文字型 PDF）。"
    else:
        return "请输入 1 或 2。"

def process_text_submit(text):
    if not text.strip():
        return "文本不能为空。请重新选择模式。"
    with st.spinner("正在分析论文..."):
        core, detail, kw, lang = analyze_paper(text)
    st.session_state.core = core
    st.session_state.detail = detail
    st.session_state.keywords = kw
    st.session_state.paper_text = text
    st.session_state.step = "wait_metadata"
    st.session_state.meta = {}
    st.session_state.meta_index = 0
    return (f"🔍 语言检测：{'中文' if lang=='zh' else '英文'}\n\n"
            f"✅ 分析完成！\n\n"
            f"📌 核心观点：{core}\n\n"
            f"🔑 关键发现：\n" + "\n".join([f"{i+1}. {k}" for i,k in enumerate(kw)]) + "\n\n"
            f"📄 详细摘要：{detail}\n\n"
            f"接下来请补充论文的发表信息，以便生成标准引用。\n")

def process_pdf_upload(uploaded_file):
    if uploaded_file is None:
        return None
    with st.spinner("正在提取 PDF 文本..."):
        text = extract_text_from_pdf(uploaded_file)
    if not text:
        return "PDF 无法提取文字，请尝试文本模式。"
    with st.spinner("正在分析论文..."):
        core, detail, kw, lang = analyze_paper(text)
    st.session_state.core = core
    st.session_state.detail = detail
    st.session_state.keywords = kw
    st.session_state.paper_text = text
    st.session_state.step = "wait_metadata"
    st.session_state.meta = {}
    st.session_state.meta_index = 0
    return (f"🔍 语言检测：{'中文' if lang=='zh' else '英文'}\n\n"
            f"✅ 分析完成！\n\n"
            f"📌 核心观点：{core}\n\n"
            f"🔑 关键发现：\n" + "\n".join([f"{i+1}. {k}" for i,k in enumerate(kw)]) + "\n\n"
            f"📄 详细摘要：{detail}\n\n")

# ---------- 渲染历史消息 ----------
for msg in st.session_state.history:
    if msg["role"] == "user":
        st.markdown(f"<div style='display:flex; justify-content:flex-end'><div class='chat-message-user'>🧑‍🎓 {msg['content']}</div></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='display:flex; justify-content:flex-start'><div class='chat-message-assistant'>🤖 {msg['content']}</div></div>", unsafe_allow_html=True)

# ---------- 根据当前 step 显示不同的 UI ----------
if st.session_state.step == "menu":
    with st.form(key="menu_form"):
        choice = st.text_input("请输入数字选择：\n1️⃣ 粘贴文本\n2️⃣ 上传 PDF", key="menu_choice")
        submitted = st.form_submit_button("确定")
        if submitted and choice:
            response = process_menu_choice(choice.strip())
            add_message("user", choice)
            add_message("assistant", response)
            st.rerun()

elif st.session_state.step == "wait_text":
    with st.form(key="text_form"):
        paper_text = st.text_area("请粘贴论文全文", height=300, key="paper_text_input")
        submitted = st.form_submit_button("提交文本")
        if submitted:
            response = process_text_submit(paper_text)
            add_message("user", "[提交了论文文本]")
            add_message("assistant", response)
            st.rerun()

elif st.session_state.step == "wait_pdf":
    uploaded = st.file_uploader("上传 PDF 文件", type="pdf", key="pdf_upload")
    if uploaded is not None:
        response = process_pdf_upload(uploaded)
        if response:
            add_message("user", "[上传了PDF文件]")
            add_message("assistant", response)
            st.rerun()

elif st.session_state.step == "wait_metadata":
    # 定义字段顺序
    fields = ["title", "authors", "year", "journal", "volume", "pages"]
    prompts = {
        "title": "📖 请输入论文标题",
        "authors": "✍️ 请输入作者（多个作者用英文分号 ; 分隔）",
        "year": "📅 请输入发表年份",
        "journal": "📚 请输入期刊/会议名称",
        "volume": "🔢 请输入卷号（如果没有，输入 无）",
        "pages": "📄 请输入页码（如果没有，输入 无）"
    }
    idx = st.session_state.meta_index
    if idx < len(fields):
        field = fields[idx]
        # 直接显示输入框，不用 form，使用 on_change 回调
        prompt = prompts[field]
        user_val = st.text_input(prompt, key=f"meta_{field}")
        col1, col2 = st.columns([1, 5])
        with col1:
            btn = st.button("下一步")
        if btn and user_val:
            # 保存
            value = user_val.strip()
            if field == "volume" and value.lower() in ["无", "none", ""]:
                value = ""
            if field == "pages" and value.lower() in ["无", "none", ""]:
                value = ""
            st.session_state.meta[field] = value
            # 移动到下一个字段
            next_idx = idx + 1
            if next_idx == len(fields):
                # 所有字段收集完毕，生成引用
                apa, mla = format_citations(
                    st.session_state.meta.get("authors", ""),
                    st.session_state.meta.get("title", ""),
                    st.session_state.meta.get("year", ""),
                    st.session_state.meta.get("journal", ""),
                    st.session_state.meta.get("volume", ""),
                    st.session_state.meta.get("pages", "")
                )
                result = (
                    f"📖 参考文献：\n"
                    f"APA: {apa}\n"
                    f"MLA: {mla}\n\n"
                    f"分析全部完成！你可以输入「新对话」重新开始，或刷新页面。"
                )
                add_message("assistant", result)
                st.session_state.step = "done"
                st.rerun()
            else:
                st.session_state.meta_index = next_idx
                # 不添加用户消息，只更新状态，刷新页面后显示下一个输入框
                st.rerun()
    else:
        # 防御
        st.session_state.step = "done"
        st.rerun()

elif st.session_state.step == "done":
    with st.form(key="reset_form"):
        reset = st.text_input("输入「新对话」重置", key="reset_cmd")
        submitted = st.form_submit_button("重置")
        if submitted and reset.strip() == "新对话":
            for key in list(st.session_state.keys()):
                if key not in ["_streamlit_config", "_is_running_with_streamlit"]:
                    del st.session_state[key]
            st.rerun()
        elif submitted:
            st.warning("请输入「新对话」")
