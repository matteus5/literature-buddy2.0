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
st.caption("华师大·学习智能体 | 自动提取元数据 | 总结论文 | 生成引用")

# ---------- 会话状态 ----------
if "history" not in st.session_state:
    st.session_state.history = []
if "step" not in st.session_state:
    st.session_state.step = "menu"
if "paper_text" not in st.session_state:
    st.session_state.paper_text = ""
if "core" not in st.session_state:
    st.session_state.core = ""
if "detail" not in st.session_state:
    st.session_state.detail = ""
if "keywords" not in st.session_state:
    st.session_state.keywords = []
if "auto_meta" not in st.session_state:
    st.session_state.auto_meta = {}          # 自动提取的元数据
if "missing_fields" not in st.session_state:
    st.session_state.missing_fields = []     # 缺失的字段列表
if "current_missing_idx" not in st.session_state:
    st.session_state.current_missing_idx = 0
if "final_meta" not in st.session_state:
    st.session_state.final_meta = {}

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

# ---------- 自动元数据提取 ----------
def extract_title(text):
    """提取标题：通常取第一行非空且长度适中，或包含常见标题模式"""
    lines = text.split('\n')
    for line in lines[:15]:
        line = line.strip()
        if len(line) > 10 and len(line) < 200:
            # 排除常见开头词
            if not re.match(r'^(Abstract|摘要|引言|Introduction|参考文献|References|致谢|Acknowledgement)', line, re.I):
                # 去除数字编号如 "1. " 或 "1.1 "
                line = re.sub(r'^\d+(\.\d+)*\s+', '', line)
                return line
    # 取第一段前100字符
    first_para = text[:200].replace('\n', ' ')
    return first_para[:100]

def extract_authors(text):
    """提取作者：匹配常见模式如 "J. Zhang", "Wei Li", "张三", "李四" 等，多作者用分号连接"""
    # 英文: 大写字母开头，点或空格分隔，可能包含第二作者
    pattern_en = r'([A-Z][a-z]*\.?\s+[A-Z][a-z]+|[A-Z][a-z]+\s+[A-Z][a-z]+|[A-Z]\.\s+[A-Z][a-z]+)'
    # 中文: 两到四个汉字（可能包含空格）
    pattern_zh = r'([\u4e00-\u9fa5]{2,4}(?:\s*[\u4e00-\u9fa5]{2,4})*)'
    matches = re.findall(pattern_en, text[:1500]) + re.findall(pattern_zh, text[:1500])
    if matches:
        # 去重，取前3个作者
        unique = []
        for m in matches:
            if m not in unique:
                unique.append(m)
        return '; '.join(unique[:3])
    return ""

def extract_year(text):
    """提取年份：四位数字，通常19xx或20xx，且在上下文中可能是年份"""
    # 优先匹配 "20xx" 或 "19xx"，且前后有空格或标点
    matches = re.findall(r'\b(19|20)\d{2}\b', text[:2000])
    if matches:
        # 取第一个看起来合理的年份
        for y in matches:
            if 1950 <= int(y) <= 2026:
                return y
    return ""

def extract_journal(text):
    """提取期刊/会议名：常见模式如 "Journal of X", "Proceedings of X", "Conference on X" """
    patterns = [
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Journal(?!\w))',
        r'(Proceedings\s+of\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'(International\s+Conference\s+on\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'([A-Z][a-z]+\s+(?:Transactions|Letters|Magazine|Review))'
    ]
    for pat in patterns:
        m = re.search(pat, text[:2000], re.I)
        if m:
            return m.group(1).strip()
    return ""

def extract_volume_pages(text):
    """提取卷号和页码：常见格式如 "Vol.42, pp.123-130" 或 "42:123-130" """
    # 卷号
    volume = ""
    pages = ""
    vol_match = re.search(r'[Vv]ol(?:ume)?\.?\s*(\d+)', text[:2000])
    if vol_match:
        volume = vol_match.group(1)
    page_match = re.search(r'[Pp]p?\.?\s*(\d+[-–]\d+)', text[:2000])
    if page_match:
        pages = page_match.group(1)
    else:
        page_match = re.search(r'(\d{3,5}[-–]\d{3,5})', text[:2000])
        if page_match:
            pages = page_match.group(1)
    return volume, pages

def auto_extract_metadata(text):
    """自动提取所有元数据"""
    title = extract_title(text)
    authors = extract_authors(text)
    year = extract_year(text)
    journal = extract_journal(text)
    volume, pages = extract_volume_pages(text)
    return {
        "title": title,
        "authors": authors,
        "year": year,
        "journal": journal,
        "volume": volume,
        "pages": pages
    }

# ---------- 摘要和关键词 ----------
def get_summary(text, lang, sentence_count=4):
    try:
        if lang == 'zh':
            # 中文：简单分句 + 基于句子长度的简单摘要（取前几个较长句子）
            sentences = re.split(r'[。！？!?]', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            if len(sentences) > sentence_count:
                # 取前几个
                return sentences[:sentence_count]
            return sentences
        else:
            # 英文：使用 sumy
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            stemmer = Stemmer("english")
            summarizer = LsaSummarizer(stemmer)
            summarizer.stop_words = get_stop_words("english")
            summary = summarizer(parser.document, sentence_count)
            return [str(s) for s in summary]
    except Exception as e:
        # 后备：返回前几个句子
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
        return sentences[:sentence_count] if sentences else [text[:200]]

def extract_keywords(text, lang, num_keywords=3):
    try:
        lan = 'zh' if lang == 'zh' else 'en'
        kw_extractor = yake.KeywordExtractor(lan=lan, top=num_keywords, dedupLim=0.9)
        keywords = kw_extractor.extract_keywords(text)
        return [kw[0] for kw in keywords]
    except:
        # 后备：简单词频
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        if not words:
            return []
        from collections import Counter
        common = Counter(words).most_common(num_keywords)
        return [w for w, c in common]

def analyze_paper(text):
    lang = detect_language(text)
    sentences = get_summary(text, lang)
    core = sentences[0] if sentences else "（无法生成摘要）"
    detail = " ".join(sentences[1:4]) if len(sentences) > 1 else ""
    kw = extract_keywords(text, lang)
    return core, detail, kw, lang

# ---------- 格式化引用 ----------
def format_citations(meta):
    authors = meta.get("authors", "")
    title = meta.get("title", "")
    year = meta.get("year", "")
    journal = meta.get("journal", "")
    volume = meta.get("volume", "")
    pages = meta.get("pages", "")
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

# ---------- 交互流程 ----------
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
        auto_meta = auto_extract_metadata(text)
    st.session_state.paper_text = text
    st.session_state.core = core
    st.session_state.detail = detail
    st.session_state.keywords = kw
    st.session_state.auto_meta = auto_meta
    # 确定缺失的字段
    required_fields = ["title", "authors", "year", "journal"]
    missing = []
    for field in required_fields:
        if not auto_meta.get(field):
            missing.append(field)
    # 卷和页码可选，但如果都没有，也可以询问一个（可选）
    if not auto_meta.get("volume") and not auto_meta.get("pages"):
        # 可选，不强制，跳过
        pass
    st.session_state.missing_fields = missing
    st.session_state.current_missing_idx = 0
    st.session_state.final_meta = auto_meta.copy()
    if missing:
        st.session_state.step = "ask_missing"
        # 构建显示消息
        msg = (f"🔍 语言检测：{'中文' if lang=='zh' else '英文'}\n\n"
               f"✅ 分析完成！\n\n"
               f"📌 核心观点：{core}\n\n"
               f"🔑 关键发现：\n" + "\n".join([f"{i+1}. {k}" for i,k in enumerate(kw)]) + "\n\n"
               f"📄 详细摘要：{detail}\n\n"
               f"📖 自动提取的元数据：\n"
               f"   标题: {auto_meta['title'] or '未提取到'}\n"
               f"   作者: {auto_meta['authors'] or '未提取到'}\n"
               f"   年份: {auto_meta['year'] or '未提取到'}\n"
               f"   期刊: {auto_meta['journal'] or '未提取到'}\n"
               f"   卷号: {auto_meta['volume'] or '未提取到'}\n"
               f"   页码: {auto_meta['pages'] or '未提取到'}\n\n"
               f"以下信息未提取到，请补充：\n")
        return msg
    else:
        # 所有信息齐全，直接生成引用
        apa, mla = format_citations(auto_meta)
        result = (f"🔍 语言检测：{'中文' if lang=='zh' else '英文'}\n\n"
                  f"✅ 分析完成！\n\n"
                  f"📌 核心观点：{core}\n\n"
                  f"🔑 关键发现：\n" + "\n".join([f"{i+1}. {k}" for i,k in enumerate(kw)]) + "\n\n"
                  f"📄 详细摘要：{detail}\n\n"
                  f"📖 参考文献：\n"
                  f"APA: {apa}\n"
                  f"MLA: {mla}\n\n"
                  f"分析全部完成！你可以输入「新对话」重新开始。")
        st.session_state.step = "done"
        return result

def process_pdf_upload(uploaded_file):
    if uploaded_file is None:
        return None
    with st.spinner("正在提取 PDF 文本..."):
        text = extract_text_from_pdf(uploaded_file)
    if not text:
        return "PDF 无法提取文字，请尝试文本模式。"
    return process_text_submit(text)

def ask_next_missing():
    if st.session_state.current_missing_idx < len(st.session_state.missing_fields):
        field = st.session_state.missing_fields[st.session_state.current_missing_idx]
        prompt_map = {
            "title": "📖 请输入论文标题",
            "authors": "✍️ 请输入作者（多个作者用英文分号 ; 分隔）",
            "year": "📅 请输入发表年份",
            "journal": "📚 请输入期刊/会议名称"
        }
        return prompt_map[field]
    else:
        # 所有缺失补充完毕，生成最终引用
        apa, mla = format_citations(st.session_state.final_meta)
        result = (f"📖 参考文献：\n"
                  f"APA: {apa}\n"
                  f"MLA: {mla}\n\n"
                  f"分析全部完成！你可以输入「新对话」重新开始。")
        st.session_state.step = "done"
        return result

# ---------- 渲染历史消息 ----------
for msg in st.session_state.history:
    if msg["role"] == "user":
        st.markdown(f"<div style='display:flex; justify-content:flex-end'><div class='chat-message-user'>🧑‍🎓 {msg['content']}</div></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='display:flex; justify-content:flex-start'><div class='chat-message-assistant'>🤖 {msg['content']}</div></div>", unsafe_allow_html=True)

# ---------- 主逻辑 ----------
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

elif st.session_state.step == "ask_missing":
    # 依次询问缺失字段
    if st.session_state.current_missing_idx < len(st.session_state.missing_fields):
        field = st.session_state.missing_fields[st.session_state.current_missing_idx]
        prompt = {
            "title": "📖 请输入论文标题",
            "authors": "✍️ 请输入作者（分号分隔）",
            "year": "📅 请输入发表年份",
            "journal": "📚 请输入期刊/会议名称"
        }[field]
        # 显示输入框
        user_val = st.text_input(prompt, key=f"missing_{field}")
        col1, col2 = st.columns([1, 5])
        with col1:
            btn = st.button("下一步")
        if btn and user_val:
            val = user_val.strip()
            st.session_state.final_meta[field] = val
            st.session_state.current_missing_idx += 1
            # 重新生成回答消息
            next_prompt = ask_next_missing()
            if st.session_state.step == "done":
                # 所有缺失补充完毕，直接显示结果
                add_message("assistant", next_prompt)
            else:
                # 继续询问下一个字段，不添加消息，只刷新页面
                pass
            st.rerun()
    else:
        # 理论上不会进入这里，但防御
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
