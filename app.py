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

st.set_page_config(page_title="文献伴侣", page_icon="📚")
st.title("📚 文献伴侣智能体")
st.markdown("总结论文摘要 · 提取关键观点 · 生成 APA/MLA 引用")

def detect_language(text):
    try:
        lang = detect(text[:500])
        return 'zh' if lang.startswith('zh') else 'en'
    except:
        return 'en'

def get_text_from_pdf(uploaded_file):
    text = ""
    metadata = {}
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            if pdf.metadata:
                metadata = pdf.metadata
    except:
        try:
            reader = PdfReader(tmp_path)
            if reader.metadata:
                metadata = reader.metadata
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            st.error(f"PDF读取失败: {e}")
    os.unlink(tmp_path)
    return text.strip(), metadata

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

# 页面布局
input_method = st.radio("选择输入方式", ("粘贴文本", "上传PDF文件"))

text = ""
if input_method == "粘贴文本":
    text = st.text_area("请输入论文全文", height=300)
else:
    uploaded = st.file_uploader("上传PDF文件", type="pdf")
    if uploaded:
        with st.spinner("提取PDF文本..."):
            text, meta = get_text_from_pdf(uploaded)
        if text:
            st.success(f"提取成功，共 {len(text)} 字符")
        else:
            st.error("PDF无法提取文字，请检查是否扫描版")

if text:
    lang = detect_language(text)
    st.info(f"检测到语言：{'中文' if lang=='zh' else '英文'}")
    with st.spinner("生成摘要和关键词..."):
        sentences = get_summary(text, lang)
        core = sentences[0] if sentences else ""
        detail = " ".join(sentences[1:4]) if len(sentences) > 1 else ""
        keywords = extract_keywords(text, lang)

    st.markdown("### 📖 引用信息（用于生成APA/MLA）")
    col1, col2 = st.columns(2)
    with col1:
        title = st.text_input("标题")
        authors = st.text_input("作者（多作者用分号;分隔）")
        year = st.text_input("发表年份")
    with col2:
        journal = st.text_input("期刊/会议名称")
        volume = st.text_input("卷号（可选）")
        pages = st.text_input("页码（可选）")

    if title and authors and year and journal:
        apa, mla = format_citations(authors, title, year, journal, volume, pages)
    else:
        apa = mla = "请填写完整的标题、作者、年份、期刊"

    st.markdown("---")
    st.markdown("### 📌 核心观点")
    st.write(core if core else "（未能生成）")

    st.markdown("### 🔑 关键发现/论点")
    for i, kw in enumerate(keywords, 1):
        st.write(f"{i}. {kw}")

    st.markdown("### 📄 详细摘要")
    st.write(detail if detail else "（未能生成）")

    st.markdown("### 📖 参考文献")
    st.markdown(f"**APA**  \n{apa}")
    st.markdown(f"**MLA**  \n{mla}")

    md_content = f"""# 文献伴侣分析报告
生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 核心观点
{core}

## 关键发现/论点
{chr(10).join([f"{i+1}. {kw}" for i,kw in enumerate(keywords)])}

## 详细摘要
{detail}

## 参考文献
**APA**  
{apa}

**MLA**  
{mla}
"""
    st.download_button("📥 下载报告 (Markdown)", md_content, file_name=f"文献伴侣_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md", mime="text/markdown")
