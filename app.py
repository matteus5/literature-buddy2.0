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
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib import colors
import io
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.tmt.v20180321 import tmt_client, models

# ---------- 页面配置 ----------
st.set_page_config(page_title="文献伴侣·双语版", page_icon="📚", layout="centered")
st.markdown("""
<style>
    html, body, .stApp, .stApp * { color: #1e293b !important; }
    .stTextInput input, .stTextArea textarea, .stSelectbox select { color: #1e293b !important; background-color: #ffffff !important; }
    .stButton button { color: #1e293b !important; background-color: #f0f2f6 !important; border: 1px solid #cbd5e1 !important; }
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    [data-testid="stToolbar"] {display: none;}
    .chat-message-user { background-color: #dbeafe; padding: 12px; border-radius: 20px; margin-bottom: 12px; max-width: 80%; align-self: flex-end; color: #1e293b !important; }
    .chat-message-assistant { background-color: #f1f5f9; padding: 12px; border-radius: 20px; margin-bottom: 12px; max-width: 80%; align-self: flex-start; color: #1e293b !important; }
    h1, h2, h3 { color: #0f172a !important; }
</style>
""", unsafe_allow_html=True)

st.title("📚 文献伴侣·双语版")
st.caption("华师大·学习智能体 | 自动提取元数据 | 双语摘要 | 论文对照翻译")

# ---------- 会话状态 ----------
if "history" not in st.session_state:
    st.session_state.history = []
if "step" not in st.session_state:
    st.session_state.step = "menu"
if "paper_text" not in st.session_state:
    st.session_state.paper_text = ""
if "core" not in st.session_state:
    st.session_state.core = ""
if "core_trans" not in st.session_state:
    st.session_state.core_trans = ""
if "detail" not in st.session_state:
    st.session_state.detail = ""
if "detail_trans" not in st.session_state:
    st.session_state.detail_trans = ""
if "keywords" not in st.session_state:
    st.session_state.keywords = []
if "keywords_trans" not in st.session_state:
    st.session_state.keywords_trans = []
if "paper_lang" not in st.session_state:
    st.session_state.paper_lang = "en"
if "auto_meta" not in st.session_state:
    st.session_state.auto_meta = {}
if "missing_fields" not in st.session_state:
    st.session_state.missing_fields = []
if "current_missing_idx" not in st.session_state:
    st.session_state.current_missing_idx = 0
if "final_meta" not in st.session_state:
    st.session_state.final_meta = {}

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

# ---------------- 腾讯云翻译（使用官方SDK，稳定可靠） ----------------
def translate_text(text, src_lang, target_lang):
    """使用腾讯云 SDK 翻译文本，自动处理签名，返回译文"""
    if not text or not text.strip():
        return ""

    # 映射语言代码（腾讯云要求大写）
    src = 'zh' if src_lang == 'zh' else 'en'
    tgt = 'zh' if target_lang == 'en' else 'en'

    # 获取密钥（优先从 Streamlit secrets，其次从环境变量）
    secret_id = None
    secret_key = None
    try:
        secret_id = st.secrets["TENCENT_SECRET_ID"]
        secret_key = st.secrets["TENCENT_SECRET_KEY"]
    except:
        secret_id = os.environ.get("TENCENT_SECRET_ID")
        secret_key = os.environ.get("TENCENT_SECRET_KEY")

    if not secret_id or not secret_key:
        return "[错误] 未找到腾讯云 API 密钥，请在 Streamlit Cloud 的 Secrets 中配置 TENCENT_SECRET_ID 和 TENCENT_SECRET_KEY"

    try:
        cred = credential.Credential(secret_id, secret_key)
        http_profile = HttpProfile()
        http_profile.endpoint = "tmt.tencentcloudapi.com"
        client_profile = ClientProfile()
        client_profile.httpProfile = http_profile
        client = tmt_client.TmtClient(cred, "ap-guangzhou", client_profile)

        req = models.TextTranslateRequest()
        req.SourceText = text
        req.Source = src.upper()
        req.Target = tgt.upper()
        req.ProjectId = 0

        resp = client.TextTranslate(req)
        return resp.TargetText
    except Exception as e:
        st.error(f"翻译接口调用失败: {str(e)}")
        return f"[翻译失败] {text[:50]}..."

# ---------- 自动元数据提取 ----------
def extract_title(text):
    lines = text.split('\n')
    for line in lines[:15]:
        line = line.strip()
        if len(line) > 10 and len(line) < 200:
            if not re.match(r'^(Abstract|摘要|引言|Introduction|参考文献|References|致谢|Acknowledgement)', line, re.I):
                line = re.sub(r'^\d+(\.\d+)*\s+', '', line)
                return line
    first_para = text[:200].replace('\n', ' ')
    return first_para[:100]

def extract_authors(text):
    pattern_en = r'([A-Z][a-z]*\.?\s+[A-Z][a-z]+|[A-Z][a-z]+\s+[A-Z][a-z]+|[A-Z]\.\s+[A-Z][a-z]+)'
    pattern_zh = r'([\u4e00-\u9fa5]{2,4}(?:\s*[\u4e00-\u9fa5]{2,4})*)'
    matches = re.findall(pattern_en, text[:1500]) + re.findall(pattern_zh, text[:1500])
    if matches:
        unique = []
        for m in matches:
            if m not in unique:
                unique.append(m)
        return '; '.join(unique[:3])
    return ""

def extract_year(text):
    matches = re.findall(r'\b(19|20)\d{2}\b', text[:2000])
    for y in matches:
        if 1950 <= int(y) <= 2026:
            return y
    return ""

def extract_journal(text):
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
    return {
        "title": extract_title(text),
        "authors": extract_authors(text),
        "year": extract_year(text),
        "journal": extract_journal(text),
        "volume": extract_volume_pages(text)[0],
        "pages": extract_volume_pages(text)[1]
    }

# ---------- 摘要和关键词 ----------
def get_summary(text, lang, sentence_count=4):
    try:
        if lang == 'zh':
            sentences = re.split(r'[。！？!?]', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            if len(sentences) > sentence_count:
                return sentences[:sentence_count]
            return sentences
        else:
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            stemmer = Stemmer("english")
            summarizer = LsaSummarizer(stemmer)
            summarizer.stop_words = get_stop_words("english")
            summary = summarizer(parser.document, sentence_count)
            return [str(s) for s in summary]
    except Exception as e:
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
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        if not words:
            return []
        from collections import Counter
        common = Counter(words).most_common(num_keywords)
        return [w for w, c in common]

def analyze_paper_bilingual(text):
    lang = detect_language(text)
    st.session_state.paper_lang = lang
    sentences = get_summary(text, lang)
    core = sentences[0] if sentences else "（无法生成摘要）"
    detail = " ".join(sentences[1:4]) if len(sentences) > 1 else ""
    kw = extract_keywords(text, lang)
    # 翻译
    target = "zh" if lang == "en" else "en"
    core_trans = translate_text(core, lang, target)
    detail_trans = translate_text(detail, lang, target)
    kw_trans = [translate_text(k, lang, target) for k in kw]
    return core, core_trans, detail, detail_trans, kw, kw_trans, lang

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

def generate_dual_pdf(original_text, translated_text, title):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            leftMargin=20*mm, rightMargin=20*mm,
                            topMargin=20*mm, bottomMargin=20*mm)
    styles = getSampleStyleSheet()
    def split_paragraphs(text, max_chars=400):
        paras = []
        for p in text.split('\n'):
            p = p.strip()
            if p:
                paras.append(p)
        return paras
    left_paras = split_paragraphs(original_text)
    right_paras = split_paragraphs(translated_text)
    max_rows = max(len(left_paras), len(right_paras))
    while len(left_paras) < max_rows:
        left_paras.append("")
    while len(right_paras) < max_rows:
        right_paras.append("")
    data = [["原文 (Original)", "译文 (Translation)"]]
    for l, r in zip(left_paras, right_paras):
        data.append([Paragraph(l, styles['Normal']), Paragraph(r, styles['Normal'])])
    table = Table(data, colWidths=[doc.width/2.0, doc.width/2.0])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
    ]))
    doc.build([table])
    buffer.seek(0)
    return buffer

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
    with st.spinner("正在分析论文并生成双语摘要..."):
        core, core_trans, detail, detail_trans, kw, kw_trans, lang = analyze_paper_bilingual(text)
        auto_meta = auto_extract_metadata(text)
    st.session_state.paper_text = text
    st.session_state.core = core
    st.session_state.core_trans = core_trans
    st.session_state.detail = detail
    st.session_state.detail_trans = detail_trans
    st.session_state.keywords = kw
    st.session_state.keywords_trans = kw_trans
    st.session_state.auto_meta = auto_meta
    st.session_state.final_meta = auto_meta.copy()
    required_fields = ["title", "authors", "year", "journal"]
    missing = [f for f in required_fields if not auto_meta.get(f)]
    st.session_state.missing_fields = missing
    st.session_state.current_missing_idx = 0
    lang_name = "英文" if lang == "en" else "中文"
    target_name = "中文" if lang == "en" else "英文"
    result = (f"🔍 检测到原文语言：{lang_name}\n\n"
              f"✅ 分析完成！\n\n"
              f"📌 核心观点：\n- {lang_name}：{core}\n- {target_name}：{core_trans}\n\n"
              f"🔑 关键发现：\n" + 
              "\n".join([f"- {lang_name}: {kw[i]}  |  {target_name}: {kw_trans[i]}" for i in range(len(kw))]) + "\n\n"
              f"📄 详细摘要：\n- {lang_name}：{detail}\n- {target_name}：{detail_trans}\n\n"
              f"📖 自动提取的元数据：\n"
              f"   标题: {auto_meta['title'] or '未提取到'}\n"
              f"   作者: {auto_meta['authors'] or '未提取到'}\n"
              f"   年份: {auto_meta['year'] or '未提取到'}\n"
              f"   期刊: {auto_meta['journal'] or '未提取到'}\n"
              f"   卷号: {auto_meta['volume'] or '未提取到'}\n"
              f"   页码: {auto_meta['pages'] or '未提取到'}\n")
    if missing:
        missing_names = {"title":"标题", "authors":"作者", "year":"年份", "journal":"期刊"}
        result += f"\n⚠️ 以下信息未提取到，请补充：\n" + "\n".join([f"- {missing_names[f]}" for f in missing])
        st.session_state.step = "ask_missing"
    else:
        apa, mla = format_citations(auto_meta)
        result += f"\n📖 参考文献：\nAPA: {apa}\nMLA: {mla}\n"
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
        apa, mla = format_citations(st.session_state.final_meta)
        result = (f"📖 参考文献：\nAPA: {apa}\nMLA: {mla}\n\n"
                  f"分析全部完成！你可以使用下方的「生成对照 PDF」按钮将论文全文翻译并导出对照版。"
                  f"\n\n如需重置，输入「新对话」。")
        st.session_state.step = "done"
        return result

# ---------- 界面渲染 ----------
for msg in st.session_state.history:
    if msg["role"] == "user":
        st.markdown(f"<div style='display:flex; justify-content:flex-end'><div class='chat-message-user'>🧑‍🎓 {msg['content']}</div></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='display:flex; justify-content:flex-start'><div class='chat-message-assistant'>🤖 {msg['content']}</div></div>", unsafe_allow_html=True)

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
        paper_text = st.text_area("请粘贴论文全文", height=300)
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
    if st.session_state.current_missing_idx < len(st.session_state.missing_fields):
        field = st.session_state.missing_fields[st.session_state.current_missing_idx]
        prompt = {
            "title": "📖 请输入论文标题",
            "authors": "✍️ 请输入作者（分号分隔）",
            "year": "📅 请输入发表年份",
            "journal": "📚 请输入期刊/会议名称"
        }[field]
        user_val = st.text_input(prompt, key=f"missing_{field}")
        if st.button("下一步") and user_val:
            val = user_val.strip()
            st.session_state.final_meta[field] = val
            st.session_state.current_missing_idx += 1
            next_prompt = ask_next_missing()
            if st.session_state.step == "done":
                add_message("assistant", next_prompt)
            st.rerun()
    else:
        st.session_state.step = "done"
        st.rerun()

elif st.session_state.step == "done":
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🌐 生成双语对照 PDF"):
            if st.session_state.paper_text:
                with st.spinner("正在翻译全文并生成 PDF..."):
                    src_lang = st.session_state.paper_lang
                    tgt_lang = "zh" if src_lang == "en" else "en"
                    full_trans = translate_text(st.session_state.paper_text, src_lang, tgt_lang)
                    pdf_buffer = generate_dual_pdf(st.session_state.paper_text, full_trans, st.session_state.final_meta.get("title", "论文"))
                    st.download_button(
                        label="📥 下载对照 PDF",
                        data=pdf_buffer,
                        file_name=f"bilingual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
            else:
                st.warning("没有论文文本，请先上传或粘贴论文。")
    with col2:
        if st.button("🔄 新对话"):
            for key in list(st.session_state.keys()):
                if key not in ["_streamlit_config", "_is_running_with_streamlit"]:
                    del st.session_state[key]
            st.rerun()
    with st.form(key="reset_form"):
        reset = st.text_input("或者输入「新对话」重置", key="reset_cmd")
        if st.form_submit_button("重置") and reset.strip() == "新对话":
            for key in list(st.session_state.keys()):
                if key not in ["_streamlit_config", "_is_running_with_streamlit"]:
                    del st.session_state[key]
            st.rerun()
