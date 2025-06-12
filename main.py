__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import time
import functools
import uuid
import logging
import requests
import zipfile

import streamlit as st
import streamlit.components.v1 as components

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.document_transformers import LongContextReorder
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI

# 로그 레벨 감소
logging.basicConfig(level=logging.WARNING)

# ——— 🔧 벡터 DB 다운로드 함수 ———
@st.cache_resource
def download_and_extract_databases(verbose=True):
    """허깅페이스에서 벡터 DB 다운로드"""
    urls = {
        "chroma_db_law_real_final": "https://huggingface.co/datasets/sujeonggg/chroma_db_law_real_final/resolve/main/chroma_db_law_real_final.zip",
        "ja_chroma_db": "https://huggingface.co/datasets/sujeonggg/chroma_db_law_real_final/resolve/main/ja_chroma_db.zip",
    }

    def download_and_unzip(url, extract_to):
        os.makedirs(extract_to, exist_ok=True)
        zip_path = os.path.join(extract_to, "temp.zip")

        # 이미 존재하는지 확인
        if os.path.exists(os.path.join(extract_to, "chroma.sqlite3")) or \
           any(os.path.exists(os.path.join(extract_to, f)) for f in ["index", "chroma", "data"]):
            if verbose:
                print(f"✅ Already exists: {extract_to}")
            return True

        try:
            if verbose:
                print(f"📦 Downloading from {url}...")
            r = requests.get(url, stream=True)
            r.raise_for_status()
            
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            if verbose:
                print(f"🧩 Unzipping to {extract_to}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)

            os.remove(zip_path)
            return True
        except Exception as e:
            if verbose:
                print(f"❌ Failed to download {url}: {e}")
            return False

    success = True
    for name, url in urls.items():
        if not download_and_unzip(url, name):
            success = False

    return success

# ——— 🔧 임베딩 모델 및 DB 초기화 ———
@st.cache_resource
def initialize_embeddings_and_databases():
    """임베딩 모델과 벡터 DB 초기화"""
    try:
        # 1. 벡터 DB 다운로드
        print("📥 벡터 DB 다운로드 중...")
        download_success = download_and_extract_databases(verbose=False)
        if not download_success:
            return None, None, None, False
        
        # 2. 임베딩 모델 초기화
        print("🔄 임베딩 모델 로딩 중...")
        embedding_model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
        print("✅ 임베딩 모델 로딩 완료")
        
        # 3. Chroma DB 연결
        legal_db = None
        news_db = None
        
        if os.path.exists("chroma_db_law_real_final"):
            try:
                legal_db = Chroma(
                    persist_directory="chroma_db_law_real_final",
                    embedding_function=embedding_model
                )
                print("✅ 법률 DB 연결 완료")
            except Exception as e:
                print(f"⚠️ 법률 DB 연결 실패: {e}")
        
        if os.path.exists("ja_chroma_db"):
            try:
                news_db = Chroma(
                    persist_directory="ja_chroma_db",
                    embedding_function=embedding_model
                )
                print("✅ 뉴스 DB 연결 완료")
            except Exception as e:
                print(f"⚠️ 뉴스 DB 연결 실패: {e}")
        
        return embedding_model, legal_db, news_db, True
        
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        return None, None, None, False

# ——— 커스텀 CSS 스타일 ———
def load_custom_css():
    st.markdown("""
    <style>
    /* 전체 배경 */
    .stApp {
        background: linear-gradient(135deg, #f5f3ff 0%, #faf9ff 50%, #fffbeb 100%);
    }
    
    /* 메인 컨테이너 */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* 헤더 스타일 */
    .header-container {
        background: linear-gradient(135deg, #8b5cf6 0%, #a78bfa 50%, #c4b5fd 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(139, 92, 246, 0.3);
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .header-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    .header-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        position: relative;
        z-index: 1;
    }
    
    .header-subtitle {
        color: #fef3c7;
        font-size: 1.2rem;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    
    .highlight {
        color: #fbbf24;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    /* 채팅 컨테이너 */
    .chat-container {
        background: transparent;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* 사용자 메시지 */
    .user-message {
        display: flex;
        justify-content: flex-end;
        margin: 1rem 0;
        animation: slideInRight 0.3s ease-out;
    }
    
    .user-bubble {
        max-width: 75%;
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 8px 20px;
        border: 2px solid #f59e0b;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.2);
        position: relative;
    }
    
    .user-bubble::before {
        content: '👤';
        position: absolute;
        right: -0.5rem;
        top: -0.5rem;
        background: #f59e0b;
        border-radius: 50%;
        width: 2rem;
        height: 2rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
    }
    
    /* AI 메시지 */
    .ai-message {
        display: flex;
        justify-content: flex-start;
        margin: 1rem 0;
        animation: slideInLeft 0.3s ease-out;
    }
    
    .ai-bubble {
        max-width: 75%;
        background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%);
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 8px;
        border: 2px solid #8b5cf6;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.2);
        position: relative;
    }
    
    .ai-bubble::before {
        content: '🤖';
        position: absolute;
        left: -0.5rem;
        top: -0.5rem;
        background: #8b5cf6;
        border-radius: 50%;
        width: 2rem;
        height: 2rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
    }
    
    @keyframes slideInRight {
        from { transform: translateX(50px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideInLeft {
        from { transform: translateX(-50px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    /* 버튼 스타일 */
    .stButton > button {
        background: linear-gradient(135deg, #8b5cf6 0%, #a78bfa 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #7c3aed 0%, #8b5cf6 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.4);
    }
    
    /* 입력 필드 스타일 */
    .stTextInput > div > div > input {
        border-radius: 15px;
        border: 2px solid #c4b5fd;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #8b5cf6;
        box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.1);
    }
    
    /* 사이드바 카드 스타일 */
    .sidebar-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    /* 제목 스타일 개선 */
    h1, h2, h3 {
        color: #6b21a8;
        font-weight: 700;
    }
    
    /* 구분선 스타일 */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, #c4b5fd 50%, transparent 100%);
        margin: 2rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ——— 법률 쿼리 전처리 클래스 ———
class LegalQueryPreprocessor:
    """일상어를 법률 용어로 변환하는 전처리기"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            max_tokens=200,
        )
        
        self._query_cache = {}
        
        self.term_mapping = {
            "집주인": "임대인", "세입자": "임차인", "전세금": "임대차보증금",
            "보증금": "임대차보증금", "월세": "차임", "방세": "차임",
            "계약서": "임대차계약서", "집 나가라": "명도청구", "쫓겨나다": "명도",
            "돈 안줘": "채무불이행", "돈 못받아": "보증금반환청구", "사기": "사기죄",
            "속았다": "기망행위", "깡통전세": "전세사기", "이중계약": "중복임대",
            "고소": "형사고발", "고발": "형사고발", "소송": "민사소송",
            "재판": "소송", "변호사": "법무사", "상담": "법률상담",
            "해결": "분쟁해결", "보상": "손해배상", "배상": "손해배상",
            "계약": "법률행위", "약속": "계약", "위반": "채무불이행", "어기다": "위반하다"
        }
    
    def _apply_rule_based_conversion(self, query: str) -> str:
        converted_query = query
        for common_term, legal_term in self.term_mapping.items():
            if common_term in converted_query:
                converted_query = converted_query.replace(common_term, legal_term)
        return converted_query
    
    def _is_already_legal_query(self, query: str) -> bool:
        legal_indicators = [
            "임대인", "임차인", "임대차", "명도", "채무불이행", 
            "손해배상", "민사소송", "형사고발", "보증금반환",
            "법률", "판례", "법령", "소송", "계약서"
        ]
        return any(term in query for term in legal_indicators)
    
    @functools.lru_cache(maxsize=100)
    def _gpt_convert_to_legal_terms(self, user_query: str) -> str:
        try:
            prompt = f"""다음 일상어 질문을 법률 검색에 적합한 전문 용어로 변환해주세요.
            원래 질문: {user_query}
            변환 규칙:
            1. 일상어를 정확한 법률 용어로 바꾸기
            2. 핵심 법적 쟁점을 부각시키기
            3. 검색에 도움이 되는 관련 법률 키워드 추가
            4. 원래 의미는 유지하면서 더 정확하고 전문적으로 표현
            변환된 검색 쿼리:"""

            messages = [{"role": "user", "content": prompt}]
            response = self.llm.invoke(messages)
            
            converted = response.content.strip()
            if "변환된 검색 쿼리:" in converted:
                converted = converted.split("변환된 검색 쿼리:")[-1].strip()
            
            return converted
            
        except Exception as e:
            print(f"⚠️ GPT 변환 실패, 룰베이스 변환 사용: {e}")
            return self._apply_rule_based_conversion(user_query)
    
    def convert_query(self, user_query: str) -> tuple[str, str]:
        try:
            if self._is_already_legal_query(user_query):
                return user_query, "no_conversion"
            
            if user_query in self._query_cache:
                return self._query_cache[user_query], "cached"
            
            rule_converted = self._apply_rule_based_conversion(user_query)
            
            if len(rule_converted) != len(user_query) or rule_converted != user_query:
                self._query_cache[user_query] = rule_converted
                return rule_converted, "rule_based"
            
            print("🔄 정교한 법률 용어 변환 중...")
            gpt_converted = self._gpt_convert_to_legal_terms(user_query)
            
            self._query_cache[user_query] = gpt_converted
            return gpt_converted, "gpt_converted"
            
        except Exception as e:
            print(f"⚠️ 쿼리 변환 오류: {e}")
            return user_query, "error"

# ——— RAG 시스템 ———
class OptimizedConditionalRAGSystem:
    def __init__(self, legal_db, news_db):
        print("🚀 RAG 시스템 초기화 중...")
        
        # 데이터베이스 연결
        self.legal_db = legal_db
        self.news_db = news_db
        
        # 쿼리 전처리기 초기화
        self.query_preprocessor = LegalQueryPreprocessor()
        print("✅ 법률 용어 전처리기 준비 완료")
        
        # 리트리버 초기화
        if self.legal_db:
            self.legal_vector_retriever = self.legal_db.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": 5}
            )
        else:
            self.legal_vector_retriever = None
            
        if self.news_db:
            self.news_vector_retriever = self.news_db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )
        else:
            self.news_vector_retriever = None
    
    def search_legal_db(self, query):
        """법률 DB 검색"""
        if self.legal_vector_retriever is None:
            return [], 0.0
        
        try:
            legal_docs = self.legal_vector_retriever.invoke(query)
            print(f"📄 법률 검색 결과: {len(legal_docs)}개 문서")
            return legal_docs, 0.8
        except Exception as e:
            print(f"❌ 법률 DB 검색 오류: {e}")
            return [], 0.0
    
    def search_news_db(self, query):
        """뉴스 DB 검색"""
        if self.news_vector_retriever is None:
            return [], 0.0
        
        try:
            news_docs = self.news_vector_retriever.invoke(query)
            print(f"📰 뉴스 검색 결과: {len(news_docs)}개")
            return news_docs, 0.7
        except Exception as e:
            print(f"❌ 뉴스 DB 검색 오류: {e}")
            return [], 0.0
    
    def conditional_retrieve(self, original_query):
        """조건부 검색"""
        try:
            print(f"🔍 검색 쿼리: {original_query}")
            
            # 쿼리 전처리
            converted_query, conversion_method = self.query_preprocessor.convert_query(original_query)
            
            if conversion_method != "no_conversion":
                print(f"🔄 변환된 쿼리: {converted_query}")
                search_query = converted_query
            else:
                search_query = original_query
            
            # 법률 DB 검색
            legal_docs, legal_score = self.search_legal_db(search_query)
            
            # 뉴스 DB 검색
            news_docs, news_score = self.search_news_db(search_query)
            
            # 결과 결합
            combined_docs = []
            if legal_docs:
                combined_docs.extend(legal_docs[:8])
            if news_docs:
                combined_docs.extend(news_docs[:3])
            
            search_type = "legal_and_news" if (legal_docs and news_docs) else ("legal_only" if legal_docs else "news_only")
            
            print(f"🎯 최종 결과: {len(combined_docs)}개 문서 ({search_type})")
            return combined_docs, search_type
                
        except Exception as e:
            print(f"❌ 검색 오류: {e}")
            return [], "error"

# 4. 최적화된 문서 포맷팅 (기존과 동일)
def format_docs_optimized(docs, search_type):
    """최적화된 문서 포맷팅 - 출처별 명확한 구분"""
    if not docs:
        return "관련 자료를 찾을 수 없습니다."
    
    formatted_docs = []
    news_count = 0
    precedent_count = 0
    interpretation_count = 0
    qa_count = 0
    
    for i, doc in enumerate(docs):
        try:
            meta = doc.metadata if doc.metadata else {}
            content = str(doc.page_content)[:1000] if doc.page_content else ""
            
            is_news = ('url' in meta and 'title' in meta) or ('date' in meta and 'title' in meta)
            
            if is_news:
                news_count += 1
                title = str(meta.get("title", "제목없음"))[:80]
                date = str(meta.get("date", "날짜미상"))
                source = str(meta.get("source", "뉴스"))
                
                formatted = f"[뉴스-{news_count}] 📰 뉴스\n"
                formatted += f"제목: {title}\n"
                formatted += f"출처: {source} | 날짜: {date}\n"
                formatted += f"내용: {content}...\n"
                
            else:
                doc_type = str(meta.get("doc_type", "")).lower()
                
                if any(keyword in doc_type for keyword in ["판례", "판결", "대법원", "고등법원", "지방법원"]) or \
                   any(key in meta for key in ["판결요지", "판시사항", "case_id", "court"]):
                    
                    case_id = str(meta.get("case_id", ""))
                    if case_id and case_id.strip() != "":
                        formatted = f"[판례-{case_id}] 🏛️ 판례\n"
                    else:
                        precedent_count += 1
                        formatted = f"[판례-{precedent_count}] 🏛️ 판례\n"
                    
                    formatted += f"내용: {content}...\n"
                    
                elif any(keyword in doc_type for keyword in ["법령해석", "해석례", "유권해석", "행정해석"]) or \
                     any(key in meta for key in ["해석내용", "법령명", "interpretation_id"]):
                    
                    interpretation_id = str(meta.get("interpretation_id", ""))
                    if interpretation_id and interpretation_id.strip() != "":
                        formatted = f"[법령해석례-{interpretation_id}] ⚖️ 법령해석례\n"
                    else:
                        interpretation_count += 1
                        formatted = f"[법령해석례-{interpretation_count}] ⚖️ 법령해석례\n"
                    
                    formatted += f"내용: {content}...\n"
                    
                elif any(keyword in doc_type for keyword in ["백문백답", "생활법령", "qa", "질의응답", "faq"]) or \
                     any(key in meta for key in ["질문", "답변", "question", "answer", "qa_id"]):
                    
                    qa_id = str(meta.get("qa_id", ""))
                    if qa_id and qa_id.strip() != "":
                        formatted = f"[백문백답-{qa_id}] 💡 생활법령 Q&A\n"
                    else:
                        qa_count += 1
                        formatted = f"[백문백답-{qa_count}] 💡 생활법령 Q&A\n"
                    
                    formatted += f"내용: {content}...\n"
                    
                else:
                    precedent_count += 1
                    source = str(meta.get("doc_type", "법률자료"))
                    formatted = f"[법률-{precedent_count}] 📋 {source}\n"
                    formatted += f"내용: {content}...\n"
            
            formatted_docs.append(formatted)
            
        except Exception as e:
            print(f"⚠️ 문서 포맷팅 오류: {e}")
            try:
                content = str(doc.page_content)[:1000] if doc.page_content else "내용 없음"
                formatted_docs.append(f"[문서-{i+1}] {content}...")
            except:
                continue
    
    # 결과 조합 - 유형별 개수 표시
    header_parts = []
    if precedent_count > 0:
        header_parts.append(f"판례 {precedent_count}개")
    if interpretation_count > 0:
        header_parts.append(f"법령해석례 {interpretation_count}개")
    if qa_count > 0:
        header_parts.append(f"생활법령Q&A {qa_count}개")
    if news_count > 0:
        header_parts.append(f"뉴스 {news_count}개")
    
    header = f"📋 검색결과: {', '.join(header_parts)}\n"
    header += "="*60 + "\n"
    header += "⚠️ AI가 아래 자료 유형을 정확히 확인하고 답변하세요:\n"
    
    if precedent_count > 0:
        header += f"• 판례 자료: [판례-번호] 🏛️ 판례 형태로 표시됨\n"
    if interpretation_count > 0:
        header += f"• 법령해석례 자료: [법령해석례-번호] ⚖️ 법령해석례 형태로 표시됨\n"
    if qa_count > 0:
        header += f"• 생활법령 자료: [백문백답-번호] 💡 생활법령 Q&A 형태로 표시됨\n"
    if news_count > 0:
        header += f"• 뉴스 자료: [뉴스-번호] 📰 뉴스 형태로 표시됨\n"
    
    header += "="*60 + "\n\n"
    
    result = header + "\n\n".join(formatted_docs)
    
    return result

# 7. 최적화된 검색 함수
def optimized_retrieve_and_format(query):
    """최적화된 검색 및 포맷팅 - 전처리 포함"""
    try:
        rag_system = get_rag_system()
        docs, search_type = rag_system.conditional_retrieve(query)
        
        if not isinstance(docs, list):
            return f"검색 결과 형식 오류: {type(docs)}"
        
        return format_docs_optimized(docs, search_type)
        
    except Exception as e:
        print(f"❌ 검색 오류: {e}")
        return f"검색 중 오류가 발생했습니다: {str(e)}"

# ——— 채팅 체인 생성 ———
def create_user_friendly_chat_chain(rag_system):
    """사용자 친화적 채팅 체인 생성"""
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.3,
        max_tokens=3000,
    )

    system_message = """
    당신은 부동산 임대차, 전세사기, 법령해석, 생활법령 Q&A, 뉴스 기사 등 다양한 법률 데이터를 바탕으로 청년을 돕는 법률 전문가 AI 챗봇입니다.  
    특히 전세사기 피해 등 부동산 문제로 어려움을 겪는 사람들에게 쉽고 실질적인 도움을 제공하는 역할을 합니다.

    ---

    ### ✅ 답변 원칙
    1. **어려운 법률 용어는 일상적인 표현**으로 바꿔 설명합니다.  
    2. **구체적이고 실천 가능한 해결 방법**을 제시합니다.  
    3. **친절하고 따뜻한 말투**를 사용하여, 불안한 상황에 있는 사람에게 위로와 힘이 되도록 합니다.  
    4. **법률적 근거가 있는 정보만 제공**하며, 출처를 명확히 표기합니다.  

    ---

    ### 🧭 답변 구조

    [질문 해석 안내]  
    사용자의 질문을 법률 검색에 적합하게 바꿔서 이해했음을 간단히 설명합니다.<br>
    (예: "질문하신 내용을 법률 용어로 바꾸면 '보증금을 돌려받지 못한 경우에 대한 법적 대응'으로 볼 수 있어요.")<br>

    ##### 🔹 **유사 판례 요약**  
    법률 벡터DB에서 찾은 관련 판례를 설명하고, 사용자 질문에 맞게 핵심 내용을 풀어 설명합니다.  
    유사 판례는 2개를 가져오세요. 
    출처 표기는 판례 요약 앞에는 달지 마세요. 답변 뒤에 달아주세요. 
    각 판례 앞에는 1, 2번으로 숫자를 적어주세요. 
    → 출처 표기: (예: **[참고: 판례-194950]**

    ##### 🔹 **관련 뉴스**  
    뉴스 백터DB에서 찾은 관련 뉴스를 설명하고, 사용자 질문에 맞게 핵심 내용을 풀어 설명합니다.  
    뉴스 백터DB에서 찾은 뉴스가 없다면 **[관련 뉴스]** 부분은 전체 생략하세요.  
    → 출처 표기: **[참고: 뉴스]**

    ##### 🔹 **법령해석례, 생활법령 Q&A 참고**  
    법령해석례, 생활법령 Q&A 에 유사한 내용이 있는 경우 사용자 질문에 맞게 설명하고,  
    없다면 **[법령해석례, 생활법령 Q&A 참고]** 부분은 전체 생략하세요.  

    정부 기관의 유권해석이 있는 경우 설명하고, 실생활에 어떻게 적용되는지도 안내합니다.  
    → 출처 표기: **[참고: 법령해석례]**

    생활법령정보 '백문백답' 중 유사 사례가 있다면 사용자 질문에 맞게 설명하며 연결해줍니다.  
    → 출처 표기: **[참고: 생활법령]**

    ---

    ##### ✔️ **행동방침 제안**  
    위의 법률 자료들을 종합하여 지금 상황에서 할 수 있는 **단계별 실행 계획** 제시:
 

    각 단계별로 방법, 연락처, 비용 등을 안내합니다.
    단계 당 1줄을 넘지 마세요.

    ###### ※ **유의사항**  
    법률 자료를 바탕으로 한 주의점:  
    - 판례/해석례에서 나타난 주의할 점들  
    - 실수하기 쉬운 부분과 대비책  
    - 전문가 상담이 필요한 경우와 상담 기관 안내  
    - 법적 분쟁에서 주의해야 할 점이나 추가로 고려할 사항 정리  

    유의사항은 핵심 내용만 1줄로 요약하세요.

    ---

    ### 📌 중요 지침
    - 각 자료의 **구체적인 번호나 식별자**를 정확히 인용하세요.  
    - 참고자료의 내용을 **단순 복사하지 말고**, 사용자 질문에 맞게 **해석하여 설명**하세요.  
    - context에 해당 유형의 자료가 없으면 **그 자료는 생략**하세요.  
    - 필요 시 **법률 상담, 상담 기관 등도 안내**합니다.  
    - **중복된 내용은 한 번만** 표기하세요.  
    → 출처 표기: **[참고: 판례]**
    """

    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
        ("system", "참고자료:\n{context}")
    ])
    
    def user_friendly_retrieve_and_format(query):
        """사용자 친화적 검색 및 포맷팅 - 전처리 포함"""
        try:
            rag_system = get_rag_system()
            docs, search_type = rag_system.conditional_retrieve(query)
            formatted_result = format_docs_optimized(docs, search_type)
            return formatted_result
        except Exception as e:
            print(f"❌ 검색 오류: {e}")
            return "검색 중 오류가 발생했습니다."
    
    chain = (
        {
            "context": RunnableLambda(lambda x: user_friendly_retrieve_and_format(x["question"])),
            "question": RunnableLambda(lambda x: x["question"]),
            "chat_history": RunnableLambda(lambda x: x.get("chat_history", [])),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# ——— 메모리 관리 ———
store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    history = store[session_id]
    if len(history.messages) > 20:
        history.messages = history.messages[-20:]
    return history

def create_chat_chain_with_memory(rag_system):
    """메모리 기능이 있는 채팅 체인"""
    base_chain = create_user_friendly_chat_chain(rag_system)
    chain_with_history = RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )
    return chain_with_history

# ——— 광고 배너 함수 ———
def display_ad_banner():
    st.markdown("---")
    st.markdown('<h5 style="color: #b45309;">✨ 추천 부동산 전문가</h5>', unsafe_allow_html=True)

    ads = [
        {
            "img": "https://search.pstatic.net/common/?autoRotate=true&type=w560_sharpen&src=https%3A%2F%2Fldb-phinf.pstatic.net%2F20180518_269%2F1526627900915a2haI_PNG%2FDhZnKmpdc0bNIHMpMyeDLuUE.png",
            "title": "🏢 대치래미안공인중개사사무소",
            "phone": "0507-1408-0123",
            "desc": "📍 서울 강남구 대치동",
            "link": "https://naver.me/xslBVRJX"
        },
        {
            "img": "https://search.pstatic.net/common/?src=https%3A%2F%2Fldb-phinf.pstatic.net%2F20250331_213%2F1743412607070OviNF_JPEG%2F1000049538.jpg",
            "title": "🏡 메종공인중개사사무소",
            "phone": "0507-1431-4203",
            "desc": "🏠 전문 부동산 상담",
            "link": "https://naver.me/IgJnnCcG"
        },
        {
            "img": "https://search.pstatic.net/common/?autoRotate=true&type=w560_sharpen&src=https%3A%2F%2Fldb-phinf.pstatic.net%2F20200427_155%2F15879809374237E6dq_PNG%2FALH-zx7fy26wJg1T6EUOHC0W.png",
            "title": "👑 로얄공인중개사사무소",
            "phone": "02-569-8889",
            "desc": "🌟 신뢰할 수 있는 거래",
            "link": "https://naver.me/5GGPXQe8"
        }
    ]

    for ad in ads:
        st.markdown(f"""
        <div style="
            background-color: #fffbea;
            border-radius: 15px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.06);
        ">
            <div style="display: flex; align-items: center;">
                <img src="{ad['img']}" style="width: 3cm; height: 2cm; object-fit: cover; border-radius: 8px; margin-right: 15px;" />
                <div>
                    <p style="margin-bottom: 5px; font-size: 16px; font-weight: 600;">{ad['title']}</p>
                    <p style="margin: 0;">☎ <strong>{ad['phone']}</strong></p>
                    <p style="margin: 0;">{ad['desc']}</p>
                    <a href="{ad['link']}" target="_blank" style="color: #b45309; font-weight: bold;">🔗 바로가기</a>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("💡 **신뢰할 수 있는 부동산 전문가와 상담하세요**")

# ——— 메인 애플리케이션 ———
def main():
    """메인 애플리케이션 함수"""
    
    # Streamlit 페이지 설정
    st.set_page_config(
        page_title="AI 스위치온 - 판례 검색 시스템", 
        page_icon="🏠", 
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 커스텀 CSS 로드
    load_custom_css()

    # ——— 헤더 ———
    st.markdown("""
    <div class="header-container">
        <div class="header-title">💡 <span class="highlight">AI 스위치온</span></div>
        <div class="header-subtitle">판례 기반 AI 부동산 거래 지원 서비스</div>
        <div style="margin-top: 1rem; font-size: 1rem; color: #e5e7eb;">
            💡 상황을 자세하게 설명해주시면 맞춤형 법률 정보를 제공해드립니다
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ——— 🚀 핵심! 시스템 초기화 ———
    with st.spinner("🔄 AI 시스템 초기화 중..."):
        embedding_model, legal_db, news_db, system_ready = initialize_embeddings_and_databases()

    # ——— 세션 초기화 ———
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ——— RAG 시스템 및 채팅 체인 생성 ———
    if system_ready and (legal_db or news_db):
        try:
            rag_system = OptimizedConditionalRAGSystem(legal_db, news_db)
            chain = create_chat_chain_with_memory(rag_system)
        except Exception as e:
            st.error(f"❌ RAG 시스템 오류: {str(e)}")
            chain = None
            rag_system = None
    else:
        chain = None
        rag_system = None

    # ——— 사이드바 ———
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h2 style="color: #6b21a8; margin-bottom: 1rem;">🔍 빠른 질문</h2>
        </div>
        """, unsafe_allow_html=True)
        
        example_questions = [
            "전세사기 당했을 때 대처방법은?",
            "보증금을 돌려받을 수 있을까요?",
            "임차권등기명령이란 무엇인가요?",
            "집주인이 등기이전을 안 해줄 때 어떻게 하나요?",
            "집이 경매로 넘어갔을 때 전세보증금은 어떻게 되나요?"
        ]
        
        for i, q in enumerate(example_questions):
            if st.button(f" {q}", key=f"example_{i}", use_container_width=True):
                st.session_state["sidebar_prompt"] = q
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("↻ 대화 기록 초기화", use_container_width=True, type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

        st.markdown("<hr>", unsafe_allow_html=True)
        
        # 시스템 상태 표시
        st.markdown("""
        <div class="sidebar-card" style="border: 2px solid #10b981; background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);">
            <h4 style="color: #065f46; margin-bottom: 1rem;">📊 시스템 상태</h4>
        </div>
        """, unsafe_allow_html=True)
        
        if system_ready:
            st.success("✅ 시스템 준비완료")
        else:
            st.error("❌ 시스템 초기화 실패")
        
        # 데이터베이스 상태
        if legal_db:
            st.success("✅ 법률 DB 연결됨")
        else:
            st.warning("⚠️ 법률 DB 미연결")
            
        if news_db:
            st.success("✅ 뉴스 DB 연결됨")
        else:
            st.warning("⚠️ 뉴스 DB 미연결")
        
        st.markdown("""
        <div class="sidebar-card" style="border: 2px solid #8b5cf6;">
            <h4 style="color: #6b21a8; margin-bottom: 1rem;">✔️ 서비스 안내</h4>
            <ul style="color: #4b5563; line-height: 1.6; font-weight: bold;">
                <li>부동산 관련 법률 문제 상담</li>
                <li>판례 기반 답변 제공</li>
                <li>전세사기 피해 대처방안 안내</li>
                <li>일반인도 이해하기 쉬운 설명</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="sidebar-card" style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border: 2px solid #f5bd5f;">
            <h4 style="color: #80370b; margin-bottom: 1rem;">⚠️ 주의사항</h4>
            <ul style="color: #92400e; line-height: 1.6; margin: 0;">
                <li>본 서비스는 부동산 법률 정보를 참고용으로 제공하는 AI입니다.</li>
                <li>중요한 법적 문제는 반드시 변호사와 상담하시기 바랍니다.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # ——— 채팅 UI ———
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    # 채팅 기록 표시
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                <div class="user-bubble">
                    {message["content"]}
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif message["role"] == "assistant":
            st.markdown(f"""
            <div class="ai-message">
                <div class="ai-bubble">
                    {message["content"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # AI 답변 후 광고 배너 표시
            display_ad_banner()

    st.markdown('</div>', unsafe_allow_html=True)

    # ——— 질문 입력 ———
    prompt = st.session_state.pop("sidebar_prompt", None)
    if not prompt:
        st.markdown("""
        <div style="position: sticky; bottom: 0; background: rgba(255,255,255,0.95); 
                    padding: 1rem; border-radius: 15px; margin-top: 2rem;
                    box-shadow: 0 -5px 15px rgba(139, 92, 246, 0.1);
                    backdrop-filter: blur(10px);">
        """, unsafe_allow_html=True)
        
        prompt = st.chat_input("💭 질문을 입력하세요 (예: 보증금 돌려받을 수 있을까요?)", key="user_input")
        
        st.markdown("</div>", unsafe_allow_html=True)

    # ——— 질문 처리 ———
    if prompt:
        # 사용자 메시지 저장
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # 답변 생성
        with st.spinner("🤖 AI가 판례를 검색하고 답변을 생성하고 있습니다..."):
            try:
                if chain and system_ready:
                    response = chain.invoke(
                        {"question": prompt},
                        config={"configurable": {"session_id": st.session_state.session_id}},
                    )
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                else:
                    error_message = "죄송합니다. 현재 시스템 초기화 중입니다. 잠시 후 다시 시도해주세요."
                    st.session_state.chat_history.append({"role": "assistant", "content": error_message})
                    
            except Exception as e:
                error_message = f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {str(e)}"
                st.session_state.chat_history.append({"role": "assistant", "content": error_message})

        # 답변 생성 후 페이지 새로고침
        st.rerun()

    # ——— 푸터 ———
    st.markdown("""
    <div style="margin-top: 3rem; padding: 2rem; text-align: center; 
               background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
               border-radius: 15px; border-top: 3px solid #8b5cf6;">
        <p style="color: #6b7280; margin: 0;">
            💡 <strong>AI 스위치온</strong> | 부동산 법률 상담 AI 서비스<br>
            <span style="font-size: 0.9rem;">※ 본 서비스는 참고용이며, 실제 법률 문제는 전문가와 상담하시기 바랍니다.</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
