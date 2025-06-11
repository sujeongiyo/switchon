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
from dotenv import load_dotenv

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

load_dotenv()

# ——— 🚀 Google Drive 다운로드 함수 ———
@st.cache_data
def download_and_extract_databases():
    """Google Drive에서 ChromaDB 파일들을 다운로드하고 압축 해제"""
    
    # 압축 파일 정보 (파일명, 추출 위치, Google Drive 파일 ID)
    files_to_download = [
        {
            "filename": "chroma_db_law_real_final.zip",
            "extract_dir": "chroma_db_law_real_final",
            "gdrive_id": "1gp5h0QScWB3wcsbs4i12ny1wEMY_HAqX"
        },
        {
            "filename": "ja_chroma_db.zip", 
            "extract_dir": "ja_chroma_db",
            "gdrive_id": "1dU9TLAPMg-Q8DLQjZM38CC-TsK477dSO"
        }
    ]

    def download_and_extract_single(file_info):
        url = f"https://drive.google.com/uc?export=download&id={file_info['gdrive_id']}"
        zip_path = file_info["filename"]
        extract_path = file_info["extract_dir"]

        if not os.path.exists(extract_path):
            st.info(f"📥 다운로드 중: {zip_path}")
            
            # 다운로드 진행 상황 표시
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # requests로 파일 다운로드
                status_text.text("서버 연결 중...")
                progress_bar.progress(10)
                
                r = requests.get(url, stream=True)
                r.raise_for_status()
                
                total_size = int(r.headers.get('content-length', 0))
                downloaded_size = 0
                
                status_text.text(f"다운로드 중: {zip_path}")
                
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            if total_size > 0:
                                progress = min(50, int((downloaded_size / total_size) * 40) + 10)
                                progress_bar.progress(progress)

                progress_bar.progress(60)
                status_text.text(f"압축 해제 중: {zip_path}")

                # 압축 해제
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
                
                progress_bar.progress(90)
                
                # 압축 파일 삭제 (용량 절약)
                os.remove(zip_path)
                
                progress_bar.progress(100)
                status_text.text(f"✅ 완료: {extract_path}")
                
                # UI 정리
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
                
                st.success(f"✅ {extract_path} 준비 완료!")
                
            except Exception as e:
                st.error(f"❌ 다운로드 실패: {zip_path} - {str(e)}")
                return False
        else:
            st.success(f"✅ 이미 존재함: {extract_path}")
        
        return True

    # 모든 파일 다운로드 실행
    all_success = True
    for file_info in files_to_download:
        success = download_and_extract_single(file_info)
        all_success = all_success and success
    
    return all_success

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
    
    /* 광고 배너 */
    .ad-banner {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border: 2px solid #f59e0b;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 6px 20px rgba(245, 158, 11, 0.15);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .ad-banner:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(245, 158, 11, 0.25);
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

# ——— 원본 코드의 모든 클래스들 (그대로 유지) ———

# 1. 일상어 → 법률어 전처리 클래스
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

# 2. 싱글톤 패턴으로 임베딩 모델 최적화
class SingletonMeta(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class OptimizedKoSBERTEmbeddings(metaclass=SingletonMeta):
    def __init__(self, model_name="jhgan/ko-sbert-sts"):
        if not hasattr(self, 'model'):
            print(f"🔄 KoSBERT 모델 로딩: {model_name}")
            self.model = SentenceTransformer(model_name)
            print("✅ KoSBERT 모델 로딩 완료")
    
    @functools.lru_cache(maxsize=128)
    def embed_query_cached(self, text):
        return tuple(self.model.encode(text))
    
    def embed_documents(self, texts):
        return self.model.encode(texts, batch_size=32)
    
    def embed_query(self, text):
        cached_result = self.embed_query_cached(text)
        return np.array(cached_result)

# 3. RAG 시스템 (간소화된 버전)
class OptimizedConditionalRAGSystem:
    def __init__(self):
        print("🚀 최적화된 RAG 시스템 초기화 중...")
        
        # 쿼리 전처리기 초기화
        self.query_preprocessor = LegalQueryPreprocessor()
        print("✅ 법률 용어 전처리기 준비 완료")
        
        # 임베딩 함수 초기화
        self.legal_embedding_function = OptimizedKoSBERTEmbeddings()
        print("📊 KoSBERT 768차원 임베딩 사용")
        
        # 임계값 설정
        self.legal_similarity_threshold = 0.7
        self.news_similarity_threshold = 0.6
        self.min_relevant_docs = 3
        
        # 🚀 ChromaDB 연결 (다운로드된 파일 사용)
        self._init_databases()
    
    def _init_databases(self):
        """ChromaDB 초기화"""
        try:
            # 법률 DB 연결
            if os.path.exists("chroma_db_law_real_final"):
                self.legal_db = Chroma(
                    persist_directory="chroma_db_law_real_final",
                    collection_name="legal_db",
                    embedding_function=self.legal_embedding_function
                )
                self.legal_vector_retriever = self.legal_db.as_retriever(
                    search_type="similarity", 
                    search_kwargs={"k": 5}
                )
                print("✅ 법률 DB 연결 완료")
            else:
                print("⚠️ 법률 DB 파일이 없습니다")
                self.legal_db = None
                self.legal_vector_retriever = None
            
            # 뉴스 DB 연결  
            if os.path.exists("ja_chroma_db"):
                self.news_db = Chroma(
                    persist_directory="ja_chroma_db",
                    collection_name="jeonse_fraud_embedding",
                    embedding_function=self.legal_embedding_function
                )
                self.news_vector_retriever = self.news_db.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4}
                )
                print("✅ 뉴스 DB 연결 완료")
            else:
                print("⚠️ 뉴스 DB 파일이 없습니다")
                self.news_db = None
                self.news_vector_retriever = None
                
        except Exception as e:
            print(f"❌ DB 연결 실패: {e}")
            self.legal_db = None
            self.news_db = None
            self.legal_vector_retriever = None
            self.news_vector_retriever = None
    
    def search_legal_db(self, query):
        """법률 DB 검색"""
        if self.legal_db is None:
            return [], 0.0
        
        try:
            legal_docs = self.legal_vector_retriever.invoke(query)
            print(f"📄 법률 검색 결과: {len(legal_docs)}개 문서")
            return legal_docs, 0.8  # 간단한 고정 점수
        except Exception as e:
            print(f"❌ 법률 DB 검색 오류: {e}")
            return [], 0.0
    
    def search_news_db(self, query):
        """뉴스 DB 검색"""
        if self.news_db is None:
            return [], 0.0
        
        try:
            news_docs = self.news_vector_retriever.invoke(query)
            print(f"📰 뉴스 검색 결과: {len(news_docs)}개")
            return news_docs, 0.7  # 간단한 고정 점수
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

# 4. 문서 포맷팅 (간소화된 버전)
def format_docs_optimized(docs, search_type):
    """문서 포맷팅"""
    if not docs:
        return "관련 자료를 찾을 수 없습니다."
    
    formatted_docs = []
    for i, doc in enumerate(docs[:5]):  # 최대 5개만 표시
        try:
            meta = doc.metadata if doc.metadata else {}
            content = str(doc.page_content)[:500] if doc.page_content else ""
            
            is_news = ('url' in meta and 'title' in meta) or ('date' in meta and 'title' in meta)
            
            if is_news:
                title = str(meta.get("title", "제목없음"))[:60]
                formatted = f"[뉴스-{i+1}] 📰 {title}\n내용: {content}...\n"
            else:
                doc_type = str(meta.get("doc_type", "법률자료"))
                formatted = f"[법률-{i+1}] 🏛️ {doc_type}\n내용: {content}...\n"
            
            formatted_docs.append(formatted)
        except:
            continue
    
    return "\n\n".join(formatted_docs)

# 5. 전역 시스템 인스턴스
_conditional_rag = None

def get_rag_system():
    """RAG 시스템 싱글톤 인스턴스 반환"""
    global _conditional_rag
    if _conditional_rag is None:
        _conditional_rag = OptimizedConditionalRAGSystem()
    return _conditional_rag

# 6. 검색 함수
def optimized_retrieve_and_format(query):
    """검색 및 포맷팅"""
    try:
        rag_system = get_rag_system()
        docs, search_type = rag_system.conditional_retrieve(query)
        return format_docs_optimized(docs, search_type)
    except Exception as e:
        print(f"❌ 검색 오류: {e}")
        return f"검색 중 오류가 발생했습니다: {str(e)}"

# 7. 채팅 체인
def create_user_friendly_chat_chain():
    """사용자 친화적 체인"""
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.3,
        max_tokens=3000,
    )

    system_message = """
    당신은 부동산 임대차, 전세사기 등 법률 문제를 돕는 AI 챗봇입니다.
    
    답변 구조:
    1. 질문 해석 안내
    2. 유사 판례 요약 (2개)
    3. 관련 뉴스 (있는 경우)
    4. 행동방침 제안
    5. 유의사항
    
    친절하고 따뜻한 말투로 실질적인 도움을 제공하세요.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
        ("system", "참고자료:\n{context}")
    ])
    
    def user_friendly_retrieve_and_format(query):
        try:
            return optimized_retrieve_and_format(query)
        except Exception as e:
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
