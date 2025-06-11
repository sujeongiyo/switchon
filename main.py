import os
import time
import functools
import uuid
import logging

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
        background: transparent;  /* 배경을 투명하게 */
        /* 나머지 스타일들은 제거 */
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
    
    .ad-item {
        display: flex;
        align-items: center;
        padding: 1rem;
        margin: 0.5rem 0;
        background: rgba(255, 255, 255, 0.7);
        border-radius: 12px;
        transition: background 0.3s ease;
    }
    
    .ad-item:hover {
        background: rgba(255, 255, 255, 0.9);
    }
    
    .ad-icon {
        margin-right: 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 80px;
        height: 60px;
        border-radius: 10px;
        color: white;
        font-size: 24px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    
    /* 사이드바 스타일 */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
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
    
    /* 스피너 스타일 */
    .stSpinner > div {
        border-top-color: #8b5cf6 !important;
    }
    
    /* 제목 스타일 개선 */
    h1, h2, h3 {
        color: #6b21a8;
        font-weight: 700;
    }
    
    /* 정보 박스 스타일 */
    .stInfo {
        background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%);
        border: 1px solid #8b5cf6;
        border-radius: 12px;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 1px solid #f59e0b;
        border-radius: 12px;
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
# 최적화된 조건부 법률-뉴스 RAG 시스템 (일상어→법률어 전처리 추가)
import os
import numpy as np
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.document_transformers import LongContextReorder
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from chromadb.utils import embedding_functions
import logging
from functools import lru_cache
import time

# 메모리 관련 import
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI

# 로깅 설정 최적화
logging.basicConfig(level=logging.WARNING)

# 1. 일상어 → 법률어 전처리 클래스 추가
class LegalQueryPreprocessor:
    """일상어를 법률 용어로 변환하는 전처리기"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,  # 일관된 변환을 위해 낮게 설정
            max_tokens=200,
        )
        
        # 캐시를 위한 딕셔너리 (세션 동안 유지)
        self._query_cache = {}
        
        # 기본 용어 매핑 (빠른 처리를 위한 룰베이스)
        self.term_mapping = {
            # 부동산 관련
            "집주인": "임대인",
            "세입자": "임차인", 
            "전세금": "임대차보증금",
            "보증금": "임대차보증금",
            "월세": "차임",
            "방세": "차임",
            "계약서": "임대차계약서",
            "집 나가라": "명도청구",
            "쫓겨나다": "명도",
            "돈 안줘": "채무불이행",
            "돈 못받아": "보증금반환청구",
            "사기": "사기죄",
            "속았다": "기망행위",
            "깡통전세": "전세사기",
            "이중계약": "중복임대",
            
            # 법적 절차 관련
            "고소": "형사고발",
            "고발": "형사고발", 
            "소송": "민사소송",
            "재판": "소송",
            "변호사": "법무사",
            "상담": "법률상담",
            "해결": "분쟁해결",
            "보상": "손해배상",
            "배상": "손해배상",
            
            # 기타
            "계약": "법률행위",
            "약속": "계약",
            "위반": "채무불이행",
            "어기다": "위반하다"
        }
    
    def _apply_rule_based_conversion(self, query: str) -> str:
        """룰베이스 용어 변환 (빠른 처리)"""
        converted_query = query
        for common_term, legal_term in self.term_mapping.items():
            if common_term in converted_query:
                converted_query = converted_query.replace(common_term, legal_term)
        return converted_query
    
    def _is_already_legal_query(self, query: str) -> bool:
        """이미 법률 용어가 포함된 쿼리인지 확인"""
        legal_indicators = [
            "임대인", "임차인", "임대차", "명도", "채무불이행", 
            "손해배상", "민사소송", "형사고발", "보증금반환",
            "법률", "판례", "법령", "소송", "계약서"
        ]
        return any(term in query for term in legal_indicators)
    
    @lru_cache(maxsize=100)
    def _gpt_convert_to_legal_terms(self, user_query: str) -> str:
        """GPT를 사용한 정교한 법률 용어 변환 (캐싱 적용)"""
        try:
            prompt = f"""다음 일상어 질문을 법률 검색에 적합한 전문 용어로 변환해주세요.
            
            원래 질문: {user_query}

            변환 규칙:
            1. 일상어를 정확한 법률 용어로 바꾸기
            - 집주인 → 임대인
            - 세입자 → 임차인  
            - 전세금/보증금 → 임대차보증금
            - 월세 → 차임
            - 계약서 → 임대차계약서
            - 사기 → 전세사기 또는 사기죄
            - 쫓겨나다 → 명도청구

            2. 핵심 법적 쟁점을 부각시키기
            3. 검색에 도움이 되는 관련 법률 키워드 추가
            4. 원래 의미는 유지하면서 더 정확하고 전문적으로 표현

            변환된 검색 쿼리:"""

            messages = [{"role": "user", "content": prompt}]
            response = self.llm.invoke(messages)
            
            # 응답에서 불필요한 부분 제거
            converted = response.content.strip()
            if "변환된 검색 쿼리:" in converted:
                converted = converted.split("변환된 검색 쿼리:")[-1].strip()
            
            return converted
            
        except Exception as e:
            print(f"⚠️ GPT 변환 실패, 룰베이스 변환 사용: {e}")
            return self._apply_rule_based_conversion(user_query)
    
    def convert_query(self, user_query: str) -> tuple[str, str]:
        """
        사용자 쿼리를 법률 검색에 적합하게 변환
        
        Returns:
            tuple: (변환된_쿼리, 변환_방법)
        """
        try:
            # 1. 이미 법률 용어인 경우 그대로 사용
            if self._is_already_legal_query(user_query):
                return user_query, "no_conversion"
            
            # 2. 캐시 확인
            if user_query in self._query_cache:
                return self._query_cache[user_query], "cached"
            
            # 3. 먼저 룰베이스 변환 시도
            rule_converted = self._apply_rule_based_conversion(user_query)
            
            # 4. 룰베이스 변환으로 충분한 경우 (많은 변환이 일어난 경우)
            if len(rule_converted) != len(user_query) or rule_converted != user_query:
                # 룰베이스 변환이 효과가 있었다면 결과 캐싱
                self._query_cache[user_query] = rule_converted
                return rule_converted, "rule_based"
            
            # 5. 복잡한 경우 GPT 변환 (시간이 더 걸리지만 정확함)
            print("🔄 정교한 법률 용어 변환 중...")
            gpt_converted = self._gpt_convert_to_legal_terms(user_query)
            
            # 결과 캐싱
            self._query_cache[user_query] = gpt_converted
            return gpt_converted, "gpt_converted"
            
        except Exception as e:
            print(f"⚠️ 쿼리 변환 오류: {e}")
            return user_query, "error"

# 2. 싱글톤 패턴으로 임베딩 모델 최적화 (기존과 동일)
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
    
    @lru_cache(maxsize=128)
    def embed_query_cached(self, text):
        return tuple(self.model.encode(text))
    
    def embed_documents(self, texts):
        return self.model.encode(texts, batch_size=32)
    
    def embed_query(self, text):
        cached_result = self.embed_query_cached(text)
        return np.array(cached_result)

class OptimizedChromaDefaultEmbeddings(metaclass=SingletonMeta):
    def __init__(self):
        if not hasattr(self, 'embedding_function'):
            from chromadb.utils import embedding_functions
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
    
    @lru_cache(maxsize=128)
    def embed_query_cached(self, text):
        result = self.embedding_function([text])
        return tuple(result[0])
    
    def embed_documents(self, texts):
        try:
            return self.embedding_function(texts)
        except Exception as e:
            print(f"⚠️ 배치 임베딩 실패, 개별 처리: {e}")
            results = []
            for text in texts:
                try:
                    result = self.embedding_function([text])
                    results.append(result[0])
                except Exception as ex:
                    print(f"⚠️ 개별 텍스트 임베딩 실패: {ex}")
                    results.append([0.0] * 384)
            return results
    
    def embed_query(self, text):
        try:
            cached_result = self.embed_query_cached(text)
            return np.array(cached_result)
        except Exception as e:
            print(f"⚠️ 쿼리 임베딩 캐시 실패: {e}")
            result = self.embedding_function([text])
            return np.array(result[0])

# 3. 전처리 기능이 추가된 최적화된 조건부 검색 시스템
class OptimizedConditionalRAGSystem:
    def __init__(self, legal_db_path, news_db_path, legal_collection, news_collection):
        print("🚀 최적화된 RAG 시스템 초기화 중...")
        start_time = time.time()
        
        # 쿼리 전처리기 초기화
        print("🔄 법률 용어 전처리기 초기화 중...")
        self.query_preprocessor = LegalQueryPreprocessor()
        print("✅ 법률 용어 전처리기 준비 완료")
        
        # 임베딩 함수 초기화
        self.legal_embedding_function = OptimizedKoSBERTEmbeddings()
        print("📊 법률 DB와 뉴스 DB 모두 KoSBERT 768차원 임베딩 사용")
        
        # 임계값 설정
        self.legal_similarity_threshold = 0.7
        self.news_similarity_threshold = 0.6
        self.min_relevant_docs = 3
        
        # DB 연결 최적화
        self._init_legal_db(legal_db_path, legal_collection)
        self._init_news_db(news_db_path, news_collection)
        
        init_time = time.time() - start_time
        print(f"⚡ 시스템 초기화 완료 ({init_time:.2f}초)")
    
    def _init_legal_db(self, legal_db_path, legal_collection):
        """법률 DB 초기화 최적화"""
        print(f"🏛️ 법률 DB 연결 중...")
        try:
            self.legal_db = Chroma(
                persist_directory=legal_db_path,
                collection_name=legal_collection,
                embedding_function=self.legal_embedding_function
            )
            
            self.legal_documents = None
            self.legal_bm25_retriever = None
            self.legal_hybrid_retriever = None
            
            self.legal_vector_retriever = self.legal_db.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": 5}
            )
            print("✅ 법률 DB 연결 완료")
            
        except Exception as e:
            print(f"❌ 법률 DB 연결 실패: {e}")
            self.legal_db = None
            self.legal_vector_retriever = None
    
    def _init_news_db(self, news_db_path, news_collection):
        """뉴스 DB 초기화 최적화"""
        print(f"📰 뉴스 DB 연결 중...")
        try:
            self.news_db = Chroma(
                persist_directory=news_db_path,
                collection_name=news_collection,
                embedding_function=self.legal_embedding_function
            )
            
            news_count = self.news_db._collection.count()
            print(f"✅ 뉴스 DB 연결 완료 (문서 수: {news_count})")
            
            try:
                test_docs = self.news_db.similarity_search("전세", k=1)
                print(f"🧪 검색 테스트 결과: {len(test_docs)}개")
                if test_docs:
                    print(f"   샘플 제목: {test_docs[0].metadata.get('title', '제목없음')[:50]}...")
            except Exception as test_e:
                print(f"⚠️ 검색 테스트 실패: {test_e}")
            
            self.news_vector_retriever = self.news_db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )
            
            print("✅ 뉴스 DB KoSBERT 768차원 임베딩 설정 완료")
            
        except Exception as e:
            print(f"❌ 뉴스 DB 연결 실패: {e}")
            import traceback
            print(f"상세 오류: {traceback.format_exc()}")
            self.news_db = None
            self.news_vector_retriever = None
    
    def _lazy_init_hybrid_retriever(self):
        """하이브리드 리트리버 지연 초기화"""
        if self.legal_hybrid_retriever is None and self.legal_db is not None:
            print("🔄 하이브리드 리트리버 초기화 중...")
            try:
                legal_data = self.legal_db.get()
                
                if not legal_data or not legal_data.get("documents"):
                    print("⚠️ 법률 DB에서 문서를 가져올 수 없음")
                    return
                
                all_legal_docs = legal_data["documents"]
                all_legal_metadatas = legal_data.get("metadatas", [{}] * len(all_legal_docs))
                
                if len(all_legal_metadatas) < len(all_legal_docs):
                    all_legal_metadatas.extend([{}] * (len(all_legal_docs) - len(all_legal_metadatas)))
                
                self.legal_documents = [
                    Document(page_content=doc, metadata=meta or {})
                    for doc, meta in zip(all_legal_docs, all_legal_metadatas)
                ]
                
                print(f"📄 법률 문서 로딩 완료: {len(self.legal_documents)}개")
                
                self.legal_bm25_retriever = BM25Retriever.from_documents(self.legal_documents)
                self.legal_bm25_retriever.k = 8
                
                self.legal_hybrid_retriever = EnsembleRetriever(
                    retrievers=[self.legal_vector_retriever, self.legal_bm25_retriever],
                    weights=[0.65, 0.35]
                )
                print("✅ 하이브리드 리트리버 초기화 완료")
                
            except Exception as e:
                print(f"⚠️ 하이브리드 리트리버 초기화 실패, 벡터 검색만 사용: {e}")
                import traceback
                print(f"상세 오류: {traceback.format_exc()}")
                self.legal_hybrid_retriever = None
    
    def calculate_cosine_similarity_score(self, query, docs, use_news_embedding=False):
        """최적화된 코사인 유사도 계산"""
        if not docs:
            return 0.0
        
        try:
            query_embedding = self.legal_embedding_function.embed_query(query)
            
            if not isinstance(query_embedding, np.ndarray):
                query_embedding = np.array(query_embedding)
            
            if query_embedding.ndim > 1:
                query_embedding = query_embedding.flatten()
            
            limited_docs = docs[:5]
            doc_texts = [doc.page_content[:1500] for doc in limited_docs]
            
            doc_embeddings = self.legal_embedding_function.embed_documents(doc_texts)
            
            if not isinstance(doc_embeddings, np.ndarray):
                doc_embeddings = np.array(doc_embeddings)
            
            if doc_embeddings.ndim == 1:
                doc_embeddings = doc_embeddings.reshape(1, -1)
            
            if query_embedding.shape[0] != doc_embeddings.shape[1]:
                print(f"⚠️ 차원 불일치: 쿼리 {query_embedding.shape[0]}, 문서 {doc_embeddings.shape[1]}")
                return 0.65
            
            similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
            
            return float(np.max(similarities)) if len(similarities) > 0 else 0.0
            
        except Exception as e:
            print(f"⚠️ 유사도 계산 오류: {e}")
            return 0.65
    
    def search_legal_db(self, query):
        """최적화된 법률 DB 검색"""
        if self.legal_db is None:
            print("❌ 법률 DB가 연결되지 않음")
            return [], 0.0
        
        try:
            # 🔍 쿼리 확장 적용
            expanded_query = self._expand_query_with_legal_terms(query)
            if expanded_query != query:
                print(f"🔄 확장된 법률 검색 쿼리: {expanded_query}")
            else:
                print(f"🔍 법률 DB 검색 쿼리: {query}")
            
            # 확장된 쿼리로 검색
            legal_docs = self.legal_vector_retriever.invoke(expanded_query)
            print(f"📄 벡터 검색 결과: {len(legal_docs)}개 문서")
            
            if legal_docs:
                for i, doc in enumerate(legal_docs[:2]):
                    doc_type = doc.metadata.get('doc_type', '유형불명')
                    content_preview = str(doc.page_content)[:50] if doc.page_content else "내용없음"
                    print(f"   [{i+1}] {doc_type}: {content_preview}...")
            else:
                print("   ⚠️ 벡터 검색에서 문서를 찾지 못함")
            
            if len(legal_docs) < self.min_relevant_docs:
                print(f"📊 문서 개수 부족({len(legal_docs)} < {self.min_relevant_docs}), 하이브리드 검색 시도")
                self._lazy_init_hybrid_retriever()
                if self.legal_hybrid_retriever is not None:
                    # 하이브리드 검색도 확장된 쿼리 사용
                    hybrid_docs = self.legal_hybrid_retriever.invoke(expanded_query)
                    print(f"📄 하이브리드 검색 결과: {len(hybrid_docs)}개 문서")
                    if len(hybrid_docs) > len(legal_docs):
                        legal_docs = hybrid_docs
                        print("✅ 하이브리드 검색 결과로 교체")
            
            if len(legal_docs) > 5:
                legal_docs = LongContextReorder().transform_documents(legal_docs)
                print("🔄 문서 재정렬 완료")
            
            # 유사도 계산은 원본 쿼리로 (더 정확한 평가를 위해)
            similarity_score = self.calculate_cosine_similarity_score(query, legal_docs, use_news_embedding=False)
            print(f"📊 법률 DB 최종 점수: {similarity_score:.3f}")
            
            return legal_docs, similarity_score
            
        except Exception as e:
            print(f"❌ 법률 DB 검색 오류: {e}")
            import traceback
            print(f"상세 오류: {traceback.format_exc()}")
            return [], 0.0
    
    def search_news_db(self, query):
        """최적화된 뉴스 DB 검색"""
        if self.news_db is None or self.news_vector_retriever is None:
            print("❌ 뉴스 DB가 연결되지 않음")
            return [], 0.0
        
        try:
            enhanced_query = self._enhance_news_query(query)
            print(f"🔍 뉴스 검색 쿼리: {enhanced_query}")
            
            news_docs = self.news_vector_retriever.invoke(enhanced_query)
            print(f"📰 뉴스 검색 결과: {len(news_docs)}개")
            
            if news_docs:
                for i, doc in enumerate(news_docs[:2]):
                    title = doc.metadata.get('title', '제목없음')
                    print(f"   [{i+1}] {title[:50]}...")
            else:
                print("   ⚠️ 뉴스 검색에서 문서를 찾지 못함")
            
            if news_docs:
                similarity_score = self.calculate_cosine_similarity_score(
                    enhanced_query, news_docs, use_news_embedding=False
                )
                print(f"📊 뉴스 검색 점수: {similarity_score:.3f}")
            else:
                similarity_score = 0.0
            
            return news_docs, similarity_score
            
        except Exception as e:
            print(f"❌ 뉴스 DB 검색 오류: {e}")
            import traceback
            print(f"상세 오류: {traceback.format_exc()}")
            return [], 0.0
    
    def _expand_query_with_legal_terms(self, query):
        """법률 쿼리 확장 - 동의어와 관련 용어 추가"""
        expansion_terms = []
        
        # 부동산 관련 확장
        if "보증금" in query: 
            expansion_terms.extend(["임대차보증금", "전세금"])
        if "집주인" in query: 
            expansion_terms.append("임대인")
        if "세입자" in query: 
            expansion_terms.append("임차인")
        if "월세" in query:
            expansion_terms.append("차임")
        if "계약서" in query:
            expansion_terms.append("임대차계약서")
        
        # 법적 절차 관련 확장
        if "소송" in query:
            expansion_terms.extend(["민사소송", "소송절차"])
        if "손해배상" in query:
            expansion_terms.append("배상청구")
        if "명도" in query:
            expansion_terms.extend(["명도청구", "퇴거"])
        
        # 전세사기 관련 확장
        if any(term in query for term in ["사기", "깡통전세", "전세사기"]):
            expansion_terms.extend(["전세사기", "임대차사기", "보증금사기"])
        
        # 최대 3개 용어만 추가 (너무 길어지지 않게)
        if expansion_terms:
            unique_terms = list(set(expansion_terms))[:3]
            expanded_query = f"{query} {' '.join(unique_terms)}"
            return expanded_query
        
        return query
    
    def _enhance_news_query(self, query):
        """뉴스 쿼리 강화"""
        # 기본 강화
        enhanced = query
        if any(term in query for term in ["전세", "부동산", "임대", "사기"]):
            enhanced = f"{query} 전세사기"
        
        return enhanced
    
    def conditional_retrieve(self, original_query):
        """조건부 검색 - 쿼리 전처리 추가"""
        try:
            print(f"🔍 원본 검색 쿼리: {original_query}")
            
            # ✅ 핵심 추가: 일상어 → 법률어 전처리
            converted_query, conversion_method = self.query_preprocessor.convert_query(original_query)
            
            if conversion_method != "no_conversion":
                print(f"🔄 변환된 검색 쿼리: {converted_query}")
                print(f"📋 변환 방법: {conversion_method}")
                # 실제 검색에는 변환된 쿼리 사용
                search_query = converted_query
            else:
                print("📋 이미 법률 용어로 구성된 쿼리")
                search_query = original_query
            
            # 1단계: 법률 DB 검색 (변환된 쿼리 사용)
            print("🏛️ 법률 DB 검색 중...")
            legal_docs, legal_score = self.search_legal_db(search_query)
            print(f"📊 법률 DB 결과: {len(legal_docs)}개 문서, 점수: {legal_score:.3f}")
            
            # 2단계: 법률 DB 결과 충분성 평가
            is_legal_sufficient = (
                legal_score >= self.legal_similarity_threshold and 
                len(legal_docs) >= self.min_relevant_docs
            )
            
            if is_legal_sufficient:
                print("✅ 법률 DB 결과만으로 충분함")
                return legal_docs, "legal_only"
            
            # 3단계: 법률 DB 결과가 부족한 경우에만 뉴스 DB 검색
            print("📰 법률 DB 결과 부족, 뉴스 DB로 보완 검색...")
            news_docs, news_score = self.search_news_db(search_query)
            print(f"📊 뉴스 DB 결과: {len(news_docs)}개 문서, 점수: {news_score:.3f}")
            
            # 4단계: 결과 결합
            combined_docs = []
            
            if legal_docs:
                combined_docs.extend(legal_docs[:8])  # 법률 문서 최대 8개로 증가
                print(f"✅ 법률 문서 {len(legal_docs[:8])}개 추가")
            
            if news_docs and news_score >= self.news_similarity_threshold:
                combined_docs.extend(news_docs[:3])  # 뉴스는 3개로 조정 (총 11개 방지)
                print(f"✅ 뉴스 문서 {len(news_docs[:5])}개 추가")
                search_type = "legal_and_news"
            elif legal_docs:
                search_type = "legal_only"
            else:
                search_type = "no_results"
            
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

# 5. 환경 변수 및 설정
load_dotenv()
CHROMA_DB_PATH = "D:\\chroma_db_law_real_final"
NEWS_DB_PATH = "D:\\ja_chroma_db"
LEGAL_COLLECTION_NAME = "legal_db"
NEWS_COLLECTION_NAME = "jeonse_fraud_embedding"

# 6. 전역 시스템 인스턴스 (지연 초기화)
_conditional_rag = None

def get_rag_system():
    """RAG 시스템 싱글톤 인스턴스 반환"""
    global _conditional_rag
    if _conditional_rag is None:
        _conditional_rag = OptimizedConditionalRAGSystem(
            legal_db_path=CHROMA_DB_PATH,
            news_db_path=NEWS_DB_PATH,
            legal_collection=LEGAL_COLLECTION_NAME,
            news_collection=NEWS_COLLECTION_NAME
        )
    return _conditional_rag

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

# 8. 일반 사용자용 체인 생성 - 전처리 정보 포함
def create_user_friendly_chat_chain():
    """일반 사용자를 위한 친화적 체인 - 쿼리 전처리 정보 포함"""
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


# 9. 메모리 관리 최적화
store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    history = store[session_id]
    if len(history.messages) > 20:
        history.messages = history.messages[-20:]
    return history

def create_chat_chain_with_memory():
    """메모리 기능이 있는 사용자 친화적 체인 생성"""
    base_chain = create_user_friendly_chat_chain()
    chain_with_history = RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )
    return chain_with_history



# ——— 광고 배너 함수 (이미지 포함된 버전) ———

import streamlit as st

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





# ——— Streamlit UI ———
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

# ——— 세션 초기화 ———
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

chain = create_chat_chain_with_memory()



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
    
    st.markdown("""
    <div class="sidebar-card" style="border: 2px solid #8b5cf6; padding: 1rem; border-radius: 0.5rem; box-shadow: 2px 2px 10px rgba(139, 92, 246, 0.2);">
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
            <li>본 서비스는 부동산 법률 정보를 참고용으로 제공하는 AI로, 법률 전문가가 아닙니다.</li>
            <li>중요한 법적 문제는 반드시 변호사와 상담하시며, 챗봇의 답변에 대한 법적 책임을 지지 않습니다.</li>
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
    # 커스텀 입력창 스타일
    st.markdown("""
    <div style="position: sticky; bottom: 0; background: rgba(255,255,255,0.95); 
                padding: 1rem; border-radius: 15px; margin-top: 2rem;
                box-shadow: 0 -5px 15px rgba(139, 92, 246, 0.1);
                backdrop-filter: blur(10px);">
    """, unsafe_allow_html=True)
    
    prompt = st.chat_input("💭 질문을 입력하세요 (예: 보증금 돌려받을 수 있을까요?)", key="user_input")
    
    st.markdown("</div>", unsafe_allow_html=True)


if prompt:
    # 사용자 메시지 저장
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # 로딩 애니메이션과 함께 답변 생성
    with st.spinner("🤖 AI가 판례를 검색하고 답변을 생성하고 있습니다..."):
        try:
            response = chain.invoke(
                {"question": prompt},
                config={"configurable": {"session_id": st.session_state.session_id}},
            )
            st.session_state.chat_history.append({"role": "assistant", "content": response})
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
