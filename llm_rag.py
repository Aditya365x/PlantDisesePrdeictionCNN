import os
from typing import Optional

from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 👉 OpenAI + LangChain integration
from langchain_openai import ChatOpenAI


# ==========================================================
# 1. ENV, PATHS, GLOBALS
# ==========================================================
load_dotenv()  # loads OPENAI_API_KEY, OPENAI_MODEL

# Path where your FAISS index (index.faiss + index.pkl) is stored
VECTOR_DIR = r"D:\ml3\sih_disese_pred2\faiss"   # ⬅️ change if needed

# Embedding model used when you built the index
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# OpenAI model name (can also be read from env)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


_QA_CHAIN: Optional[RetrievalQA] = None   # global cache


# ==========================================================
# 2. LOAD EXISTING FAISS VECTORSTORE
# ==========================================================
def _load_vectorstore() -> FAISS:
    if not os.path.isdir(VECTOR_DIR):
        raise FileNotFoundError(
            f"FAISS index folder not found at: {VECTOR_DIR}\n"
            f"Make sure index.faiss and index.pkl are present there."
        )

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = FAISS.load_local(
        VECTOR_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore


# ==========================================================
# 3. BUILD LLM + RETRIEVAL QA CHAIN (OpenAI)
# ==========================================================
def _build_qa_chain() -> RetrievalQA:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Set it in your environment or .env file."
        )

    # ChatOpenAI automatically picks up OPENAI_API_KEY
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0.2,
        max_tokens=512,
    )

    vectorstore = _load_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )

    prompt_template = """
You are an expert plant pathologist and agronomist.
Use ONLY the information from the provided context to answer.

Context:
{context}

Question:
{question}

Answer in well-structured bullet points and short paragraphs.
Focus specifically on:
1. Disease description and typical symptoms.
2. Preventive measures (cultural practices, field hygiene, resistant varieties, etc.).
3. Recommended treatments or management practices (biological and chemical, if mentioned).
4. Any additional farmer-friendly practical tips.

If the context does not contain information, clearly say:
"Information not available in the provided documents."
"""

    prompt = PromptTemplate(
        template=prompt_template.strip(),
        input_variables=["context", "question"],
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False,
    )

    return qa_chain


def _get_chain() -> RetrievalQA:
    global _QA_CHAIN
    if _QA_CHAIN is None:
        _QA_CHAIN = _build_qa_chain()
    return _QA_CHAIN


# ==========================================================
# 4. PUBLIC FUNCTION FOR YOUR CNN APP
# ==========================================================
def get_preventive_measures(disease_name: str) -> str:
    """
    Main function you call from `main demo.py`.

    Example:
        text = get_preventive_measures("Sunflower Rust")
    """
    chain = _get_chain()

    question = (
        f"The CNN model has predicted the plant disease: '{disease_name}'. "
        f"Using the PDF knowledge base, explain this disease and provide:\n"
        f"1) Key symptoms\n"
        f"2) Preventive agronomic practices\n"
        f"3) Recommended treatments or controls (if mentioned)\n"
        f"Write the answer in clear, farmer-friendly language."
    )

    result = chain.invoke({"query": question})

    # RetrievalQA can return dict or str depending on LC version
    if isinstance(result, dict) and "result" in result:
        return result["result"]
    elif isinstance(result, str):
        return result
    else:
        return str(result)


if __name__ == "__main__":
    # Small manual test
    print(get_preventive_measures("Rust"))
