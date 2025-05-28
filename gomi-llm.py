# rag_sample.py

import os
import json
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# APIキー読み込み
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# ① Webページのテキスト抽出
def scrape_text(url):
    """改善されたスクレイピング関数"""
    response = requests.get(url)
    response.encoding = 'utf-8'
    soup = BeautifulSoup(response.text, "html.parser")
    
    garbage_info = []
    
    # 50音順のセクションを取得（h2とh3の両方を考慮）
    sections = soup.find_all(['h2', 'h3'], class_=None)
    for section in sections:
        section_title = section.get_text(strip=True)
        if not section_title:
            continue
            
        table = section.find_next('table')
        if not table:
            continue
            
        rows = table.find_all('tr')
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 3:
                item = extract_cell_text(cells[0])
                category = extract_cell_text(cells[1])
                fee = extract_cell_text(cells[2])
                if fee == "-":
                    fee = "無料"
                
                note = ""
                if len(cells) > 3:
                    note = extract_cell_text(cells[3])
                
                if item and category:
                    info = {
                        "品目": str(item),
                        "分別区分": str(category),
                        "手数料": str(fee),
                        "備考": str(note) if note and note != "&nbsp;" else ""
                    }
                    garbage_info.append(info)
    
    if not garbage_info:
        return soup.get_text()
    
    garbage_info = clean_html_from_items(garbage_info)
    
    # JSONファイルに保存
    with open('garbage_data.json', 'w', encoding='utf-8') as f:
        json.dump(garbage_info, f, ensure_ascii=False, indent=2)
    
    # テキスト形式でも保存
    with open('garbage_data.txt', 'w', encoding='utf-8') as f:
        for info in garbage_info:
            f.write(f"品目: {info['品目']} | 分別区分: {info['分別区分']} | 手数料: {info['手数料']}")
            if info['備考']:
                f.write(f" | 備考: {info['備考']}")
            f.write('\n')
    
    return '\n'.join([f"品目: {info['品目']} | 分別区分: {info['分別区分']} | 手数料: {info['手数料']}" + 
                     (f" | 備考: {info['備考']}" if info['備考'] else "") for info in garbage_info])

def extract_cell_text(cell):
    """セルからテキストを抽出する改善された関数"""
    if '<' in str(cell) and '>' in str(cell):
        return BeautifulSoup(str(cell), 'html.parser').get_text(strip=True)
    
    return cell.get_text(strip=True)

def clean_html_from_items(data):
    """HTMLタグを品目名から除去"""
    for item in data:
        for key in ['品目', '分別区分', '手数料', '備考']:
            if key in item and '<' in str(item[key]) and '>' in str(item[key]):
                soup = BeautifulSoup(str(item[key]), 'html.parser')
                item[key] = soup.get_text(strip=True)
    
    return data

# ② テキストをチャンク分割
def chunk_text(text, chunk_size=1200, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

# ③ EmbeddingとFAISSで保存
def create_vector_store(chunks):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(chunks, embeddings)

from langchain.prompts import PromptTemplate

template = """あなたは札幌市のゴミ分別アシスタントです。
以下の情報を元に、質問に対して正確かつ丁寧に回答してください。

情報が見つからない場合は、類似する品目を探して参考情報として提供してください。
例えば、「紙パック」が見つからない場合は「紙製容器」や「カートン」などの情報を参考にしてください。

情報:
{context}

質問: {question}

回答:"""

CUSTOM_PROMPT_TEMPLATE = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# ④ 質問を処理して回答を生成
def ask_question(vectorstore, query):
    query = preprocess_query(query)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0.3, model="gpt-4"),
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={
            "prompt": CUSTOM_PROMPT_TEMPLATE
        }
    )
    return qa.run(query)

def preprocess_query(query):
    """検索クエリの前処理"""
    synonyms = {
        "牛乳パック": ["紙パック", "ミルクパック", "飲料パック"],
        "紙パック": ["牛乳パック", "ミルクパック", "飲料パック"]
    }
    
    for key, values in synonyms.items():
        if key in query:
            return query
        for value in values:
            if value in query:
                return query.replace(value, key)
    
    return query

# 実行
if __name__ == "__main__":
    url = "https://www.city.sapporo.jp/seiso/bunbetsu/index.html"
    raw_text = scrape_text(url)
    chunks = chunk_text(raw_text)
    vectorstore = create_vector_store(chunks)
    
    print("✅ データ準備完了。質問をどうぞ！\n")

    while True:
        query = input("🔍 質問してください（例: 牛乳パックは何ごみ？）：")
        if query in ["exit", "quit", ""]:
            break
        answer = ask_question(vectorstore, query)
        print("🧠 回答:", answer)
