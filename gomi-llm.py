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
    # 文字エンコーディングを指定してリクエスト
    response = requests.get(url)
    response.encoding = 'utf-8'  # 明示的にUTF-8を指定
    soup = BeautifulSoup(response.text, "html.parser")
    
    # ゴミ分別情報を抽出
    garbage_info = []
    
    # 50音順のセクションを取得
    sections = soup.find_all('h2', class_=None)
    for section in sections:
        section_title = section.get_text(strip=True)
        if not section_title:  # 空のセクションはスキップ
            continue
            
        # セクション内のテーブルを取得
        table = section.find_next('table')
        if not table:
            continue
            
        # テーブルの行を取得
        rows = table.find_all('tr')
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 3:  # 品目、分別区分、手数料の列がある場合
                # 品目（太字タグを含む場合も考慮）
                item_cell = cells[0]
                strong_tag = item_cell.find('strong')
                item = strong_tag.get_text(strip=True) if strong_tag else item_cell.get_text(strip=True)
                
                # 分別区分（リンクを含む場合も考慮）
                category_cell = cells[1]
                category = category_cell.get_text(strip=True)
                category_link = category_cell.find('a')
                if category_link:
                    category = f"{category_link.get_text(strip=True)}"
                
                # 手数料
                fee = cells[2].get_text(strip=True)
                if fee == "-":
                    fee = "無料"
                
                # 備考（リンク、段落、リストを含む場合も考慮）
                note = ""
                if len(cells) > 3:
                    note_cell = cells[3]
                    # リスト要素を処理
                    lists = note_cell.find_all(['ul', 'ol'])
                    if lists:
                        list_items = []
                        for list_elem in lists:
                            for item in list_elem.find_all('li'):
                                # リスト項目内のリンクを処理
                                links = item.find_all('a')
                                if links:
                                    for link in links:
                                        link_text = link.get_text(strip=True)
                                        item_text = item.get_text(strip=True)
                                        # リンクテキストを保持しながら、重複を避ける
                                        if link_text in item_text:
                                            list_items.append(item_text)
                                        else:
                                            list_items.append(f"{item_text} ({link_text})")
                                else:
                                    list_items.append(item.get_text(strip=True))
                        note = " ".join(list_items)
                    else:
                        # 通常のテキストとリンクを処理
                        note_parts = []
                        for element in note_cell.stripped_strings:
                            note_parts.append(str(element))  # 確実に文字列に変換
                        note = " ".join(note_parts)
                
                if item and category:  # 空の行は除外
                    info = {
                        "品目": str(item),  # 確実に文字列に変換
                        "分別区分": str(category),  # 確実に文字列に変換
                        "手数料": str(fee),  # 確実に文字列に変換
                        "備考": str(note) if note and note != "&nbsp;" else ""  # 確実に文字列に変換
                    }
                    garbage_info.append(info)
    
    # ゴミ分別情報が見つからない場合は、従来の方法でテキストを抽出
    if not garbage_info:
        return soup.get_text()
    
    # JSONファイルに保存（UTF-8でエンコード）
    with open('garbage_data.json', 'w', encoding='utf-8') as f:
        json.dump(garbage_info, f, ensure_ascii=False, indent=2)
    
    # テキスト形式でも保存（デバッグ用）
    with open('garbage_data.txt', 'w', encoding='utf-8') as f:
        for info in garbage_info:
            f.write(f"品目: {info['品目']} | 分別区分: {info['分別区分']} | 手数料: {info['手数料']}")
            if info['備考']:
                f.write(f" | 備考: {info['備考']}")
            f.write('\n')
    
    return '\n'.join([f"品目: {info['品目']} | 分別区分: {info['分別区分']} | 手数料: {info['手数料']}" + 
                     (f" | 備考: {info['備考']}" if info['備考'] else "") for info in garbage_info])

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
