import jieba
import numpy as np
from snownlp import SnowNLP
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re

def process_news_data(news_list):
   
    if not news_list:
        return {"error": "无新闻数据"}

    clean_texts = []
    valid_news = []
    for news in news_list:
        text = news["title"] + " " + news["content"]
      
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 10:
            clean_texts.append(text)
            valid_news.append(news)

    if not clean_texts:
        return {"error": "新闻文本清洗后无有效数据"}


    sentiments = []
    for text in clean_texts:
      
        score = SnowNLP(text).sentiments
        normalized_score = score * 2 - 1
        sentiments.append(normalized_score)

    sentiment_result = {
        "sentiment_mean": round(np.mean(sentiments), 4),
        "sentiment_volatility": round(np.std(sentiments), 4),
        "positive_ratio": round((np.array(sentiments) > 0.2).mean(), 4),
        "negative_ratio": round((np.array(sentiments) < -0.2).mean(), 4),
        "neutral_ratio": round((np.abs(np.array(sentiments)) <= 0.2).mean(), 4)
    }

    topic_result = {}
    try:
        
        texts_cut = [" ".join(jieba.cut(text)) for text in clean_texts]
      
        vectorizer = CountVectorizer(
            max_df=0.85,
            min_df=2,
            stop_words=["的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这"]
        )
        text_matrix = vectorizer.fit_transform(texts_cut)
     
        lda = LatentDirichletAllocation(n_components=5, random_state=42, max_iter=100)
        lda.fit(text_matrix)
       
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for idx, topic in enumerate(lda.components_):
            top_keywords = [feature_names[i] for i in topic.argsort()[-10:][::-1]]
            topic_weight = np.sum(topic) / np.sum(lda.components_)
            topics.append({
                "topic_id": idx + 1,
                "keywords": top_keywords,
                "weight": round(topic_weight, 4)
            })
        topic_result = {
            "topic_count": 5,
            "topics": topics,
            "topic_distribution": lda.transform(text_matrix).mean(axis=0).tolist()
        }
    except Exception as e:
        topic_result = {"error": f"主题分析失败：{str(e)}"}

 
    return {
        "news_count": len(valid_news),
        "news_list": valid_news[:10],  # 返回前10条新闻
        "sentiment_analysis": sentiment_result,
        "lda_topic_analysis": topic_result
    }