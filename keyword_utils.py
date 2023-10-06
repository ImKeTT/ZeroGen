import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
import re, json
from tqdm import tqdm
import csv

def run(corpus):
    corpus = [text.replace('.', ' ') for text in corpus]
    corpus = [re.sub(r'\s+', ' ', re.sub(r'[^\w \s]', '', text)).lower() for text in corpus]
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(corpus)
    names = vectorizer.get_feature_names()
    data = vectors.todense().tolist()
    # Create a dataframe with the results
    df = pd.DataFrame(data, columns=names)

    st = set(stopwords.words('english'))
    # remove all columns containing a stop word from the resultant dataframe.
    df = df[filter(lambda x: x not in list(st), df.columns)]

    N = 10
    out = []
    for i in df.iterrows():
        # print(i[1])
        out.append(list(i[1].sort_values(ascending=False)[:N].keys()))
    return out

def process_keywords(topic_list):
    word = {}
    for item in topic_list:
        for ii in item:
            if ii in word.keys():
                word[ii] += 1
            else:
                word[ii] = 1
    word = sorted(word.items(), key=lambda x: x[1], reverse=True)
    return word


if __name__ == '__main__':
    # topics = json.load(open("/mnt/data0/tuhq21/dataset/visnews/agnews/news_topics.json"))
    # topic_tmp = {}
    # for key in topics.keys():
    #     topic_tmp[key] = []
    #     tmp1 = []
    #     tmp = process_keywords(topics[key])[:100]
    #     for i in tmp:
    #         tmp1.append(i[0])
    #     topic_tmp[key] = tmp1
    # with open("/mnt/data0/tuhq21/dataset/visnews/agnews/news_topics_compressed.json", 'w') as f:
    #     json.dump(topic_tmp, f, indent=4)

    ## get style flikr10k topic words from roberta classified high-quality sentences
    # topics = {}
    # ## ['romantic': 0, 'humor': 1]
    # topics = {}
    # idxs = json.load(open("/mnt/data0/tuhq21/dataset/FlickrStyle/representative_style_captions.json"))
    # captions = json.load(open("/mnt/data0/tuhq21/dataset/FlickrStyle/train_classification.json"))
    # selected_caps = {}
    # for key in idxs.keys():
    #     tmp = []
    #     for idx in idxs[key]:
    #         tmp.append(captions[key][int(idx)])
    #     selected_caps[key] = tmp
    # topics['romantic'] = run(selected_caps['romantic'])
    # topics['humor'] = run(selected_caps['humor'])
    # with open("/mnt/data0/tuhq21/dataset/FlickrStyle/topics_filtered_small.json", 'w') as f:
    #     json.dump(topics, f, indent=4)

    # get.append(" ".join(tmp1))
    # get.append(" ".join(tmp2))
    # topics['romantic'] = run(tmp1)
    # topics['humor'] = run(tmp2)
    # with open("/mnt/data0/tuhq21/dataset/FlickrStyle/topic_words_finegrained.json", 'w') as f:
    #     json.dump(topics, f, indent=4)

    # topics = json.load(open("/mnt/data0/tuhq21/dataset/FlickrStyle/topics_filtered_small.json"))
    # topic_tmp = {}
    # for key in topics.keys():
    #     topic_tmp[key] = []
    #     tmp1 = []
    #     tmp = process_keywords(topics[key])[:100]
    #     for i in tmp:
    #         tmp1.append(i[0])
    #     topic_tmp[key] = tmp1
    # with open("/mnt/data0/tuhq21/dataset/FlickrStyle/topics_filtered_small_filtered.json", 'w') as f:
    #     json.dump(topic_tmp, f, indent=4)

    # news = json.load(open("/mnt/data0/tuhq21/dataset/visnews/agnews/news.json"))
    # indexes = json.load(open("/mnt/data0/tuhq21/dataset/visnews/agnews/representative_news_index.json"))
    # topics = {}
    # print(indexes.keys(), news.keys())
    #
    # topics_news = []
    # for topic in tqdm(indexes.keys()):
    #     news_list = []
    #     for item in indexes[topic][:500]:
    #         news_list.append(news[topic][item])
    #     current_news = " ".join(news_list)
    #     topics_news.append(current_news)
    # topics["topics"] = run(topics_news)
    # with open("/mnt/data0/tuhq21/dataset/visnews/agnews/news_topics_unified.json", 'w') as f:
    #     json.dump(topics, f, indent=4)

    # for topic in tqdm(indexes.keys()):
    #     news_list = []
    #     for item in indexes[topic]:
    #         news_list.append(news[topic][item])
    #     topics[topic] = run(news_list)
    # with open("/mnt/data0/tuhq21/dataset/visnews/agnews/news_topics.json", 'w') as f:
    #     json.dump(topics, f, indent=4)

    # with open("/mnt/data0/tuhq21/dataset/visnews/agnews/train.tsv", 'r') as f:
    #     data = f.readlines()
    # data = [item.strip() for item in data]
    # with open("/mnt/data0/tuhq21/dataset/visnews/agnews/dev.tsv", 'r') as f:
    #     data1 = f.readlines()
    # data1 = [item.strip() for item in data1]
    # data.extend(data1)
    #
    # class_file = {}
    # class_file['world'] = []
    # class_file['sports'] = []
    # class_file['business'] = []
    # class_file['science'] = []
    #
    # for line in data:
    #     line = line.split("\t")
    #     label = int(line[0])
    #     if label==1:
    #         class_file['world'].append(line[-1])
    #     elif label == 2:
    #         class_file['sports'].append(line[-1])
    #     elif label == 3:
    #         class_file['business'].append(line[-1])
    #     elif label == 4:
    #         class_file['science'].append(line[-1])
    # with open("/mnt/data0/tuhq21/dataset/visnews/agnews/news.json", 'w') as f:
    #     json.dump(class_file, f, indent=4)

    # csv.writer(open("/mnt/data0/tuhq21/dataset/visnews/agnews/dev.tsv"


    ## process news data
    """
    id 39136
    caption Candace Pickens and her son Zachaeus
    topic law_crime
    source washington_post
    image_path ./washington_post/images/0376/501.jpg
    article_path ./washington_post/articles/39136.txt
    """

    # relations = json.load(open("/mnt/data0/tuhq21/dataset/visnews/origin/data.json"))
    #
    # topics = set()
    # news = {}
    # news['science'] = []
    # news['business'] = []
    # news['world'] = []
    # news['sports'] = []
    # for ii in relations:
    #     if ii['topic'] == "science":
    #         news['science'].append(ii)
    #     elif ii['topic'] == "business":
    #         news['business'].append(ii)
    #     elif ii['topic'] == "world":
    #         news['world'].append(ii)
    #     elif ii['topic'] == "sports":
    #         news['sports'].append(ii)
    # for ii in news.keys():
    #     print(ii, len(news[ii]))
    # with open("/mnt/data0/tuhq21/dataset/visnews/origin/4topic_data.json", 'w') as f:
    #     json.dump(news, f, indent=4)

