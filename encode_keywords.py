import time
import torch
import json
import os
import numpy as np
import scipy.io as sio
import argparse

import gensim.downloader as api
import pickle
import argparse

from transformers import GPT2Tokenizer
os.environ['GENSIM_DATA_DIR']='./gensim-data'

word_embedding = {
    'glove': "glove-wiki-gigaword-300",
    'word2vec': "word2vec-google-news-300"
}
CACHE_DIR='/mnt/data0/tuhq21/.cache/torch/transformers'


def create_enc_dict(folder_name, file_name, embedding, task):
    embedding_file = word_embedding[embedding]
    # if task == 'key2article':
    #     folder_name = file_name
    # else:
    #     folder_name = os.path.dirname(file_name)
    file_name = os.path.join(folder_name, file_name)

    print('file_name: ', file_name)
    print('folder_name: ', folder_name)
    print('word_embedding: ', embedding)

    ######## Load word embedding data
    print('{} word embeddings loading...'.format(embedding))
    encoder = api.load(embedding_file)
    print('{} word embeddings loaded'.format(embedding))
    glove_dict = {}

    if task == 'key2article':
        file1 = open(file_name, "r+")
        lines = file1.readlines()

        i = 0
        for line in lines:
            keywords = list(line.strip().split(", "))
            print(keywords)
            for word in keywords:
                glove_dict[word] = encoder[word]

            # save_path = folder_name + '/' + str(embedding) + '_set_' +str(i) + '.npy'
            # np.save(save_path, glove_words)
            i = i + 1
    elif task == "obj2caption":
        keyword_sets = set()
        file1 = json.load(open(file_name))
        feas = {}
        for key in file1.keys():
            keywords = list(file1[key])
            for keyword in keywords:
                keyword_sets.add(keyword)
        for word in list(keyword_sets):
            word_sp = word.split(" ")
            if len(word_sp) > 1:
                encoded = encoder[word_sp[0]]
                for item in word_sp[1:]:
                    encoded = encoded + encoder[item]
                encoded = encoded / len(word_sp)
            else:
                encoded = encoder[word]
            glove_dict[word] = encoded
        for key in file1.keys():
            tmp = []
            keywords = list(file1[key])
            for word in keywords:
                tmp.append(glove_dict[word])
            feas[key] = tmp
    else:
        keyword_sets = []
        for filename in os.listdir(folder_name):
            if filename.endswith('txt'):
                file1 = open(folder_name + filename, "r+")
                lines = file1.readlines()
                keywords = list(lines[2].strip().split(", "))
                in_text = lines[1].split()[:30]
                keyword_sets.append((' '.join(in_text), keywords))
                for word in keywords:
                    glove_dict[word] = encoder[word]

    save_path_dict = folder_name + '/dict_visnews_obj_' + str(embedding) + '.pkl'
    with open(save_path_dict, 'wb') as f:
        pickle.dump(glove_dict, f)

    if task == "obj2caption":
        save_path_arr = folder_name + '/dict_line_visnews_obj_' + str(embedding) + '.pkl'
        with open(save_path_arr, 'wb') as f:
            pickle.dump(feas, f)
def checker(string):
    string = string.replace("'ve", '')
    string = string.replace("@", '')
    string = string.replace("'re", '')
    string = string.replace("'d", '')
    string = string.replace("?", '')
    string = string.replace("'s", '')
    string = string.replace(":", '')
    string = string.replace("!", '')
    string = string.replace('"', '')
    string = string.replace(".", '')
    string = string.replace("--", '')
    string = string.replace("'", '')
    string = string.replace(",", '')
    string = string.replace(';', '')
    string = string.replace('‘', '')
    string = string.replace('(', '')
    string = string.replace(')', '')
    string = string.replace('\'', '')
    string = string.replace(' ', '')
    return(string)

def converter_table_glove(gpt_version):
    import gensim.downloader as api
    glove_encoder = api.load("glove-wiki-gigaword-300")

    path = 'npy_data/converter_table_glove'

    # load gpt-2 model
    tokenizer = GPT2Tokenizer.from_pretrained(gpt_version, cache_dir=CACHE_DIR)
    sos_token, pad_token = r'<-start_of_text->', r'<-pad->'
    tokenizer.add_tokens([sos_token])
    tokenizer.add_tokens([pad_token])
    print(f"Tokenizer Vocab Size: {tokenizer.vocab_size}.")

    holder = np.zeros((tokenizer.vocab_size, 300))

    # translate every word from the gpt-2 space into a glove representation
    for i in range(tokenizer.vocab_size):
        try:
            word = tokenizer.decode([i])
            word = checker(word.strip().lower())
            glove = glove_encoder[word]
            holder[i, :] = glove
        except:
            word = tokenizer.decode([i])
            holder[i, :] = np.zeros((300))  # + 500

    # Save all 50'000 glove representations of the gpt-2 words
    np.save(file=path, arr=holder)
    print('Table was generated')


def converter_table_word2vec(gpt_version):
    import gensim.downloader as api
    word2vec_encoder = api.load("word2vec-google-news-300")

    path = 'npy_data/converter_table_word2vec'

    # load gpt-2 model
    tokenizer = GPT2Tokenizer.from_pretrained(gpt_version, cache_dir=CACHE_DIR)
    sos_token, pad_token = r'<-start_of_text->', r'<-pad->'
    tokenizer.add_tokens([sos_token])
    tokenizer.add_tokens([pad_token])

    holder = np.zeros((tokenizer.vocab_size, 300))

    # translate every word from the gpt-2 space into a word2vec representation
    for i in range(tokenizer.vocab_size):
        try:
            word = tokenizer.decode([i])
            word = checker(word.strip().lower())
            word2vec = word2vec_encoder[word]
            holder[i, :] = word2vec
        except:
            word = tokenizer.decode([i])
            holder[i, :] = np.zeros((300))  # + 500

    # Save all 50'000 word2vec representations of the gpt-2 words
    np.save(file=path, arr=holder)
    print('Table was generated')

# if encode_articles == True:

#     for n in [4, 5, 8, 9, 10, 12, 13, 14, 15, 16]:
#         print(n)
#         file1 = open(str(os.path.dirname(os.path.abspath(__file__))) +
#                      "/data/keyword_to_articles/test_" + str(n) + ".txt", "r+")

#         lines = file1.readlines()

#         keywords = list(lines[2].strip().split(", "))
#         print(keywords)
#         glove_words = []
#         for word in keywords:
#             glove = encoder[word]
#             glove_words.append(glove)

#         save_path = str(os.path.dirname(
#             os.path.abspath(__file__))) + '/data/keyword_to_articles/test_' +str(n) + '.npy'
#         np.save(save_path, glove_words)

if __name__ == "__main__":
    # converter_table_glove("cambridgeltl/magic_mscoco")
    # converter_table_word2vec("cambridgeltl/magic_mscoco")

    ######## Parse arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-file', default='test_image_obj.json', type=str)
    # parser.add_argument('-folder_name', default='/data/tuhq/multimodal/flickr30k/ctg-data', type=str)
    # parser.add_argument('-word_embedding', type=str, default='glove',
    #                     choices=list(word_embedding.keys()), help='word_embedding')
    # parser.add_argument('-task', type=str, default='obj2caption')  # 'key2article', 'commongen'
    # args = parser.parse_args()
    file_name = "test_image_obj.json"
    folder_name = "/mnt/data0/tuhq21/dataset/visnews/origin"
    embedding = 'glove'
    task = 'obj2caption'

    create_enc_dict(folder_name, file_name, embedding, task)

    # a = json.load(open("/data/tuhq/multimodal/coco14/test_image_obj.json"))



