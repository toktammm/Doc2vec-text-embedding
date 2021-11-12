"""
This code uses the Doc2vev model from Gensim to create document embeddings. 
A sample visualization is also provided which is achieved using PCA and t-sne methods.
"""

import pandas as pd
import numpy as np
import re
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import requests
import nltk
nltk.download('all')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import multiprocessing
import sklearn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def data_request(df):
    #set up headers for API requests
    headers = {'token': "TOKEN", 'Content-Type': "application/json"}

    # create list to save JSON card objects
    card_objects = []

    cardIds = df["kpi_id"]
    n=100
    cards = [cardIds[i * n:(i + 1) * n] for i in range((len(cardIds) + n - 1) // n )]

    # query card API to get card objects
    for card_slice in cards:
      cards_string = ','.join(card_slice)
      response = requests.get('URL TO DATA='+cards_string, headers=headers)
      try:
        assert response.status_code == 200
        card_objects = card_objects + response.json()
      except:
        continue
        
    return cardIds, card_objects
 

def get_card_data(cardId, headers=headers):
    response = requests.get(f'URL TO DATA', headers=headers)
    assert response.status_code == 200
    json_response = response.json()
    return json_response


def clean_text(text):
    if type(text) != str:
        text =''
    text = text.lower()
    text = re.split('[^a-zA-Z]+', text) # split the text by every non-alphabet character
    text = [w for w in text if w not in stopwords.words('english')]
    text = ' '.join(text)
    text = re.sub(' +', ' ', text)
    return text
    
    
def remove_duplicates(text):
    return ' '.join(list(set(text.split())))
    
    
def prepare_data(cards, ids, card_obj):
    card_data_text = []
    
    for id, card in zip(ids, cards):
        try:
            summary_label = card['summary']['label']
        except:                  # this field doesn't exists for this card 
            summary_label = ''   # because the type of this field is str, I used an empty str when this field doesn't exists
        try:
            summary_column = card['summary']['data']['columns']
        except:
            summary_column = []  # because the type of this field is list, I used an empty list when this field doesn't exists
        try:
            summary_aliases = card['summary']['data']['aliases']
        except:
            summary_aliases = [] # empty list if field doesn't exists
        try:
            data_columns = card['data']['columns']
        except:
            data_columns = []    # empty list if field doesn't exists
        card_data_text.append((id, summary_label, ' '.join(summary_column + summary_aliases + data_columns)))
    
    card_data_attr = pd.DataFrame(card_data_text, columns=['kpi_id', 'summary_label', 'data_columns'])
    card_data_attr = pd.merge(card_obj, card_data_attr, on='kpi_id') # merging card objects and data based on kpi ids
    
    for data in ['description', 'title', 'summary_label', 'data_columns']:
        card_data_attr[data] = card_data_attr[data].apply(clean_text) # cleaning the text for these fields
    
    # as these fields might not exists for many cards, combining the text will increase the size of context data
    combined_fields = card_data_attr['title'] + ' ' + card_data_attr['summary_label'] + ' ' + card_data_attr['data_columns']
    combined_fields = combined_fields.apply(remove_duplicates)  # removing the repetitive words in title, label, and columns 

    card_data_text = card_data_attr['description'] + ' ' + combined_fields 
    return card_data_text, card_data_attr
    
    
def tsne_plot(model, df, tags_cat, N_size, color, n_dim):
    fig = plt.figure()
    fig.set_size_inches(10, 10)
    if n_dim == 3:
        ax = fig.add_subplot(projection='3d')
    else:
        ax = fig.add_subplot()
    
    for idx, doc_tags in enumerate(tags_cat):
        X = model[doc_tags]
        pca = PCA(n_components=10)
        X = pca.fit_transform(X)
        X = sklearn.preprocessing.normalize(X, norm='l2', axis=1, copy=True)
        tsne = TSNE(n_components=n_dim, perplexity=10, learning_rate=20.0, n_iter=2000, init='pca', n_jobs=10)
        X_tsne = tsne.fit_transform(X)

        idx_ls = np.random.choice(X_tsne.shape[0], N_size, replace=False) # randomly select N_size vectors to plot
        point_labels = list(df.iloc[idx_ls]['title'])

        xs = X_tsne[idx_ls, 0]
        ys = X_tsne[idx_ls, 1]
        if n_dim == 3:
            zs = X_tsne[idx_ls, 2]
            ax.scatter(xs, ys, zs, c=color[idx], linewidth=5) 
        else:
            ax.scatter(xs, ys, c=color[idx], linewidth=5) 

        label_idx = np.random.choice(list(range(N_size)), size=N_size//5, replace=False)    # randomly selecting 1/5 of idx for the labels to annotate points
        point_labels = list(df.iloc[idx_ls]['title'])              # using the card titles to annotate the points
        point_labels = ['' if i not in label_idx else ' '.join(point_labels[i].split()[:2]) for i in range(N_size)]

        if n_dim == 3:
            for x, y, z, label in zip(xs, ys, zs, point_labels):
                if len(label):
                    label = ' '.join(label.split()[0:3])
                    ax.text(x, y, z, label)
        else: 
            for x, y, label in zip(xs, ys, point_labels):
                if len(label):
                    label = ' '.join(label.split()[0:3])
                    ax.text(x, y, label) 
            
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    if n_dim == 3:
        ax.set_zlabel('Z Label')

    plt.title('Doc2vec Embeddings of Cards', fontweight="bold")
    plt.show()

    
def main():
    # Download data: in our dataset, each data sample is a card that has description and title, and the content of card can be accessed using its id
    df_all_cards = pd.read_csv("data_cards.csv")
    df_all_cards.kpi_id = df_all_cards.kpi_id.apply(lambda x: str(x)) # change type of kpi ids to string
    cardIds, card_objects = data_request(df_all_cards)

    # Filtering the attributes to only include text content, in our data these are "description" and "title" fields
    card_objects_df = pd.DataFrame(card_objects)[['id', 'description', 'title']].rename(columns={'id':'kpi_id'})
    card_objects_df.kpi_id = card_objects_df.kpi_id.apply(lambda x: str(x))
    
    # Downloading the content of cards associated with each card_id
    cards_data = []
    card_ids_data = []
    for card_id in cardIds:
        try:
            cards_data.append(get_card_data(card_id))
            card_ids_data.append(card_id)
        except:
            continue
    
    train_data, card_objects_df = prepare_data(cards_data, card_ids_data, card_objects_df)

    # Here I use nltk word tokenizer to prepare the training data for Doc2vec. The tag identifiers are df indices
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(id)]) for i, (id, _d) in enumerate(zip(range(len(card_objects_df)), train_data))]

    # Gensim Doc2vec model and parameters
    cores = multiprocessing.cpu_count() - 1
    size = 100          # dimension of our document embeddings
    context_window = 5  # for larger data and lengthy text, choose a bigger context_window size
    seed = 42
    min_count = 20  # ignore the tokens that have been appeared less than 20 times in all data
    alpha = 0.1     # initial learning rate
    epochs = 1000   

    # Building the Model
    model = Doc2Vec(documents = tagged_data, 
    alpha = alpha, # initial learning rate
    seed = seed,
    min_count = min_count, # ignore words with freq less than min_count
    max_vocab_size = None,  
    window = context_window, # n_gram context window
    vector_size = size, # feature vector size
    sample = 1e-4, 
    negative = 50,      # negative sampling
    ns_exponent = 1.,
    workers = cores,
    epochs = epochs)
    
    # Results
    doc_tags = list(model.docvecs.key_to_index)
    X = model[doc_tags]
    
    # Choosing 6 random vectors for visualization
    idxs = np.random.randint(0, len(df_all_cards), size=6)
    tags_cat = []
    for idx in idxs:
        tokens = train_data[idx].split()

        new_vector = model.infer_vector(tokens, epochs=1000)
        sims = model.docvecs.similar_by_vector(new_vector,topn=20)
        doc_tags = [sims[i][0] for i in range(len(sims))]

        tags_cat.append(doc_tags)
    
    # Plotting top 10 most similar vector embeddings for each idxs
    colors = np.random.rand(len(idxs), 3)
    tsne_plot(model, card_objects_df, tags_cat, 10, colors, 2)

    
main()
