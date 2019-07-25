import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

def plot_TSNE(data, feature_to_color = None, *params):
    tsne = TSNE(random_state=17)
    tsne_representation = tsne.fit_transform(data)
    if feature_to_color:
        plt.scatter(tsne_representation[:, 0], tsne_representation[:, 1], 
            c=data[feature_to_color].map({0: 'blue', 1: 'orange'}))
    else:
        plt.scatter(tsne_representation[:, 0], tsne_representation[:, 1])
    
    return tsne_representation

def plot_feature_distribution(data, feature, title = 'distribution of feature in dataset'):
    plt.figure(figsize = (12 , 6))
    plt.title(title)
    sns.distplot(data.loc[~data[feature].isnull() , feature] , kde = True , hist = False , bins = 20 , label = feature)
    plt.xlabel('')
    plt.legend()
    plt.show()

def plot_feature_countplot(data, feature, title = 'distibution of feature in dataset', size=4):
    f , ax = plt.subplots(1, 1, figsize = (4*size , 4))
    total =float(len(data))
    g = sns.countplot(data[feature], order = data[feature].value_counts().index[:20], palette='Set3')
    g.set_title("Number and percentage of {}".format(title))
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(100*height/total),
                ha="center") 
    plt.show()

def plot_crosstab(data, feature1, feature2, normalize=True):
    pd.crosstab(data[feature1], data[feature2], normalize=normalize)

def plot_boxplot(data, feature1, feature2):
    '''
    params: data - pd.DataFrame with data
            type(feature1, feature2) - string
    '''

    sns.boxplot(y=feature1, x=feature2, data=data, orient='h')

def plot_pairplot(data, cols):
    '''
    params:
        cols - list of columns needed for selecting in data
    '''
    sns.pairplot(data[cols])

def plot_heatmap(data, cat_feature1, cat_feature2, num_feature, aggfunc=sum):
    pivot_table = data.pivot_table(
                        index=cat_feature1,
                        columns=cat_feature2, 
                        values=num_feature, 
                        aggfunc=aggfunc).fillna(0).applymap(float)
    sns.heatmap(pivot_table, annot=True, fmt=".1f", linewidths=.5)

def plot_corrmatr(data, cols):
    sns.heatmap(data[cols].corr())

stopwords = set(STOPWORDS)
def show_wordcloud(data , title = None):
    wordcloud = WordCloud(
        background_color = 'white',
        stopwords = stopwords,
        max_words = 50,
        max_font_size = 40,
        scale = 5,
        random_state = 1
    ).generate(str(data))
    
    fig = plt.figure(1 , figsize = (10 , 10))
    plt.axis('off')
    if(title):
        fig.suptitle(title , fontsize = 20)
        fig.subplots_adjust(top = 2.3)
    
    plt.imshow(wordcloud)
    plt.show()


