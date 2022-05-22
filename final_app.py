# The code below titled final_app.py is Eryn Yuasa's capstone project for data science 2.

# Please contact Eryn Yuasa at eyuasa1@jhu.edu for any questions regarding the code
# or application usage. 

###############################################################

# Load libraries
# for visualizations and calculations 
from dash import Dash, dcc, html, Input, Output, dash_table
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
# for text pre-processing
import re, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
#for model-building
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
# bag of words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

###############################################################

# Reading in and cleaning data
# Read in the first 20,000 rows from the Flair_Data.csv file 
df = pd.read_csv("Flair_Data.csv", nrows=20000)
# Keep only the needed columns 
df_2 = df[["link_flair_text", "id", "author",  "question", "year"]]
# Group dataframe for figure 1 
df_2_grouped = df_2.groupby(df_2["link_flair_text"]).count().reset_index()
# Using px, create a bar chart for the overview of the dataset 
fig_overall = px.bar(df_2_grouped, x='link_flair_text', y='id', title = "Number of Post in Each Category <br><sup>N = 20,000 Posts</sup>",
            labels = {'x':"Flair of Post", 
              'y':'Number of Posts'})
# For dash application, find the flair values that have greater than or equal to
# 100 posts 
option_df = df_2_grouped[df_2_grouped["id"] >= 100]
option_text = option_df["link_flair_text"].unique()
# Preprocessing data:convert to lowercase, strip and remove punctuations
# using the re package 
def preprocess(text):
    text = text.lower() 
    text=text.strip()  
    text=re.compile('<.*?>').sub('', text) 
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)  
    text = re.sub('\s+', ' ', text)  
    text = re.sub(r'\[[0-9]*\]',' ',text) 
    text=re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d',' ',text) 
    text = re.sub(r'\s+',' ',text) 
    return text
 
# Run the pre-processing process for question text 
def finalpreprocess(string):
    return (preprocess(string))

###############################################################

# Create layout of dash application 
app = Dash(__name__)
app.title ='Final Application Eryn Yuasa'

app.layout = html.Div([
html.H1("Reddit r/AskScience Flair Classification Modeling"),
    html.H2("Eryn Yuasa Final Project"),
    html.H2("This app takes 20,000 Reddit posts from the r/AskScience subreddit and performs two different classification models using Natural Language Processing to predict the associated flair value of questions."),
    html.P("The Reddit r/AskScience Flair dataset can be found here: https://data.mendeley.com/datasets/k9r2d9z999/1. The r/AskScience dataset originally contains 519,054 datapoints; however, only the first 20,000 are used for the purpose of this analysis. On this dash web based application, you can select two different Flair values. Flairs are used to describe the posts in the subreddit. After selecting two different flair values, both a logistic regression - tdif model and a naive bayes model will be run to analyze how well the models are able to classify and predict flair values from the cleaned question text originally asked in the reddit post. The github repo for this project is linked here: https://github.com/ds4ph-bme/capstone-project-eyuasa."), 
    html.H2("An overview graph of the count of posts and flair values is shown below"),
dcc.Graph(id='full_graph', figure = fig_overall),
    html.H2("View a data table of the 20,000 Reddit Posts below"),
    html.P("Note: Filtering and sorting through the dash data table is possible."),
    dash_table.DataTable(df_2.to_dict('records'),[{"name": i, "id": i} for i in df_2.columns], id='tbl',     
        filter_action="native",
        sort_action="native",
        sort_mode="multi", page_size= 10),
    html.H2("Model and Model Results"),
    html.P("Instructions: In order to run this analysis and correctly model the data, we'll be looking only at data flairs that have 100 or more posts written about them. Choose one flair classification from each of the dropdown bars below to view the AUC model results from both the Logistic Regression (tf-idf) model and the Naive Bayes Model. Do not select the same flair twice as this will cause the models to not work. The models use a 30/70 test/training split."),
    html.P("Select a first classification from the list below:"), 
    dcc.Dropdown(option_text, 'Physics', id='demo-dropdown1'),
    html.P("Select a second classification (cannot be the same as the first) from the list below:"), 
    dcc.Dropdown(option_text, 'Astronomy', id='demo-dropdown'),
    html.H2("Logistic Regression (tf-idf) Model"),
    dcc.Graph(id='log_auc'), 
    html.Div(id='auc_log_results'),
    html.H2("Naive Bayes Model"),
    dcc.Graph(id='bayes_auc'),
    html.Div(id='auc_bayes_results'),
])

###############################################################

# Create dash app callbacks

# Create AUC graph and results from the two dropdown components of flair values chosen
# Logistic Regression Model 
@app.callback(
[Output('log_auc', 'figure'),
 Output('auc_log_results', 'children')],
[Input(component_id = 'demo-dropdown', component_property = 'value'),
 Input(component_id = 'demo-dropdown1', component_property = 'value'),
])
def update_figure_auc(second_option, first_option):
    # Filter dataframe to be only the two flair values selected 
    df_3 = df_2[(df_2["link_flair_text"] == first_option) | (df_2["link_flair_text"] == second_option)]
    # Factorize the flair number in order to correctly run the AUC 
    df_3['Flair_Number'] = pd.factorize(df_3.link_flair_text)[0]   
    # Run the preprocessing step on the question text 
    df_3['Clean_Text'] = df_3['question'].apply(lambda x: finalpreprocess(x))
    df_4 = df_3
    # Create a training and test set for labeled and unlabeleld sections using 50/50 labeled/unlabelled split 
    training_num = round(len(df_4) / 2)
    training_num_one = training_num + 1
    df_4_labeled = df_4.iloc[0:training_num,]
    df_4_unlabeled =  df_4.iloc[training_num_one:,]
    # Create training and test for the labeled dataset using a 30/70 test/training split 
    X_train, X_test, y_train, y_test = train_test_split(df_4_labeled["Clean_Text"],
                                                        df_4_labeled["Flair_Number"],test_size=0.3,shuffle=True)
    # Apply W2V
    # Tokenizes each word in the X_train and X_test dataset 
    X_train_tok= [nltk.word_tokenize(i) for i in X_train]  
    X_test_tok= [nltk.word_tokenize(i) for i in X_test]
    # Convert vectors into Tf-Idf objects 
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)
    X_train_vectors_tfidf = tfidf_vectorizer.fit_transform(X_train) 
    X_test_vectors_tfidf = tfidf_vectorizer.transform(X_test)
    # Running Logistic Regression Using tf-idf
    # Create model 
    lr_tfidf=LogisticRegression(solver = 'liblinear', C=10, penalty = 'l2')
    # Fit model on training vectors for X and Y 
    lr_tfidf.fit(X_train_vectors_tfidf, y_train)  
    # Predict y value for test dataset
    y_predict = lr_tfidf.predict(X_test_vectors_tfidf)
    # Get probabilities of y_dataset 
    y_prob = lr_tfidf.predict_proba(X_test_vectors_tfidf)[:,1]
    # Get a report of the dataset 
    report_log = classification_report(y_test,y_predict)
    # Run AUC 
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    # Create AUC figure using px.area
    fig_auc_log = px.area(
        x=fpr, y=tpr,
        title=f'Logistic Regression Model tf-idf: ROC Curve (AUC={roc_auc:.4f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=700, height=500
    )
    fig_auc_log.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    fig_auc_log.update_yaxes(scaleanchor="x", scaleratio=1)
    fig_auc_log.update_xaxes(constrain='domain')
    fig_auc_log.show()
    return (fig_auc_log, report_log)

# Create AUC graph and results from the two dropdown components of flair values chosen
# Bayes model
@app.callback(
[Output('bayes_auc', 'figure'),
Output('auc_bayes_results', 'children')],
[Input(component_id = 'demo-dropdown1', component_property = 'value'),
Input(component_id = 'demo-dropdown', component_property = 'value')]
)
def update_figure_auc(second_option, first_option):
    # Filter dataframe to be only the two flair values selected 
    df_3 = df_2[(df_2["link_flair_text"] == first_option) | (df_2["link_flair_text"] == second_option)]
    # Factorize the flair number in order to correctly run the AUC 
    df_3['Flair_Number'] = pd.factorize(df_3.link_flair_text)[0]   
    # Run the preprocessing step on the question text 
    df_3['Clean_Text'] = df_3['question'].apply(lambda x: finalpreprocess(x))
    df_4 = df_3
    # Create a training and test set for labeled and unlabeleld sections using 50/50 labeled/unlabelled split 
    training_num = round(len(df_4) / 2)
    training_num_one = training_num + 1
    df_4_labeled = df_4.iloc[0:training_num,]
    df_4_unlabeled =  df_4.iloc[training_num_one:,]
    # Create training and test for the labeled dataset using a 30/70 test/training split 
    X_train, X_test, y_train, y_test = train_test_split(df_4_labeled["Clean_Text"],                df_4_labeled["Flair_Number"],test_size=0.3,shuffle=True)
    # Apply W2V
    # Tokenizes each word in the X_train and X_test dataset 
    X_train_tok= [nltk.word_tokenize(i) for i in X_train]  
    X_test_tok= [nltk.word_tokenize(i) for i in X_test]
    # Convert vectors into Tf-Idf objects 
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)
    X_train_vectors_tfidf = tfidf_vectorizer.fit_transform(X_train) 
    X_test_vectors_tfidf = tfidf_vectorizer.transform(X_test)
    # Running Naive Bayes Classification 
    # Create model 
    # Fit model on training vectors for X and Y 
    nb_tfidf = MultinomialNB()
    nb_tfidf.fit(X_train_vectors_tfidf, y_train)  
    # Predict y value for test dataset
    y_predict = nb_tfidf.predict(X_test_vectors_tfidf)
    # Get probabilities of y_dataset 
    y_prob = nb_tfidf.predict_proba(X_test_vectors_tfidf)[:,1]
    # Get a report of the dataset 
    bayes_report = (classification_report(y_test,y_predict))
    fpr_bayes, tpr_bayes, thresholds_bayes = roc_curve(y_test, y_prob)
    # Run AUC 
    roc_auc_bayes = auc(fpr_bayes, tpr_bayes)
    # Create AUC figure using px.area
    fig_auc_log_bayes = px.area(
        x=fpr_bayes, y=tpr_bayes,
        title=f'Naive Bayes Model: ROC Curve (AUC={roc_auc_bayes:.4f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=700, height=500
    )
    fig_auc_log_bayes.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig_auc_log_bayes.update_yaxes(scaleanchor="x", scaleratio=1)
    fig_auc_log_bayes.update_xaxes(constrain='domain')
    fig_auc_log_bayes.show()
    return (fig_auc_log_bayes, bayes_report)

###############################################################

# Run application on class server on port 1090
if __name__ == '__main__':
    app.run_server(host='jupyter.biostat.jhsph.edu', port= os.getuid() + 31, debug=True)