# Capstone Project: Building Classification Models for Predicting Flairs of Reddit r/AskScience posts for Eryn Yuasa

## Project for Johns Hopkins Bloomberg School of Public Health 140.629.71 - Data Science for Public Health / Biomedical Applications

## Assignment
"The project is to create a web app based on your research or interests that performs a prediction algorithm or some other non-trivial data calculation. The prediction algorithm can be based on any model (though should be more complex than straight arithmetic, like a BMI calculator). In addition, your readme.md file in your github repo should be a brief writeup of your project (no more than a page if your print it up). You will give a brief presentation screencast of your app. This will be no more than 5 minutes (points deducted if it's longer)."

## Overview
The Reddit r/AskScience Flair Dataset was last published in September 2021 from contributor Sumit Mishra. r/AskScience is one of the subreddits on the forum site. This data was extracted from 2016 - 2021 and originally contains 547,761 datapoints and 25 columns. More information on the dataset can be found here: https://data.mendeley.com/datasets/k9r2d9z999/2. This project is intended to be a Natural Language Processing (NLP) classification algorithm to predict the Flair Value from the question asked.   

## Modeling and Design
Because of the large nature of the dataset, only the first 20,000 posts were used in this analysis and modeling. I first displayed a graphical display of number of posts in each flair category and a dash data table display, with options for live filtering and sorting, of the 20,000 posts including in this analysis. For the classification modeling, the dataset is filtered based off of the two unique flags from the dash dropdown option and models are run to classify posts in each category. Preprocessing of the text is completed, aligned with best practices from NLP. The model is run with a 30 / 70 test/training dataset split. Both a Logistic Regression - tf-idf model and a Naive Bayes model for classification are run. The output of the models is shown on the application with updating AUC for each of the models. Most models and flair classifications have AUCs over 0.70, showing both models ability to correctly classify the test dataset between the two classes chosen in the app. 

## Project Video Presentation Overview 
[[Onedrive Video Link](https://livejohnshopkins-my.sharepoint.com/:v:/g/personal/eyuasa1_jh_edu/Efb5TYlEVpBDjnah2CVpgRkBnvXFd_0RFi23eI UewAWyNw?e=mNg1yT)](https://livejohnshopkins-my.sharepoint.com/:v:/g/personal/eyuasa1_jh_edu/Efb5TYlEVpBDjnah2CVpgRkBnvXFd_0RFi23eIUewAWyNw?e=mNg1yT)

## Instructions 
To utilize this application on your local computer, you'll need to run the final_app.py file in python. You'll also need the dataset Flair_Data.csv downloaded and stored in the same places as the .py file. The .py file contains 1) package installations 2) data reading and cleaning 3) dash layout and 4) dash callbacks. The packages that you need installed on your computer are found below: 
- wordcloud
- base64
- dash
- pandas
- plotly.express
- seaborn
- os
- numpy
- re
- string
- nltk
- sklearn
- gensim
- matplotlib.pyplot

