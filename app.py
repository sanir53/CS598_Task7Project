#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 15:39:26 2022

@author: sagardalwadi
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import pandas as pd
import json
import nltk
from collections import Counter
from collections import OrderedDict 
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server # the Flask app

# load serialized model
#serialize_path = './pipeline.joblib'
#pipeline_serialized = load(serialize_path) 

app.layout = html.Div(children=[
    html.H1('Restaurant Recommendation'), 

    html.Div(
             [
        html.H4('Instructions: Please enter an Indian dish name for which you would like to get a recommendation of restaurant'),
        html.H5('Note: Dish name should be in lower case. If a dish name is not available in yelp Indian cuisine then you may get inaccurate recommendation. i.e. tikka masala, garlic naan, chicken tikka etc.'),
        html.Br(),
    ]),

    ##### Input Section #####
    html.Div([
        ### Text Input
        html.I("Please enter an Indian dish name"),
        html.Br(),
        dcc.Textarea(
                id="dishname", 
                placeholder="Please enter dish name here.",               
                style={'width': '67%', 'height': '100px'},
            ),
        html.Br(), html.Br(),

      
    ]),

    html.Button(id='submit-button', n_clicks=0, children='Submit'),

    ##### Output Section #####
    html.Br(), html.Br(),
    html.Div([
        html.H5(id="output", style={'font-family': 'Times New Roman, Times, serif','color': 'blue'})]),
   
])

@app.callback(
    Output("output", "children"),
    Input("submit-button", "n_clicks"), 
    State("dishname", "value"),
)
def update_output(n_clicks, dishname):
   # path2files="/Users/sagardalwadi/Desktop/College/CS598/Week4/yelp_dataset_challenge_academic_dataset/"
    path2buisness="yelp_academic_dataset_business.json"
    path2reviews="yelp_academic_dataset_review.json"

    business_id = []
    restaurant_name = dict()
    cuisine = 'Indian'

    with open (path2buisness, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            business_json = json.loads(line)
            
            if cuisine in business_json['categories']:
                data_id = business_json['business_id']
                business_id.append(data_id)
                restaurant_name[data_id] = business_json['name']
    reviews = [] 
    stars = []
    rest_name = []

    with open (path2reviews, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            review_json = json.loads(line)
            if review_json['business_id'] in business_id:
                reviews.append(review_json['text'])
                stars.append(review_json['stars'])
                rest_name.append(restaurant_name[review_json['business_id']])

    selected_dishes = [dishname]

    unique_restaurants = list(set(rest_name))
    rest_total_sentiment = OrderedDict(zip(unique_restaurants, [0]*len(unique_restaurants)))
    rest_review_counter = OrderedDict(zip(unique_restaurants, [0]*len(unique_restaurants)))

    for i, review in enumerate(reviews):
        review = review.replace("\t", " ").replace("\n", "").replace("\r", "").lower().strip()
        # skip "neutral" reviews by stars
        if stars[i] == 3:
            continue
        if (selected_dishes[0] in review):
            toAnalyze = TextBlob(review)               # sentiment analysis part
            sent = toAnalyze.sentiment.polarity
            scaled_sent = 5*(sent+1)
            rest_review_counter[rest_name[i]] += 1 # used for average
            rest_total_sentiment[rest_name[i]] += scaled_sent

    rest_sentiment_df = pd.DataFrame(columns=['Restaurant_Name', 'Total_Sentiment', "Review_Count"])
    rest_sentiment_df['Restaurant_Name'] = list(rest_total_sentiment.keys())
    rest_sentiment_df['Total_Sentiment'] = list(rest_total_sentiment.values())
    rest_sentiment_df['Review_Count'] = list(rest_review_counter.values())
    rest_sentiment_df['Average_Sentiment'] = (rest_sentiment_df['Total_Sentiment'] + 1e-3)/ (rest_sentiment_df['Review_Count'] + 1e-3)

    sorted_df = rest_sentiment_df.sort_values(by='Total_Sentiment',ascending=False)
    sorted_df.reset_index(drop = True, inplace = True)
    value = sorted_df.at[0, 'Restaurant_Name']
    avg_snt= sorted_df.at[0, 'Average_Sentiment']
    rvw= sorted_df.at[0, 'Review_Count']
    output_string = 'Recommended Restaurant for your selected dish is "{}" with the average sentiment rating of "{}" And total number of reviews "{}"'.format(value,avg_snt,rvw)
    return output_string


if __name__ == '__main__':
    app.run_server(debug=True)