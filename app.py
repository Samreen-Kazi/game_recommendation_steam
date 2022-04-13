# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
import itertools
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
import gradio as gr
#from datasets import load_dataset
#dataset = load_dataset('csv', data_files="steam-clean-games.csv", streaming=True)
#df = pd.DataFrame.from_dict(dataset)
df = pd.read_csv("steam-clean-games.csv",  error_bad_lines=False, encoding='utf-8')
# the function to extract years
def extract_year(date):
   year = date[:4]
   if year.isnumeric():
      return int(year)
   else:
      return np.nan
df['year'] = df['release_date'].apply(extract_year)
df['steamspy_tags'] = df['steamspy_tags'].str.replace(' ','-')
df['genres'] = df['steamspy_tags'].str.replace(';',' ')
counts = dict()
for i in df.index:
   for g in df.loc[i,'genres'].split(' '):
      if g not in counts:
         counts[g] = 1
      else:
         counts[g] = counts[g] + 1
def create_score(row):
  pos_count = row['positive_ratings']  
  neg_count = row['negative_ratings']
  total_count = pos_count + neg_count
  average = pos_count / total_count
  return round(average, 2)
def total_ratings(row):
  pos_count = row['positive_ratings']  
  neg_count = row['negative_ratings']
  total_count = pos_count + neg_count
  return total_count
df['total_ratings'] = df.apply(total_ratings, axis=1)
df['score'] = df.apply(create_score, axis=1)
# Calculate mean of vote average column
C = df['score'].mean()
m = df['total_ratings'].quantile(0.90)
# Function that computes the weighted rating of each game
def weighted_rating(x, m=m, C=C):
    v = x['total_ratings']
    R = x['score']
    # Calculation based on the IMDB formula
    return round((v/(v+m) * R) + (m/(m+v) * C), 2)
# Define a new feature 'score' and calculate its value with `weighted_rating()`
df['weighted_score'] = df.apply(weighted_rating, axis=1)
# create an object for TfidfVectorizer
tfidf_vector = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vector.fit_transform(df['genres'])
# create the cosine similarity matrix
sim_matrix = linear_kernel(tfidf_matrix,tfidf_matrix)
# create a function to find the closest title
def matching_score(a,b):
  #fuzz.ratio(a,b) calculates the Levenshtein Distance between a and b, and returns the score for the distance
   return fuzz.ratio(a,b)
"""# Make our Recommendation Engine
We need combine our formatted dataset with the similarity logic to return recommendations. This is also where we can fine-tune it if we do not like the results.
"""
##These functions needed to return different attributes of the recommended game titles
#Convert index to title_year
def get_title_year_from_index(index):
   return df[df.index == index]['year'].values[0]
#Convert index to title
def get_title_from_index(index):
   return df[df.index == index]['name'].values[0]
#Convert index to title
def get_index_from_title(title):
   return df[df.name == title].index.values[0]
#Convert index to score
def get_score_from_index(index):
   return df[df.index == index]['score'].values[0]
#Convert index to weighted score
def get_weighted_score_from_index(index):
   return df[df.index == index]['weighted_score'].values[0]
#Convert index to total_ratings
def get_total_ratings_from_index(index):
   return df[df.index == index]['total_ratings'].values[0]
#Convert index to platform
def get_platform_from_index(index):
  return df[df.index == index]['platforms'].values[0]
   
# A function to return the most similar title to the words a user type
def find_closest_title(title):
  #matching_score(a,b) > a is the current row, b is the title we're trying to match
   leven_scores = list(enumerate(df['name'].apply(matching_score, b=title))) #[(0, 30), (1,95), (2, 19)~~] A tuple of distances per index
   sorted_leven_scores = sorted(leven_scores, key=lambda x: x[1], reverse=True) #Sorts list of tuples by distance [(1, 95), (3, 49), (0, 30)~~]
   closest_title = get_title_from_index(sorted_leven_scores[0][0])
   distance_score = sorted_leven_scores[0][1]
   return closest_title, distance_score
def gradio_contents_based_recommender_v2(game, how_many, sort_option, min_year, platform, min_score):
  #Return closest game title match
  closest_title, distance_score = find_closest_title(game)
  #Create a Dataframe with these column headers
  recomm_df = pd.DataFrame(columns=['Game Title', 'Year', 'Score', 'Weighted Score', 'Total Ratings'])
  #find the corresponding index of the game title
  games_index = get_index_from_title(closest_title)
  #return a list of the most similar game indexes as a list
  games_list = list(enumerate(sim_matrix[int(games_index)]))
  #Sort list of similar games from top to bottom
  similar_games = list(filter(lambda x:x[0] != int(games_index), sorted(games_list,key=lambda x:x[1], reverse=True)))
  #Print the game title the similarity matrix is based on
  print('Here\'s the list of games similar to '+'\033[1m'+str(closest_title)+'\033[0m'+'.\n')
  #Only return the games that are on selected platform
  n_games = []
  for i,s in similar_games:
    if platform in get_platform_from_index(i):
      n_games.append((i,s))
  #Only return the games that are above the minimum score
  high_scores = []
  for i,s in n_games:
    if get_score_from_index(i) > min_score:
      high_scores.append((i,s))
    
  #Return the game tuple (game index, game distance score) and store in a dataframe
  for i,s in n_games[:how_many]: 
    #Dataframe will contain attributes based on game index
    row = {'Game Title': get_title_from_index(i), 'Year': get_title_year_from_index(i), 'Score': get_score_from_index(i), 
           'Weighted Score': get_weighted_score_from_index(i), 
           'Total Ratings': get_total_ratings_from_index(i),}
    #Append each row to this dataframe       
    recomm_df = recomm_df.append(row, ignore_index = True)
  #Sort dataframe by Sort_Option provided by user
  recomm_df = recomm_df.sort_values(sort_option, ascending=False)
  #Only include games released same or after minimum year selected
  recomm_df = recomm_df[recomm_df['Year'] >= min_year]
  return recomm_df
#Create list of unique calendar years based on main df column
years_sorted = sorted(list(df['year'].unique()))
#Interface will include these buttons based on parameters in the function with a dataframe output
recommender = gr.Interface(gradio_contents_based_recommender_v2, ["text", gr.inputs.Slider(1, 20, step=int(1)),
                                                            gr.inputs.Radio(['Year','Score','Weighted Score','Total Ratings']),
                                                            gr.inputs.Slider(int(years_sorted[0]), int(years_sorted[-1]), step=int(1)),
                                                            gr.inputs.Radio(['windows','linux','mac']),
                                                            gr.inputs.Slider(0, 10, step=0.1)],
                        "dataframe")
recommender.launch(debug=True)