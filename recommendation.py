#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
import pandas as pd


# In[32]:


import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[38]:


def get_data():
    plant_data = pd.read_csv(
        'dataset/allplants-csv.csv')
    plant_data['name'] = plant_data['name'].str.lower()
    return plant_data
    plant_data.head()


get_data()


# In[39]:


def combine_data(data):
    data_recommend = data.drop(columns=['web-scraper-order', 'url', 'name', 'genus', 'image', 'propagation', 'height',
                               'width', 'flower_color', 'problem_solvers', 'special_features', 'care_must_knows', 'description', 'foliage_color'])
    data_recommend['combine'] = data_recommend[data_recommend.columns[0:2]].apply(
        lambda x: ','.join(x.dropna().astype(str)), axis=1)
    data_recommend = data_recommend.drop(columns=['type', 'light'])
    return data_recommend


# In[40]:


def transform_data(data_combine, data_description):
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(data_combine['combine'])

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data_description['description'])

    combine_sparse = sp.hstack([count_matrix, tfidf_matrix], format='csr')

    cosine_sim = cosine_similarity(combine_sparse, combine_sparse)

    return cosine_sim


# In[42]:


def recommend_plants(name, data, combine, transform):

    indices = pd.Series(data.index, index=data['name'])
    index = indices[name]

    sim_scores = list(enumerate(transform[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]

    plant_indices = [i[0] for i in sim_scores]

    plant_id = data['web-scraper-order'].iloc[plant_indices]
    plant_title = data['name'].iloc[plant_indices]
    plant_type = data['type'].iloc[plant_indices]
    plant_image = data['image'].iloc[plant_indices]
    plant_genus = data['genus'].iloc[plant_indices]
    plant_light = data['light'].iloc[plant_indices]
    plant_description= data['description'].iloc[plant_indices]
    plant_propagation = data['propagation'].iloc[plant_indices]
    plant_height = data['height'].iloc[plant_indices]
    plant_width = data['width'].iloc[plant_indices]
    flower_color = data['flower_color'].iloc[plant_indices]
    foliage_color= data['foliage_color'].iloc[plant_indices]
    problem_solvers= data['problem_solvers'].iloc[plant_indices]
    special_features = data['special_features'].iloc[plant_indices]
    care_must_knows = data['care_must_knows'].iloc[plant_indices]
    
    
    

    recommendation_data = pd.DataFrame(
        columns=['id', 'name', 'type', 'image','genus','light','description','propagation','height','width','flower_color','foliage_color','problem_solvers','special_features','care_must_knows'])

    recommendation_data['id'] = plant_id
    recommendation_data['name'] = plant_title
    recommendation_data['type'] = plant_type
    recommendation_data['image'] = plant_image
    recommendation_data['genus'] = plant_genus
    recommendation_data['light'] = plant_light
    recommendation_data['description'] = plant_description
    recommendation_data['propagation'] = plant_propagation
    recommendation_data['height'] = plant_height
    recommendation_data['width'] = plant_width
    recommendation_data['flower_color'] = flower_color
    recommendation_data['foliage_color'] = foliage_color
    recommendation_data['problem_solvers'] = problem_solvers
    recommendation_data['special_features'] = special_features
    recommendation_data['care_must_knows'] = care_must_knows
    
    
    
    
    
    
    
    
    

    return recommendation_data


# In[43]:


def results(pla_name):
    pla_name = pla_name.lower()

    find_plant = get_data()
    combine_result = combine_data(find_plant)
    transform_result = transform_data(combine_result, find_plant)

    if pla_name not in find_plant['name'].unique():
        return 'Plant not in Database'

    else:
        recommendations = recommend_plants(
            pla_name, find_plant, combine_result, transform_result)
        return recommendations.to_dict('records')


# results(name)
# results("ginkgo tree")


# In[ ]:
