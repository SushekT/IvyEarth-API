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
    sim_scores = sim_scores[1:21]

    plant_indices = [i[0] for i in sim_scores]

    plant_id = data['web-scraper-order'].iloc[plant_indices]
    plant_title = data['name'].iloc[plant_indices]
    plant_type = data['type'].iloc[plant_indices]
    plant_image = data['image'].iloc[plant_indices]
    plant_image = data['genus'].iloc[plant_indices]
    plant_image = data['light'].iloc[plant_indices]
    plant_image = data['description'].iloc[plant_indices]
    plant_image = data['propagation'].iloc[plant_indices]
    plant_image = data['height'].iloc[plant_indices]
    plant_image = data['width'].iloc[plant_indices]
    plant_image = data['image'].iloc[plant_indices]
    plant_image = data['flower_color'].iloc[plant_indices]
    plant_image = data['foliage_color'].iloc[plant_indices]
    plant_image = data['problem_solvers'].iloc[plant_indices]
    plant_image = data['special_features'].iloc[plant_indices]
    plant_image = data['care_must_knows'].iloc[plant_indices]
    
    
    

    recommendation_data = pd.DataFrame(
        columns=['Plant_Id', 'Name', 'Type', 'Image','Genus','Light','Description','Propagation','Height','Width','Flower_Color','Foliage_Color','Problem_Solvers','Special_Features','Care_Must_Knows'])

    recommendation_data['Plant_Id'] = plant_id
    recommendation_data['Name'] = plant_title
    recommendation_data['Type'] = plant_type
    recommendation_data['Image'] = plant_image
    recommendation_data['Genus'] = plant_genus
    recommendation_data['Light'] = plant_light
    recommendation_data['Description'] = plant_description
    recommendation_data['Propagation'] = plant_propagation
    recommendation_data['Height'] = plant_height
    recommendation_data['Width'] = plant_width
    recommendation_data['Flower_Color'] = flower_color
    recommendation_data['Foliage_Color'] = foliage_color
    recommendation_data['Problem_Solvers'] = problem_solvers
    recommendation_data['Special_Features'] = special_features
    recommendation_data['Care_Must_Knows'] = care_must_knows
    
    
    
    
    
    
    
    
    

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
