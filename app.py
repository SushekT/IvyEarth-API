#!/usr/bin/env python
# coding: utf-8

# In[5]:

from flask import Flask, request, jsonify
from flask_cors import CORS


# In[6]:


# import os
# os.system('recommendation.py')
# from recommendation import *
import recommendation


# In[ ]:

app = Flask(__name__)
CORS(app)


@app.route('/plant', methods=['GET'])
def recommend_plants():
    res = recommendation.results(request.args.get('title'))
    return jsonify(res)


if __name__ == '__main__':
    app.run(port=5000, debug=True)

# %%
