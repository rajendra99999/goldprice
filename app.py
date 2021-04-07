#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np
from flask import Flask,request,render_template,jsonify,url_for


# In[2]:


app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))


# In[3]:


@app.route('/')
def home():
    return render_template('gold.html')


# In[4]:


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        ir=float(request.form['ir'])
        gdp=float(request.form['gdp'])
        i=float(request.form['i'])
        
    
        input_var=[ir,gdp,i]
        final_input=[np.array(input_var)]
        prediction = model.predict(final_input)
        output=round(prediction[0]/28.3495,2)
        
        return render_template('gold.html', prediction_text='Predicted Gold Price is {}'.format(output))

if __name__=='__main__':
        app.run(debug=True)
    


# In[ ]:




