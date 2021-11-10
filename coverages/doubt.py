#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.models import load_model, Model

class Doubt:
    def __init__(self, model, batch_size):
        self.model = model
        self.batch_size = batch_size
        
    def test_the_most_doubt(self, test_inputs):
        prediction = self.model.predict(test_inputs, batch_size=self.batch_size, verbose=1).max(axis = 1)
        prediction.sort()
        prediction2 = self.model.predict(test_inputs, batch_size=self.batch_size, verbose=1).max(axis = 1)
        list_index = []
        for i in range (712):
            for j in range (10000):
                if prediction[i] == prediction[j]:
                    list_index.append(j)
                    break
        return list_index
    
    def test_the_most_doubt_and_not(self, test_inputs):
        prediction = self.model.predict(test_inputs, batch_size=self.batch_size, verbose=1).max(axis = 1)
        prediction.sort()
        prediction2 = self.model.predict(test_inputs, batch_size=self.batch_size, verbose=1).max(axis = 1)
        list_index = []
        for i in range (237):
            for j in range (10000):
                if prediction[i] == prediction[j]:
                    list_index.append(j)
                    break
        for i in range (9763, 10000):
            for j in range (10000):
                if prediction[i] == prediction[j]:
                    list_index.append(j)
                    break
        for i in range(4881, 5118):
            for j in range (10000):
                if prediction[i] == prediction[j]:
                    list_index.append(j)
                    break
        #print(list_index)
        return list_index

