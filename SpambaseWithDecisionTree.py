#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

col_names=['word_freq_make','word_freq_address','word_freq_all','word_freq_3d','word_freq_our','word_freq_over','word_freq_remove',
           'word_freq_internet','word_freq_order','word_freq_mail','word_freq_receive',
           'word_freq_will','word_freq_people','word_freq_report','word_freq_addresses','word_freq_free','word_freq_business',
          'word_freq_email','word_freq_you','word_freq_credit','word_freq_your','word_freq_font','word_freq_000','word_freq_money',
          'word_freq_hp','word_freq_hpl','word_freq_george','word_freq_650','word_freq_lab','word_freq_labs','word_freq_telnet',
          'word_freq_857','word_freq_data','word_freq_415','word_freq_85','word_freq_technology','word_freq_1999','word_freq_parts',
          'word_freq_pm','word_freq_direct','word_freq_cs','word_freq_meeting','word_freq_original','word_freq_project','word_freq_re',
          'word_freq_edu','word_freq_table','word_freq_conference','char_freq_;','char_freq_(','char_freq_[','char_freq_!','char_freq_$',
          'char_freq_#','capital_run_length_average','capital_run_length_longest','capital_run_length_total','spamLabel']
dataset=pd.read_csv("spambase.data")
dataset.columns=col_names
dataset.head()


# In[2]:


#sınıf ve özelliği veri kümesinde ayırma
feature_cols= ['word_freq_make','word_freq_address','word_freq_all','word_freq_3d','word_freq_our','word_freq_over','word_freq_remove',
           'word_freq_internet','word_freq_order','word_freq_mail','word_freq_receive',
           'word_freq_will','word_freq_people','word_freq_report','word_freq_addresses','word_freq_free','word_freq_business',
          'word_freq_email','word_freq_you','word_freq_credit','word_freq_your','word_freq_font','word_freq_000','word_freq_money',
          'word_freq_hp','word_freq_hpl','word_freq_george','word_freq_650','word_freq_lab','word_freq_labs','word_freq_telnet',
          'word_freq_857','word_freq_data','word_freq_415','word_freq_85','word_freq_technology','word_freq_1999','word_freq_parts',
          'word_freq_pm','word_freq_direct','word_freq_cs','word_freq_meeting','word_freq_original','word_freq_project','word_freq_re',
          'word_freq_edu','word_freq_table','word_freq_conference','char_freq_;','char_freq_(','char_freq_[','char_freq_!','char_freq_$',
          'char_freq_#','capital_run_length_average','capital_run_length_longest','capital_run_length_total']
X=dataset[feature_cols] #özellikler saklandı
y=dataset.spamLabel #sınıf bilgisi
print(X)
print(y)


# In[3]:


#veri kümesini eğitim ve test olarak ayırma kısmı 
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=1) #featurelar ve sınıflar veriliyor
print(X_test) #%20sini bastırır (test ve öğrenme kümeleri ayrıldı)


# In[4]:


#Decision tree sınıflayıcısı için obje oluşturma
classification = DecisionTreeClassifier()

#karar ağacı sınıflayıcısının eğitim kısmı 
classification = classification.fit(X_train,y_train)


# In[5]:


#test veri kümesinin sınıfını(spamLabel) tahmin etme kısmı 
y_prediction = classification.predict(X_test)


# In[6]:


#modelin doğruluğu(accuracy) önemli!!!! sınıflandırıcı ne kadar doğru bir tahminde bulundu hedef bu 
print("Accuracy-Doğruluk:", metrics.accuracy_score(y_test,y_prediction))


# In[ ]:




