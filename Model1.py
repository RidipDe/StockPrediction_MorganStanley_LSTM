
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
import tensorflow as tf
# export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
# export CUDA_HOME=/usr/local/cuda
# import tensorflow as tf


# In[2]:


file = r'Data/datasetTrain.csv'
datasetTrain = pd.read_csv(file)

file = r'Data/datasetTest.csv'
datasetTest = pd.read_csv(file)

# In[3]:


# df_stocks


# # In[4]:


# df_stocks['prices'] = df_stocks['close'].apply(np.int64)


# # In[5]:


# # selecting the prices and articles
# df_stocks = df_stocks[['prices', 'articles', 'MACD', 'Stochastics', 'ATR', 'Open']]


# # In[6]:


# df_stocks


# # In[7]:



# df = df_stocks[['prices','MACD', 'Stochastics', 'ATR', 'Open']].copy()
# df


# # In[8]:


# # Adding new columns to the data frame
# df["compound"] = ''
# df["neg"] = ''
# df["neu"] = ''
# df["pos"] = ''


# # In[9]:


# df


# # In[10]:


# nltk.download()


# # In[11]:


# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# import unicodedata
# sid = SentimentIntensityAnalyzer()
# for date, row in df_stocks.T.iteritems():
#     try:
#         sentence = unicodedata.normalize('NFKD', df_stocks.loc[date, 'articles'])
#         ss = sid.polarity_scores(sentence)
#         df.set_value(date, 'compound', ss['compound'])
#         df.set_value(date, 'neg', ss['neg'])
#         df.set_value(date, 'neu', ss['neu'])
#         df.set_value(date, 'pos', ss['pos'])
#     except TypeError:
#         print (df_stocks.loc[date, 'articles'])
#         print (date)


# # In[12]:


# df


# # In[18]:


# datasetNorm = (df - df.mean()) / (df.max() - df.min())
# dataset = df
# dataset.head(2)


# # In[19]:


num_epochs = 1000

batch_size = 1

total_series_length = 3642

truncated_backprop_length = 3 #The size of the sequence

state_size = 12 #The number of neurons

num_features = 8
num_classes = 1 #[1,0]

num_batches = total_series_length//batch_size//truncated_backprop_length

min_test_size = 100

print('The total series length is: %d' %total_series_length)
print('The current configuration gives us %d batches of %d observations each one looking %d steps in the past' 
      %(num_batches,batch_size,truncated_backprop_length))


# # In[20]:


# datasetTrain = datasetNorm[dataset.index < num_batches*batch_size*truncated_backprop_length]


# for i in range(min_test_size,len(datasetNorm.index)):
    
#     if(i % truncated_backprop_length*batch_size == 0):
#         test_first_idx = len(datasetNorm.index)-i
#         break

# datasetTest =  datasetNorm[dataset.index >= test_first_idx]


# In[21]:


datasetTrain.head(2)


# In[22]:


datasetTest.head(2)


# In[23]:


xTrain = datasetTrain[['Open','MACD','Stochastics','ATR', 'pos', 'neg', 'compound', 'neu']].as_matrix()
yTrain = datasetTrain['prices'].as_matrix()


# In[25]:


print(xTrain[0:1],'\n',yTrain[0:1])


# In[26]:


xTest = datasetTest[['Open','MACD','Stochastics','ATR', 'pos', 'neg', 'compound', 'neu']].as_matrix()
yTest = datasetTest['prices'].as_matrix()


# In[27]:


# print(xTest[0:1],'\n',yTest[0:1])


# In[29]:


# import matplotlib.pyplot as plt
# # get_ipython().magic('matplotlib inline')
# plt.figure(figsize=(25,5))
# plt.plot(xTrain[:,0])
# plt.title('Train (' +str(len(xTrain))+' data points)')
# plt.show()
# plt.figure(figsize=(10,3))
# plt.plot(xTest[:,0])
# plt.title('Test (' +str(len(xTest))+' data points)')
# plt.show()


# In[30]:


batchX_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,truncated_backprop_length,num_features],name='data_ph')
batchY_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,truncated_backprop_length,num_classes],name='target_ph')


# In[31]:


W2 = tf.Variable(initial_value=np.random.rand(state_size,num_classes),dtype=tf.float32)
b2 = tf.Variable(initial_value=np.random.rand(1,num_classes),dtype=tf.float32)


# In[32]:


labels_series = tf.unstack(batchY_placeholder, axis=1)


# In[33]:


cell = tf.contrib.rnn.BasicRNNCell(num_units=state_size)

states_series, current_state = tf.nn.dynamic_rnn(cell=cell,inputs=batchX_placeholder,dtype=tf.float32)


# In[34]:


states_series = tf.transpose(states_series,[1,0,2])


# In[35]:


last_state = tf.gather(params=states_series,indices=states_series.get_shape()[0]-1)
last_label = tf.gather(params=labels_series,indices=len(labels_series)-1)


# In[36]:


weight = tf.Variable(tf.truncated_normal([state_size,num_classes]))
bias = tf.Variable(tf.constant(0.1,shape=[num_classes]))


# In[37]:


prediction = tf.matmul(last_state,weight) + bias
prediction


# In[38]:


loss = tf.reduce_mean(tf.squared_difference(last_label,prediction))

train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)


# In[39]:



loss_list = []
test_pred_list = []

with tf.Session() as sess:
    
    tf.global_variables_initializer().run()
    
    for epoch_idx in range(num_epochs):
                
        print('Epoch %d' %epoch_idx)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length * batch_size
        
            
            batchX = xTrain[start_idx:end_idx,:].reshape(batch_size,truncated_backprop_length,num_features)
            batchY = yTrain[start_idx:end_idx].reshape(batch_size,truncated_backprop_length,1)
                
            #print('IDXs',start_idx,end_idx)
            #print('X',batchX.shape,batchX)
            #print('Y',batchX.shape,batchY)
            
            feed = {batchX_placeholder : batchX, batchY_placeholder : batchY}
            
            #TRAIN!
            _loss,_train_step,_pred,_last_label,_prediction = sess.run(
                fetches=[loss,train_step,prediction,last_label,prediction],
                feed_dict = feed
            )
            
            loss_list.append(_loss)
            
           
            
            if(batch_idx % 100 == 0):
                print('Step %d - Loss: %.6f' %(batch_idx,_loss))
                
    #TEST
    
    
    for test_idx in range(len(xTest) - truncated_backprop_length):
        
        testBatchX = xTest[test_idx:test_idx+truncated_backprop_length,:].reshape((1,truncated_backprop_length,num_features))        
        testBatchY = yTest[test_idx:test_idx+truncated_backprop_length].reshape((1,truncated_backprop_length,1))

        
        #_current_state = np.zeros((batch_size,state_size))
        feed = {batchX_placeholder : testBatchX,
            batchY_placeholder : testBatchY}

        #Test_pred contains 'window_size' predictions, we want the last one
        _last_state,_last_label,test_pred = sess.run([last_state,last_label,prediction],feed_dict=feed)
        test_pred_list.append(test_pred[-1][0]) #The last one



import matplotlib.pyplot as plt
# %matplotlib inline
plt.title('Loss')
plt.scatter(x=np.arange(0,len(loss_list)),y=loss_list)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show();



plt.figure(figsize=(21,7))
plt.plot(yTest,label='Price',color='blue')
plt.plot(test_pred_list,label='Predicted',color='red')
plt.title('Price vs Predicted')
plt.legend(loc='upper left')
plt.show()