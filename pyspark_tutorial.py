#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install pyspark


# In[1]:


import pyspark
import pandas as pd


# In[2]:


from pyspark.sql import SparkSession


# In[3]:


spark = SparkSession.builder.appName('Dataframe').getOrCreate()


# In[4]:


spark


# In[5]:


df_pyspark = spark.read.option('header','true').csv('test1.csv',inferSchema = True)


# In[6]:


df_pyspark.printSchema()


# In[7]:


df_pyspark.head(3)


# In[ ]:


df_pyspark.select(['Name','Experience']).show()


# In[ ]:


df_pyspark.describe().show()


# In[ ]:


df_pyspark=df_pyspark.withColumn('Experience After 2 years',df_pyspark['Experience']+2)


# In[ ]:


df_pyspark=df_pyspark.drop('Experience After 2 years')


# In[ ]:


df_pyspark=df_pyspark.withColumnRenamed('Name','New Name')


# In[8]:


df_pyspark.show()


# In[ ]:


df_pyspark.na.drop(how='any').show()


# In[ ]:


df_pyspark.na.drop(how='all').show()


# In[ ]:


df_pyspark.na.drop(how='any',thresh=3).show()


# In[ ]:


df_pyspark.na.drop(how='any',subset=['Salary']).show()


# In[10]:


df_pyspark.na.fill(0,'Salary').show()


# In[11]:


from pyspark.ml.feature import Imputer

imputer = Imputer(inputCols=['age','Experience','Salary'],
                 outputCols=["{}_imputed".format(c) for c in ['age','Experience','Salary']]).setStrategy("mean")


# In[12]:


imputer.fit(df_pyspark).transform(df_pyspark).show()


# In[13]:


df_pyspark.filter('Salary<150').show()


# In[15]:


df_pyspark.filter((df_pyspark['Salary']>100) & (df_pyspark['Salary']<=200)).show()


# In[16]:


##Examples of Pyspark ML


# In[17]:


from pyspark.sql import SparkSession
spark=SparkSession.builder.appName('Missing').getOrCreate()


# In[18]:


training = spark.read.csv('test1.csv',header=True,inferSchema = True)


# In[19]:


training.show()


# In[20]:


from pyspark.ml.feature import VectorAssembler
featureassembler=VectorAssembler(inputCols=["age","Experience"], outputCol="Independent Features")


# In[21]:


output = featureassembler.transform(training)


# In[22]:


output.show()


# In[36]:


finalized_data=output.select("Independent Features","Salary")


# In[38]:


finalized_data = finalized_data.na.drop()


# In[39]:


finalized_data.show()


# In[40]:


from pyspark.ml.regression import LinearRegression
##train test split
train_data,test_data=finalized_data.randomSplit([0.5,0.5])
regressor=LinearRegression(featuresCol='Independent Features', labelCol='Salary')
regressor=regressor.fit(train_data)


# In[41]:


### Coefficients
regressor.coefficients


# In[42]:


### Intercepts
regressor.intercept


# In[43]:


### Prediction
pred_results=regressor.evaluate(test_data)


# In[44]:


pred_results.predictions.show()


# In[45]:


pred_results.meanAbsoluteError,pred_results.meanSquaredError


# In[ ]:




