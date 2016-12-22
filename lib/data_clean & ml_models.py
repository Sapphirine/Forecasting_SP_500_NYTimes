
# coding: utf-8

# In[1]:
from pyspark import SparkConf, SparkContext
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from sklearn.metrics import roc_curve, auc
from pyspark.mllib.evaluation import BinaryClassificationMetrics as metric
path = './'
import os
import pandas as pd
file_list = [csv for csv in os.listdir(path) if (csv[0]=='1' or csv[0]=='2') and csv[-4:]=='.csv']


# In[2]:

len(file_list)


# In[93]:

text = pd.read_csv(file_list[0])
for i in file_list[1:]:
    print i
    text = text.append(pd.read_csv(i), ignore_index=True)


# In[94]:

text.shape


# In[96]:

import pickle
with open('all_text', 'wb') as file:
    pickle.dump(text, file)


# In[97]:

data = pd.read_csv("label.csv")
data['sentence'] = ''
for t, i in enumerate(data['DATE'].values):
    temp_df = text.loc[text['pub_date'] == i]
    
    sentence_list = list(temp_df['headline'].values) + list(temp_df['lead_paragraph'].values)
    
    sentence_list = [str(value) for value in sentence_list]
    
    sentence_list_nonan = [value for value in sentence_list if value != 'nan']
    
    sentence = " ".join(sentence_list)

    data.loc[:,'sentence'][data['DATE'] == i] = sentence
    
    if (t % 1000)== 0:
            print t
    #print data.head()


# In[99]:

with open('data_with_text', 'wb') as file:
    pickle.dump(data, file)


# In[3]:

import pickle
data=pickle.load( open( "data_with_text", "rb" ) )
data.head()


# In[4]:

datause = data.loc[:2510,['ret','sentence']]
datause.tail()


# In[5]:

datause.columns=['label','sentence']
datause.shape


# In[6]:

import gc
gc.collect()


# In[6]:

#conf = (SparkConf()
#    .set("spark.driver.memory", "24g"))

sentenceData = spark.createDataFrame(datause)


# In[7]:

sentenceData


# In[9]:

tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
wordsData = tokenizer.transform(sentenceData)
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=1000)
featurizedData = hashingTF.transform(wordsData)
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)
for features_label in rescaledData.select("features", "label").take(3):
    print(features_label)


# In[10]:

# Split the data into train and test
splits = rescaledData.select("features", "label").randomSplit([0.6, 0.4], 1234)
train = splits[0]
test = splits[1]

# create the trainer and set its parameters


# In[38]:
## NB

nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

# train the model
model = nb.fit(train)
# compute accuracy on the test set
result = model.transform(test)


# In[30]:

predictionAndLabels = result.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Accuracy: " + str(evaluator.evaluate(predictionAndLabels)))

## RF

data = rescaledData.select("features", "label")
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)
# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures").fit(data)

(trainingData, testData) = data.randomSplit([0.6, 0.4])

rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=100)

# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

predictions.select("prediction", "indexedLabel", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

rfModel = model.stages[2]
print(rfModel)  # summary only

### LR

lr = LogisticRegression(maxIter=10)
pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])

paramGrid = ParamGridBuilder() \
    .addGrid(hashingTF.numFeatures, [100, 1000, 10000]) \
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .build()

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(),
                          numFolds=3)  # use 3+ folds in practice

# Run cross-validation, and choose the best set of parameters.
cvModel = crossval.fit(train)


# Make predictions on test documents. cvModel uses the best model found (lrModel).
prediction = cvModel.transform(test)
prediction

selected = prediction.select("id", "text", "probability", "prediction")
for row in selected.collect():
    print(row)

accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

results = prediction.select(['probability', 'label'])
 
## prepare score-label set
results_collect = results.collect()
results_list = [(float(i[0][0]), 1.0-float(i[1])) for i in results_collect]
scoreAndLabels = sc.parallelize(results_list)
 
metrics = metric(scoreAndLabels)
print("The ROC score is (@numTrees=200): ", metrics.areaUnderROC)
