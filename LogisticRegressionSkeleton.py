from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder.appName("SentimentRegression").getOrCreate()

#Load Data
data = spark.read.option("header", "true").option("inferSchema", "true").csv("/storage/home/dps6160/scratch/data/training.1600000.processed.noemoticon.csv")

#Filter out rows with missing headlines or scores
data = data.filter(col("text").isNotNull() & col("sentiment").isNotNull())
data = data.withColumn("sentiment", col("sentiment").cast("double"))

#Tokenize
tokenizer = Tokenizer(inputCol="text", outputCol="words")

#Vectorize
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20000)

#Re-weight importance (IDF)
idf = IDF(inputCol="rawFeatures", outputCol="features")

#Define the Model (Linear Regression for Continuous Score)
lr = LogisticRegression(featuresCol="features", labelCol="sentiment", regParam=0.01)

#chain the steps
pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, lr])

#Train
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
model = pipeline.fit(train_data)

#Evaluate
predictions = model.transform(test_data)
# Evaluate Accuracy (Standard Metric)
#evaluator = BinaryClassificationEvaluator(labelCol="sentiment", metricName="areaUnderROC")
#auc = evaluator.evaluate(predictions)

#extract_prob = udf(lambda v: float(v[1]), DoubleType())

#df_final_scores = predictions.withColumn("sentiment_score", extract_prob(col("probability")))

#df_final_scores.select("text", "label", "sentiment_score").show(10, truncate=50)

#save the model
model.write().overwrite().save("/storage/home/dps6160/CourseProject/models/sentiment_model")
