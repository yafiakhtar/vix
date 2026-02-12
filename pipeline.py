from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, avg, lag, to_timestamp, when
from pyspark.sql.window import Window
from pyspark.ml import PipelineModel
from pyspark.ml.regression import RandomForestRegressionModel, LinearRegressionModel
from pyspark.ml.feature import VectorAssembler

# Initialize Spark
spark = SparkSession.builder.appName("VIX_Final_Pipeline").getOrCreate()

# ================================================================
# 1. LOAD MODELS
# ================================================================
sentiment_model_path = "/storage/home/dps6160/CourseProject/models/sentiment_model"
vix_model_path = "/storage/home/dps6160/CourseProject/models/vix_model"

# Load Sentiment Model (It's a Pipeline)
try:
    sentiment_model = PipelineModel.load(sentiment_model_path)
except Exception as e:
    exit()

# Load VIX Model (It's likely a Random Forest now)
# We try RF first, then fallback to Linear Regression if you changed your mind
try:
    vix_model = RandomForestRegressionModel.load(vix_model_path)
except:
    try:
        vix_model = LinearRegressionModel.load(vix_model_path)
    except Exception as e:
        exit()

# ================================================================
# 2. GENERATE SENTIMENT SCORES (From Headlines)
# ================================================================
# Update this to where your NEW/INFERENCE data lives
df_headlines = spark.read.parquet("data/parquet/")

# Robust Date Check (Handle variations in Parquet column names)
if "published_at" in df_headlines.columns:
    df_headlines = df_headlines.withColumn("date", to_date(col("published_at")))
elif "timestamp" in df_headlines.columns:
    df_headlines = df_headlines.withColumn("date", to_date(col("timestamp")))
elif "date" in df_headlines.columns:
    df_headlines = df_headlines.withColumn("date", to_date(col("date")))
# If strictly using the 'training' parquet converted from CSV, it might need the substring logic
# But usually, production/inference parquet files have cleaner dates.

df_pred = sentiment_model.transform(df_headlines)

# Extract Probability Score if using LogisticRegression
# (LogReg outputs a vector [prob_neg, prob_pos]. We want index 1.)
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf

if "probability" in df_pred.columns:
    extract_prob = udf(lambda v: float(v[1]), DoubleType())
    df_pred = df_pred.withColumn("sentiment_score", extract_prob(col("probability")))
else:
    # Fallback if using LinearRegression for sentiment
    df_pred = df_pred.withColumnRenamed("prediction", "sentiment_score")

df_daily_sentiment = df_pred \
    .filter(col("date").isNotNull()) \
    .groupBy("date") \
    .agg(avg("sentiment_score").alias("sentiment_score")) \
    .sort("date")

# ================================================================
# 3. LOAD VIX HISTORY & JOIN
# ================================================================
df_vix_raw = spark.read.option("header", "true") \
    .option("inferSchema", "true") \
    .csv("/storage/home/dps6160/scratch/data/VIX_History.csv")

# Clean VIX
df_vix = df_vix_raw.select(
    col("date"), 
    col("CLOSE").alias("vix") 
).withColumn("date", to_date(col("date"), "MM/dd/yyyy"))

# Inner Join: We need BOTH Sentiment and VIX history to make a prediction
df_combined = df_vix.join(df_daily_sentiment, on="date", how="inner").sort("date")

# ================================================================
# 4. CREATE LAG FEATURES
# ================================================================
window_spec = Window.orderBy("date")

df_features = df_combined \
    .withColumn("vix_t_1", lag(col("vix"), 1).over(window_spec)) \
    .withColumn("sentiment_t_1", lag(col("sentiment_score"), 1).over(window_spec)) \
    .na.drop()

# ================================================================
# 5. PREDICT
# ================================================================

assembler = VectorAssembler(
    inputCols=["vix_t_1", "sentiment_t_1"], 
    outputCol="features"
)
df_ready = assembler.transform(df_features)

final_results = vix_model.transform(df_ready)

# Clean Output
output = final_results.select(
    "date",
    col("vix").alias("actual_vix"),
    col("prediction").alias("predicted_vix"),
    "vix_t_1",
    "sentiment_t_1"
)

output.show(10)

# Save
output_path = "/storage/home/dps6160/CourseProject/results/final_autoregressive_predictions.csv"
output.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)
