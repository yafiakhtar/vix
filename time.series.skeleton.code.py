from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, avg, when, substring, concat, lit, lag
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor, LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Initialize Spark
spark = SparkSession.builder.appName("VIX_Train_AutoRegressive").getOrCreate()

# ====================================================
# 1. LOAD VIX DATA
# ====================================================
df_vix_raw = spark.read.option("header", "true") \
    .option("inferSchema", "true") \
    .csv("/storage/home/dps6160/scratch/data/VIX_History.csv")

# Clean VIX (MM/dd/yyyy format)
df_vix = df_vix_raw.select(
    col("date"), 
    col("CLOSE").alias("vix") 
).withColumn("date", to_date(col("date"), "MM/dd/yyyy"))

# ====================================================
# 2. LOAD SENTIMENT DATA
# ====================================================
df_headlines = spark.read.option("header", "false") \
    .option("inferSchema", "true") \
    .csv("/storage/home/dps6160/scratch/data/training.1600000.processed.noemoticon.csv/")

# --- ROBUST DATE PARSING ---
df_headlines = df_headlines.withColumn(
    "date_str", 
    concat(
        substring(col("_c2"), 5, 3), lit(" "),
        substring(col("_c2"), 9, 2), lit(" "),
        substring(col("_c2"), -4, 4)
    )
).withColumn("date", to_date(col("date_str"), "MMM dd yyyy"))

# Normalize Sentiment (0=Neg, 4=Pos -> 0.0 to 1.0)
df_headlines = df_headlines.withColumn(
    "sentiment_score", 
    when(col("_c0") == 4, 1.0).otherwise(0.0)
)

df_daily_sentiment = df_headlines \
    .filter(col("date").isNotNull()) \
    .groupBy("date") \
    .agg(avg("sentiment_score").alias("sentiment_score")) \
    .sort("date")

# ====================================================
# 3. MERGE & CREATE LAGS
# ====================================================
df_combined = df_vix.join(df_daily_sentiment, on="date", how="inner").sort("date")

# --- AUTO-REGRESSIVE LOGIC ---
# We use Yesterday's VIX + Yesterday's Sentiment -> Predict Today's VIX
window_spec = Window.orderBy("date")

df_features = df_combined \
    .withColumn("vix_t_1", lag(col("vix"), 1).over(window_spec)) \
    .withColumn("sentiment_t_1", lag(col("sentiment_score"), 1).over(window_spec)) \
    .na.drop() # Drop the first day (no history)

count = df_features.count()


# ====================================================
# 4. SPLIT & TRAIN
# ====================================================
# Inputs: Yesterday's VIX, Yesterday's Sentiment
assembler = VectorAssembler(
    inputCols=["vix_t_1", "sentiment_t_1"], 
    outputCol="features"
)
df_ready = assembler.transform(df_features)

# Dynamic Time Split (80/20)
unique_dates = [row.date for row in df_ready.select("date").distinct().sort("date").collect()]
split_index = int(len(unique_dates) * 0.8)
split_date = unique_dates[split_index]

df_train = df_ready.filter(col("date") < split_date)
df_test = df_ready.filter(col("date") >= split_date)


# Train Random Forest (Good for non-linear relationships)
rf = RandomForestRegressor(labelCol="vix", featuresCol="features", numTrees=100)
vix_model = rf.fit(df_train)

# Evaluate
predictions = vix_model.transform(df_test)
rmse = RegressionEvaluator(labelCol="vix", metricName="rmse").evaluate(predictions)
r2 = RegressionEvaluator(labelCol="vix", metricName="r2").evaluate(predictions)
mae = RegressionEvaluator(labelCol="vix", metricName="mae").evaluate(predictions)
print("R_Squared:", r2)
print("RMS:", rmse)
print("MAE:", mae)

# Feature Importance (See which matters more: Yesterday's VIX or Sentiment?)

# Save
vix_model.write().overwrite().save("/storage/home/dps6160/CourseProject/models/vix_model")
