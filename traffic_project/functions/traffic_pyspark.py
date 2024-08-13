#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 10:57:32 2024

@author: cjymain
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, cos, sin, hour, minute, radians, to_date, dayofweek, when, create_map, lit, percent_rank, log, udf
from pyspark.sql.types import FloatType, IntegerType, StructField, StructType, StringType, MapType
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import LinearRegression
from pyspark.sql.functions import mean, stddev, min as _min, max as _max, count, sum as _sum
from pyspark.sql.window import Window
from pyspark.ml.evaluation import ClusteringEvaluator, RegressionEvaluator, MulticlassClassificationEvaluator

from datetime import datetime, timedelta
import numpy as np
import random

### FURTHER PREPARE DATA FOR MODELING ###

# Initialize a Spark session
spark = SparkSession.builder \
    .appName("Traffic data distributed modeling") \
    .getOrCreate()
    
def gather_group_data(data_path, forecasted=False):
    
    # Read all CSV files in the specified folder in order
    df = spark.read.format("csv").option("header", "true").load(data_path).orderBy("datetime")

    # Create weekday column (bool) to illustrate distinction between weekend and weekday traffic
    df = df.withColumn("weekday",
                       when((dayofweek(to_date("datetime")) >= 2) & (dayofweek(to_date("datetime")) <= 6), True)
                       .otherwise(False))
    
    # Get volume of each 15 minute interval and apply to every row
    volume_df = df.groupBy("datetime").count()
    df = df.join(volume_df, on="datetime", how="inner")
    
    # Convert timestamp to a 15-minute interval index
    df = df.withColumn("interval_index", hour(col("datetime")) * 4 + minute(col("datetime")) / 15)
    
    # Apply cyclical encoding
    df = df.withColumn("interval_rad", radians(col("interval_index") * 360 / 96)) # Splits 96 intervals into equally sized radians of a circle
    
    df = df.withColumn("interval_cos", cos("interval_rad")) # Takes cosine of radian
    df = df.withColumn("interval_sin", sin("interval_rad")) # Takes sine of radian
    
    # Adds fourth root of velocity and travel_time as columns because they have a more linear relationship (good for linear regression model)
    df = df.withColumn("root4_velocity", col("velocity") ** 0.25)
    df = df.withColumn("root4_travel_time", col("travel_time") ** 0.25)
    
    # Concats interval cosine and interval sine, creating 96 distinct vectors, one for each 15 minute period in 24 hours
    time_assembler = VectorAssembler(inputCols=["interval_cos", "interval_sin"], outputCol="time_vector")
    df = time_assembler.transform(df)
    
    # Changes data type of columns so that they can be numerically modeled
    if forecasted == True:
        encoding = "cluster_num" # vehicle_encoding is replaced by cluster num in the forecasted data
        
        # Counts frequencies of time of day per cluster (I.E. 10,000 vehicles in cluster 3 at 10:00 AM)
        tod_frequencies = df.groupBy("cluster_num", "interval_index").agg(count("*").alias("frequency"))
    else:
        encoding = "vehicle_encoding"
    
    df = df \
        .withColumn("id" ,
                    df["id"]
                    .cast(IntegerType()))   \
        .withColumn(encoding ,
                    df[encoding]
                    .cast(IntegerType()))   \
        .withColumn("velocity" ,
                    df["velocity"]
                    .cast(FloatType()))     \
        .withColumn("travel_time" ,
                    df["travel_time"]
                    .cast(FloatType()))     \
        .withColumn("accident" ,
                    df["accident"]
                    .cast(IntegerType()))   \
        
    # Calculate summary statistics for each 15 minute interval and each vehicle type
    time_of_day_summary = df.groupBy("interval_index", encoding).agg(
        count(encoding).alias(f"{encoding}_percentages"),
        mean("velocity").alias("mean_vel"),
        mean("travel_time").alias("mean_time"),
        stddev("velocity").alias("stddev_vel"),
        stddev("travel_time").alias("stddev_time"),
        _min("velocity").alias("min_vel"),
        _max("velocity").alias("max_vel"),
        _min("travel_time").alias("min_time"),
        _max("travel_time").alias("max_time")
    ).orderBy("interval_index")

    # Find mean and standard deviation of volume for each 15 minute interval
    volume_distributions = df.groupBy("interval_index", "time_vector").agg(
        count("*").alias("count"),
        mean("count").alias("mean_volume"),
        stddev("count").alias("stddev_volume")
    ).orderBy("interval_index")
    
    # Accidents details
    accidents_details = df.select(["datetime", "velocity", encoding]).where(col("accident") == 1)
    
    if forecasted == True:
        return df, time_of_day_summary, volume_distributions, accidents_details, tod_frequencies
    else:
        return df, time_of_day_summary, volume_distributions, accidents_details

def cluster_summary_stats(clustered_df):
    # Calculate summary statistics for each cluster
    cluster_summary = clustered_df.groupBy("cluster_num").agg(
        count("cluster_num").alias("count"),
        count(when(clustered_df["velocity"] >= 70, True)).alias("speeders_count"),
        mean("velocity").alias("mean_vel"),
        mean("travel_time").alias("mean_time"),
        stddev("velocity").alias("stddev_vel"),
        stddev("travel_time").alias("stddev_time"),
        _min("velocity").alias("min_vel"),
        _max("velocity").alias("max_vel"),
        _min("travel_time").alias("min_time"),
        _max("travel_time").alias("max_time")
    )
    return cluster_summary
    
# Path to original CSV data in S3
path = "s3://cjytc.other/traffic_project/data/june/"

june_df, time_of_day_summary, volume_distributions, accidents = gather_group_data(path)

### MODEL DATA ###

def modeling(df):
    
    ### KMEANS ###
    # Restructures data to be fit to kmeans model; context specific columns only
    kmeans_assembler = VectorAssembler(inputCols=["velocity", "travel_time"], outputCol="km_features")
    kmeans_df = kmeans_assembler.transform(df)
    
    # Initialize and fit KMeans model
    kmeans = KMeans().setK(3).setSeed(1).setFeaturesCol("km_features")
    km_model = kmeans.fit(kmeans_df)
    clustered_df = km_model.transform(kmeans_df)
    clustered_df = clustered_df.withColumnRenamed("prediction", "cluster_num") # Renames column that determines cluster of the row
    
    # Re-name each cluster to go in order of least to greatest mean velocity so the cluster num can be validly accounted for in Linear Regression
    cluster_means = clustered_df.groupBy('cluster_num').agg(mean('velocity').alias('mean_velocity'))
    
    ordered_clusters = cluster_means.orderBy('mean_velocity').select('cluster_num')
    mapping_expr = create_map([lit(x) for sublist in zip(ordered_clusters.rdd.flatMap(lambda x: x).collect(), range(ordered_clusters.count())) for x in sublist])

    # Replaces original cluster_num with the ordered version
    clustered_df = clustered_df.withColumn('cluster_num', mapping_expr.getItem(col('cluster_num')))
    
    # Find distributions of each vehicle type for each 15 minute interval
    cluster_distributions = clustered_df.groupBy("interval_index", "cluster_num").agg(
        count("*").alias("count"),
        mean("travel_time").alias("mean_cluster_time"),
        stddev("travel_time").alias("stddev_cluster_time")
    )
    
    interval_index_window = Window.partitionBy("interval_index")
    
    cluster_distributions = cluster_distributions.withColumn("sum_counts", _sum("count").over(interval_index_window))
    
    # Converts distributions into percentages to be used by np.random.choice to choose cluster for each forecasted data point
    cluster_distributions = cluster_distributions.withColumn(
        "cluster_percentages",
        col("count") / col("sum_counts")
    )
    
    # Takes the log of travel_times in cluster_0 so that log normal distribution can be performed in forecasting
    log_cluster_0 = clustered_df.select("travel_time").where(col("cluster_num") == 0).withColumn("log_travel_time", log(col("travel_time")))
    log_travel_time_summaries = log_cluster_0.agg(
        mean("log_travel_time").alias("mean_log_time"),
        stddev("log_travel_time").alias("stddev_log_time")
        ).collect()[0]
    
    # Evaluate KMeans
    evaluator = ClusteringEvaluator(predictionCol='cluster_num', featuresCol='km_features', metricName='silhouette', distanceMeasure='squaredEuclidean')
    silhouette = evaluator.evaluate(clustered_df)
    
    # Create dataframe that will store evaluation metrics for each model
    eval_schema = StructType([
        StructField("model", StringType(), True),
        StructField("scores", MapType(StringType(), FloatType()), True)
    ])
    
    eval_df = spark.createDataFrame([("kmeans", {"silhouette": silhouette})], schema=eval_schema)
    
    # Counts frequencies of time of day per cluster (I.E. 10,000 vehicles in cluster 3 at 10:00 AM)
    tod_frequencies = clustered_df.groupBy("cluster_num", "interval_index").agg(count("*").alias("frequency"))
    
    cluster_summary = cluster_summary_stats(clustered_df)

    ### LOGISTIC REGRESSION ###
    # Splits spark dataframe into random 80/20 train test splits
    train, test = clustered_df.randomSplit([0.9, 0.1], seed=68)
    
    # Restructures data to be fit to logistic regression model; context specific columns only
    log_reg_assembler = VectorAssembler(inputCols=["velocity", "time_vector"], outputCol="logreg_features")
    log_reg_train = log_reg_assembler.transform(train)
    log_reg_test = log_reg_assembler.transform(test)
    
    # Initialize and fit Logistic Regression model
    log_reg = LogisticRegression(featuresCol="logreg_features", labelCol="accident", maxIter=10, regParam=0.01)
    log_reg_model = log_reg.fit(log_reg_train)
    
    # Predictions of risk of accident based on velocity
    accident_risk_df = log_reg_model.transform(log_reg_test)
    accident_risk_df = accident_risk_df.withColumnRenamed("probability", "accident_probability")
    
    # Initialize evaluators
    evaluator = MulticlassClassificationEvaluator(labelCol="accident", predictionCol="prediction")
    
    # Calculate F1
    f1 = evaluator.evaluate(accident_risk_df, {evaluator.metricName: "f1"})
    
    # Calculate precision
    precision = evaluator.evaluate(accident_risk_df, {evaluator.metricName: "precisionByLabel"})
    
    # Calculate recall
    recall = evaluator.evaluate(accident_risk_df, {evaluator.metricName: "recallByLabel"})
    
    log_reg_eval = spark.createDataFrame([("logistic_regression", {"f1": f1, "precision": precision, "recall": recall})], schema=eval_schema)
    eval_df = eval_df.union(log_reg_eval)
    
    ### LINEAR REGRESSION ###
    # Splits data in both train and test by 90th percentile
    df = df.withColumn("rank", percent_rank().over(Window.orderBy(col("travel_time"))))
    threshold = df.where(col("rank") >= 0.20).select("travel_time").first()[0]
    
    # Train 90th percentile splits
    train_above = train.filter(col("travel_time") > threshold)
    train_below = train.filter(col("travel_time") <= threshold)
    # Test 90th percentile splits
    test_above = test.filter(col("travel_time") > threshold)
    test_below = test.filter(col("travel_time") <= threshold)
    
    # Primary linear regression model (for data within 90th percentile)
    lin_reg_primary = VectorAssembler(inputCols=["cluster_num", "root4_travel_time", "count", "time_vector", "weekday"], outputCol="linreg_features_primary").setHandleInvalid("skip")
    primary_train = lin_reg_primary.transform(train_above)
    primary_test = lin_reg_primary.transform(test_above)
    
    # Initialize and fit primary Linear Regression model
    lin_reg1 = LinearRegression(featuresCol="linreg_features_primary", labelCol="root4_velocity", predictionCol="predicted_velocity")
    lin_reg_model1 = lin_reg1.fit(primary_train)
    velocity_predictions_primary = lin_reg_model1.transform(primary_test)
    
    # Returns travel time and velocity to their normal values (**4)
    velocity_predictions_primary = velocity_predictions_primary.withColumn("predicted_velocity", col("predicted_velocity") ** 4)
    
    # Secondary linear regression model (for data above 90th percentile)
    lin_reg_secondary = VectorAssembler(inputCols=["root4_travel_time"], outputCol="linreg_features_secondary").setHandleInvalid("skip")
    secondary_train = lin_reg_secondary.transform(train_below)
    secondary_test = lin_reg_secondary.transform(test_below)
    
    # Initialize and fit secondary Linear Regression model
    lin_reg2 = LinearRegression(featuresCol="linreg_features_secondary", labelCol="root4_velocity", predictionCol="predicted_velocity")
    lin_reg_model2 = lin_reg2.fit(secondary_train)
    velocity_predictions_secondary = lin_reg_model2.transform(secondary_test)
    
    # Returns travel time and velocity to their normal values (**4)
    velocity_predictions_secondary = velocity_predictions_secondary.withColumn("predicted_velocity", col("predicted_velocity") ** 4)
    
    # Combine results from the two models to form full predictions
    full_predictions = velocity_predictions_primary.union(velocity_predictions_secondary)
        
    ### MODELS' EVALUATION ###
    
    # Initialize evaluator with different metrics
    evaluator = RegressionEvaluator(labelCol="velocity", predictionCol="predicted_velocity")
    
    # Calculate RMSE
    rmse = evaluator.evaluate(full_predictions, {evaluator.metricName: "rmse"})
        
    # Calculate MSE
    mse = evaluator.evaluate(full_predictions, {evaluator.metricName: "mse"})
    
    # Calculate MAE
    mae = evaluator.evaluate(full_predictions, {evaluator.metricName: "mae"})
    
    # Calculate R^2
    r2 = evaluator.evaluate(full_predictions, {evaluator.metricName: "r2"})
    
    lin_reg_eval = spark.createDataFrame([("linear_regression", {"rmse": rmse, "mse": mse, "mae": mae, "r2": r2})], schema=eval_schema)
    eval_df = eval_df.union(lin_reg_eval)

    ### FORECAST NEW DATA WITH EXISTING MODELS ###
    def new_data_generation():
        
        solutions = {"no_solution":
                         {"travel_time_multiplier": 1.00, "time_intervals": range(96), "clusters": [0, 1, 2]},
                    "increase_police_presence": 
                        {"travel_time_multiplier": 1.15, "time_intervals": range(79, 96), "clusters": [2]}, # Range(80, 97) time intervals translates to 8PM to 12AM
                    "plus_5_speed_limit":
                        {"travel_time_multiplier": 0.91, "time_intervals": range(96), "clusters": [1, 2]}, # Categorically, cluster 1 is normal conditions and cluster 2 is speeders
                    "minus_5_speed_limit": 
                        {"travel_time_multiplier": 1.09, "time_intervals": range(96), "clusters": [1, 2]},
                    "extra_lane":
                        {"travel_time_multiplier": 0.75, "time_intervals": range(96), "clusters": [0, 1]} # Categorically, cluster 0 *should* be accident standstills and rush hour traffic
                    }
        
        # Collect data required for data generation
        cluster_info = cluster_distributions.select(["interval_index", "cluster_percentages", "cluster_num", "mean_cluster_time", "stddev_cluster_time"]).collect()
        volume_info = volume_distributions.select(["interval_index", "time_vector", "mean_volume", "stddev_volume"]).collect()
        
        # Stores each 15 minute interval as a key with a value as a dict of each cluster's info for the period
        cluster_dict = {interval.interval_index: {cluster.cluster_num: cluster for cluster in cluster_info if cluster.interval_index == interval.interval_index} for interval in cluster_info}
        
        volume_dict = {row.interval_index: [row.mean_volume, row.stddev_volume, row.time_vector] for row in volume_info} # Maps summary statistics (mean, stdev) for each 15 minute interval
        
        for solution in solutions.keys():
            for day in range(1, 32): # Each day of July
                vid = 0 # Tracks ID of vehicles for each day
                date = datetime(2024, 7, day)
                weekno = date.weekday()
                if weekno < 5:
                    weekday = True
                else:
                    weekday = False
                rows = []
                for interval in range(96): # Each 15 minute interval in 24 hours
                    timestamp = date + timedelta(minutes=interval*15)
                    
                    # Determines volume using mean and standard deviation of volume for time interval
                    retry = True
                    while retry == True: # Volume will very, very occasionally be negative
                        volume = int(np.random.normal(loc=volume_dict[interval][0], scale=volume_dict[interval][1]))
                        if volume < 0:
                           continue
                        retry = False
                    
                    # Fetches the probabilities of a vehicle belonging to a specific cluster for the time interval
                    probabilities = {cluster: row.cluster_percentages for cluster, row in cluster_dict[interval].items()}
                    
                    # Fetches the mean and standard deviation of traffic travel_time for the time interval
                    travel_time_info = {cluster: [row.mean_cluster_time, row.stddev_cluster_time] for cluster, row in cluster_dict[interval].items()}
                    
                    # Pre-determines cluster number and travel time for each vehicle (grouped np.random calls are up to 100x faster than individual)
                    interval_cluster_nums = np.random.choice(list(probabilities.keys()), p=list(probabilities.values()), size=volume)
                    
                    for vehicle in range(volume):
                        row = []
                        
                        vid += 1
                        
                        # Randomly chooses a cluster
                        cluster = int(interval_cluster_nums[vehicle])
                        
                        # Applies a multiplier (default 1) for means of generated travel times if the current condition matches conditions outlined in the solution
                        solution_multiplier = 1
                        if cluster in solutions[solution]["clusters"] and interval in solutions[solution]["time_intervals"]:
                            solution_multiplier = solutions[solution]["travel_time_multiplier"]
                            
                        # Assigns uniform mean and standard deviation if cluster equals 0
                        if cluster == 0:
                            mean_time = log_travel_time_summaries[0] * solution_multiplier
                            stddev_time = log_travel_time_summaries[1]
                            
                            mu = np.log(mean_time - 0.032) # 0.032 is based on most accurate outcomes for mean and stddev of generated data
                            sigma = (stddev_time / mean_time) + 0.032
                            
                            travel_time = np.random.lognormal(mean=mu, sigma=sigma)
                            travel_time = float(np.exp(travel_time)) # Converts logarithmic travel_time back to standard form (floated so its float64 type, not numpy.float64 -_-)
                        else:
                            mean_time = travel_time_info[cluster][0] * solution_multiplier
                            stddev_time = travel_time_info[cluster][1]
                            
                            travel_time = np.random.normal(loc=mean_time, scale=stddev_time)
                        
                        # Matches correct time vector to interval_index
                        time_vector = volume_dict[interval][-1]
                        
                        row.append(vid)
                        row.append(timestamp)
                        row.append(cluster)
                        row.append(travel_time)
                        row.append(volume)
                        row.append(time_vector)
                        row.append(weekday)
                        
                        rows.append(row)
                    
                # Makes pyspark frame from new day's simulated data
                new_rows = spark.createDataFrame(rows, ["id", "datetime", "cluster_num", "travel_time", "count", "time_vector", "weekday"])
                
                # Create features needed for linear models
                new_rows = new_rows.withColumn("root4_travel_time", col("travel_time") ** 0.25)
                
                # Splits new_rows into sets that will be put through both the respective linear regression models
                new_rows_above = new_rows.filter(col("travel_time") > threshold)
                new_rows_below = new_rows.filter(col("travel_time") <= threshold)
                
                above_assembly = lin_reg_primary.transform(new_rows_above) # Formats frame for linear regression model
                new_rows_modeled1 = lin_reg_model1.transform(above_assembly) # Predicts velocity for each row
                
                below_assembly = lin_reg_secondary.transform(new_rows_below)
                new_rows_modeled2 = lin_reg_model2.transform(below_assembly)
                
                new_rows_lin_reg = new_rows_modeled1.union(new_rows_modeled2)
                
                # Renames predicted_velocity column to velocity to maintain format of existing models and orders frame by vid
                new_rows_lin_reg = new_rows_lin_reg.withColumn("velocity", col("predicted_velocity") ** 4).orderBy("id")
                
                log_reg_assembly = log_reg_assembler.transform(new_rows_lin_reg) # Formats frame for logistic regression model
                new_rows_modeled_final = log_reg_model.transform(log_reg_assembly) # Predicts risk of an accident for each row
                
                def accident_occurrence(probability):
                    probability_of_true = probability[1]
                    return 1 if random.random() < probability_of_true else 0 # Like Numpy.random.choice but distributed
                
                accident_udf = udf(accident_occurrence, IntegerType())
                
                new_rows_modeled_final = new_rows_modeled_final.withColumn("accident", accident_udf(col("probability")))
                
                # Selects columns to be stored in a CSV in S3 and later used to evaluate model effectiveness
                final_df = new_rows_modeled_final.select(["id", "datetime", "cluster_num", "velocity", "travel_time", "accident"])
                
                output_path = f"s3://cjytc.other/traffic_project/data/july/{solution}/highway_x_july_{day}"
                final_df.write.csv(output_path, mode="overwrite", header=True)
    
    # Generates new data using model of existing data
    new_data_generation()
    
    return cluster_summary, tod_frequencies, eval_df

# Models data and returns relevant summary statistics
cluster_summary, tod_frequencies, eval_df = modeling(june_df)   
    
# Writes model summary statistics to S3 for later evaluation
summary_json_path = "s3://cjytc.other/traffic_project/pyspark/results/original/cluster_summary.json"
cluster_summary.write.json(summary_json_path, mode="overwrite")
        
time_distribution_path = "s3://cjytc.other/traffic_project/pyspark/results/original/time_distribution.json"
time_of_day_summary.write.json(time_distribution_path, mode="overwrite")

cluster_frequency_path = "s3://cjytc.other/traffic_project/pyspark/results/original/cluster_frequencies.json"
tod_frequencies.write.json(cluster_frequency_path, mode="overwrite")
        
eval_metrics_path = "s3://cjytc.other/traffic_project/pyspark/results/original/model_evaluation_metrics.json"
eval_df.write.json(eval_metrics_path, mode="overwrite")

accidents_summary_path = "s3://cjytc.other/traffic_project/pyspark/results/original/accidents_details.json"
accidents.write.json(accidents_summary_path, mode="overwrite")
    
# Gathers and aggregates newly generated datasets by-solution and clusters them for later visualization of solution affects on data
for solution in ["no_solution", "increase_police_presence", "plus_5_speed_limit", "minus_5_speed_limit", "extra_lane"]:
    s_path = f"s3://cjytc.other/traffic_project/data/july/{solution}/*/part*.csv"
    s_df, time_of_day_summary, volume_distributions, accidents, cluster_frequencies = gather_group_data(s_path, forecasted=True)
    
    s_cluster_summary = cluster_summary_stats(s_df)
    
    # Writes model summary statistics to S3 for later evaluation
    s_summary_json_path = f"s3://cjytc.other/traffic_project/pyspark/results/{solution}/cluster_summary.json"
    s_cluster_summary.write.json(s_summary_json_path, mode="overwrite")
    
    time_distribution_path = f"s3://cjytc.other/traffic_project/pyspark/results/{solution}/time_distribution.json"
    time_of_day_summary.write.json(time_distribution_path, mode="overwrite")
    
    cluster_frequency_path = f"s3://cjytc.other/traffic_project/pyspark/results/{solution}/cluster_frequencies.json"
    cluster_frequencies.write.json(cluster_frequency_path, mode="overwrite")
            
    eval_metrics_path = f"s3://cjytc.other/traffic_project/pyspark/results/{solution}/model_evaluation_metrics.json"
    volume_distributions.write.json(eval_metrics_path, mode="overwrite")
    
    accidents_summary_path = f"s3://cjytc.other/traffic_project/pyspark/results/{solution}/accidents_details.json"
    accidents.write.json(accidents_summary_path, mode="overwrite")
    
    