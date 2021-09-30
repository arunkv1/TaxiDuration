from __future__ import print_function

# importing everything needede
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.sql.functions import col
from pyspark.ml.regression import LinearRegression
from pyspark.sql.functions import lit

# Create a SparkSession (Note, the config section is only for Windows!)
spark = SparkSession.builder.appName("DecisionTree1").getOrCreate()

# Load up data as dataframe
data = spark.read.option("header", "true").option("inferSchema", "true")\
    .csv("/Users/arunkrishnavajjala/Documents/GMU/CS657/HW2/train.csv")

# creating the "features" column
assembler = VectorAssembler().setInputCols(["vendor_id", "passenger_count", "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]).setOutputCol("features")

# assembling the data
df = assembler.transform(data).select("trip_duration", "features")

# initializaations for cross validation
gg = 0
totalMSETree = 0
totalRMSETree = 0

totalMSEEnhanced = 0
totalRMSEEnhanced = 0

# cross validation loop, goes 10 times
while gg < 10:
    print(gg)
    
    # Let's split our data into training data and testing data
    trainTest = df.randomSplit([0.8, 0.2])
    trainingDF = trainTest[0]
    testDF = trainTest[1]
    
    # Now create our decision tree
    dtr = DecisionTreeRegressor(maxDepth = 10, minInfoGain = 100, minInstancesPerNode = 30, maxBins = 200 ).setFeaturesCol("features").setLabelCol("trip_duration")
    
    # Train the model using our training data
    model = dtr.fit(trainingDF)
    
    # getes all the leeaf nodes (each prediction for the pre trained model)
    collectionTrain = model.transform(trainingDF)
    
    cPred = collectionTrain.select('prediction').collect()
    # this geets all the unique leaf nodes created by the trained tree
    leafNodes = []
    for i in cPred: 
        if i not in leafNodes:
            leafNodes.append(i)
    
    # gets the test predictions from the tree
    collectionTest = model.transform(testDF)
    
    # copies the test predictions
    copyTest = collectionTest
    
    # this is for the linear regression model within each leaf
    # goes through the leaves and makes a regression
    for i in leafNodes:
        # filters the training data with the leaf value 
        fltr = collectionTrain.filter(col("prediction") == i[0])
        fltrT = fltr.drop('prediction')
        # sets up the linearRegression model
        lr = LinearRegression(maxIter=130, regParam=0.3, elasticNetParam=0.8).setFeaturesCol("features").setLabelCol("trip_duration").setPredictionCol("NextPred")
        # trains it with the filtered training daata
        lrModel = lr.fit(fltrT)
        # filters training to have the vectors with same prediction values
        fltrTest = collectionTest.filter(col("prediction") == i[0])
        # keeps only the vectors
        vctrs = fltrTest.drop("trip_duration")
        vctrs2 = vctrs.drop("prediction")
        # runs it through the model
        res = lrModel.transform(vctrs2).cache()
        # adds it onto the collection
        copyTest = copyTest.join(res)
    # removes the vectors so it only has the predictions
    copyTest = copyTest.drop("features")
    
    # variables for RMSE and MSE
    mse1 = 0
    rmse1 = 0
    
    mse2 = 0
    rmse2 = 0
    
    count = 1
    # calculating MSE
    for i in copyTest:
        # original and both predictions
        original = copyTest.head(count)[0][0]
        pred = copyTest.head(count)[0][1]
        pred1 = copyTest.head(count)[0][2]
        
        # doing mse addition and squaring for original tree
        sub = pred - original
        exp = sub ** 2
        mse1 += exp
        
        # doing mse addition and squaring for enhanced tree
        sub2 = pred1 - original
        exp2 = sub2 ** 2
        mse2 += exp2
        
    # dividing by the count to get the average
    mse1 /= count
    mse2 /= count
    
    # doing square root to get RMSE
    rmse1 = mse1 ** (1/2)
    rmse2 = mse2 ** (1/2)
    gg += 1
    
    # adding the values to the total cross validation variables
    totalMSETree += mse1
    totalMSEEnhanced += mse2

# dividing by 10 because of 10 folds of cross validation
# then square roots the MSE to get total RMSE
    
totalMSETree /= 10
totalRMSETree = totalMSETree ** (1/2)

totalMSEEnhanced /= 10
totalRMSEEnhanced = totalMSEEnhanced ** (1/2)

# prints out the values for both trees
print("-----------------------------")
print("Tree MSE: " + str(totalMSETree))
print("Tree RMSE: " + str(totalRMSETree))
print("-----------------------------")
print("Enhanced MSE: " + str(totalMSEEnhanced))
print("Enhanced RMSE: " + str(totalRMSEEnhanced))
print("-----------------------------")

spark.stop()


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    