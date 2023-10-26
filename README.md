# 8--California-Housing



1. [Understanding the Data Set](#schema1)
2. [Import](#schema2)
3. [Creating the Spark Session](#schema3)
4. [Load the data from a file into a DataFrame](#schema4)
5. [Data Exploration](#schema5)
6. [Data Preprocessing](#schema6)
7. [Feature Engineering](#schema7)

<hr>

<a name="schema1"></a>

## 1. Understanding the Data Set

Data set:https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html

These spatial data contain 20,640 observations on housing prices with 9 economic variables:

**Longitude:** refers to the angular distance of a geographic place north or south of the earth’s equator for each block group

**Latitude:** refers to the angular distance of a geographic place east or west of the earth’s equator for each block group

**Housing Median Age:** is the median age of the people that belong to a block group. Note that the median is the value that lies at the midpoint of a frequency distribution of observed values

**Total Rooms:** is the total number of rooms in the houses per block group

**Total Bedrooms:** is the total number of bedrooms in the houses per block group

**Population:** is the number of inhabitants of a block group

**Households:** refers to units of houses and their occupants per block group

**Median Income:** is used to register the median income of people that belong to a block group

**Median House Value:** is the dependent variable and refers to the median house value per block group


<hr>

<a name="schema2"></a>

## 2. Import
Import all necessary libraries
```
import os
import pandas as pd
import numpy as np

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext

from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.sql.functions import udf, col

from pyspark.ml.regression import LinearRegression
from pyspark.mllib.evaluation import RegressionMetrics

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, CrossValidatorModel
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import RegressionEvaluator
```

```
import seaborn as sns
import matplotlib.pyplot as plt
```

```
# Visualization
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_colwidth', 400)

from matplotlib import rcParams
sns.set(context='notebook', style='whitegrid', rc={'figure.figsize': (18,4)})
rcParams['figure.figsize'] = 18,4

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
```
```
# setting random seed for notebook reproducability
rnd_seed=23
np.random.seed=rnd_seed
np.random.set_state=rnd_seed
```

<hr>

<a name="schema3"></a>

## 3. Creating the Spark Session
```
spark = SparkSession.builder.master("local[2]").appName("Linear-Regression-California-Housing").getOrCreate()

sc = spark.sparkContext
```
<hr>

<a name="schema4"></a>

##  4. Load the data from a file into a DataFrame
```
# define the schema, corresponding to a line in the csv data file.
schema = StructType([
    StructField("long", FloatType(), nullable=True),
    StructField("lat", FloatType(), nullable=True),
    StructField("medage", FloatType(), nullable=True),
    StructField("totrooms", FloatType(), nullable=True),
    StructField("totbdrms", FloatType(), nullable=True),
    StructField("pop", FloatType(), nullable=True),
    StructField("houshlds", FloatType(), nullable=True),
    StructField("medinc", FloatType(), nullable=True),
    StructField("medhv", FloatType(), nullable=True)]
)
```
```
# Load housing data
housing_df = spark.read.csv(path='./data/cal_housing.data', schema=schema).cache()
```
```
# Insepct first five rows
housing_df.take(5)
```
```
# Show first five rows
housing_df.show(5)
```
```
# show the dataframe columns
housing_df.columns
```
```
# show the schema of the dataframe
housing_df.printSchema()
```
<hr>

<a name="schema5"></a>

# 5. Data Exploration
#### 5.1 Distribution of the median age of the people living in the area:
```
# groupBy dt by median_age and see the distribution
result_df = housing_df.groupBy('median_age').count().sort("median_age",ascending = False)
```

```
result_df.toPandas().plot.bar(x='median_age',figsize=(14,6))
```
![house1](./img/house1.png)
Most of the residents are either in their youth or they settle here during their senior years. Some data are showing median age < 10 which seems to be out of place.

#### 5.2 Summary Statistics:
```
housing_df.describe().select(
    "summary",
    F.round("median_age",4).alias('median_age'),
    F.round('total_rooms',4).alias('total_rooms'),
    F.round('total_bdrms',4).alias('total_bdrms'),
    F.round('population',4).alias('population'),
    F.round("houshlds", 4).alias("houshlds"),
    F.round("medinc", 4).alias("medinc"),
    F.round("medhv", 4).alias("medhv")
).show()

```
Look at the minimum and maximum values of all the (numerical) attributes. We see that multiple attributes have a wide range of values: we will need to normalize your dataset.



<hr>

<a name="schema6"></a>


# 6. Data Preprocessing

First, let's start with the medianHouseValue, our dependent variable. To facilitate our working with the target values, we will express the house values in units of 100,000. That means that a target such as 452600.000000 should become 4.526:
```
housing_df = housing_df.withColumn("medhv",col("medhv")/100000)
```
NaN
```
from pyspark.sql.functions import col,isnan,when,count
df_Columns = housing_df.columns
housing_df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df_Columns]
   ).show()
```


<hr>

<a name="schema7"></a>

# 7. Feature Engineering

Now that we have adjusted the values in medianHouseValue, we will now add the following columns to the data set:

- Rooms per household which refers to the number of rooms in households per block group;
- Population per household, which basically gives us an indication of how many people live in households per block group; And
- Bedrooms per room which will give us an idea about how many rooms are bedrooms per block group;

As we're working with DataFrames, we can best use the `select()` method to select the columns that we're going to be working with, namely `totalRooms`, `households`, and `population`. Additionally, we have to indicate that we're working with columns by adding the col() function to our code. Otherwise, we won't be able to do element-wise operations like the division that we have in mind for these three variables:

Now that we have adjusted the values in medianHouseValue, we will now add the following columns to the data set:

- Rooms per household which refers to the number of rooms in households per block group;
- Population per household, which basically gives us an indication of how many people live in households per block group; And
- Bedrooms per room which will give us an idea about how many rooms are bedrooms per block group;

As we're working with DataFrames, we can best use the `select()` method to select the columns that we're going to be working with, namely `totalRooms`, `households`, and `population`. Additionally, we have to indicate that we're working with columns by adding the col() function to our code. Otherwise, we won't be able to do element-wise operations like the division that we have in mind for these three variables:

```
# Add the new columns to df
housing_df = (
    housing_df.withColumn("rooms_per_hs", F.round(col("total_rooms")/col("houshlds"),2))
    .withColumn("population_per_hs", F.round(col("population")/col("houshlds"),2))
    .withColumn("bedrooms_per_rooms", F.round(col("total_bdrms")/col("total_rooms"),2))
)
```
We can see that, for the first row, there are about 6.98 rooms per household, the households in the block group consist of about 2.5 people and the amount of bedrooms is quite low with 0.14:

Since we don't want to necessarily standardize our target values, we'll want to make sure to isolate those in our data set. Note also that this is the time to leave out variables that we might not want to consider in our analysis. In this case, let's leave out variables such as longitude, latitude, housingMedianAge and totalRooms.

In this case, we will use the `select()` method and passing the column names in the order that is more appropriate. In this case, the target variable medianHouseValue is put first, so that it won't be affected by the standardization.

```
# Re-order and select columns
housing_df = housing_df.select("medhv",
                               "total_bdrms",
                               "population",
                               "houshlds",
                               "medinc",
                               "rooms_per_hs",
                               "population_per_hs",
                               "bedrooms_per_rooms"
                              )
```
## 7.1 Feature Extraction
```
featureCols = ["total_bdrms","population","houshlds","medinc","rooms_per_hs","population_per_hs",
               "bedrooms_per_rooms"]
```

#### Use a VectorAssembler to put features into a feature vector column:
```
# put features into a feature vector column
assembler = VectorAssembler(inputCols=featureCols, outputCol="features")

assembler_df = assembler.transform(housing_df)
```
## 7.2 Standardization

Next, we can finally scale the data using `StandardScaler`. The input columns are the `features`, and the output column with the rescaled that will be included in the scaled_df will be named `"features_scaled"`:

```
# Initialize the standardScaler
standardScaler = StandardScaler(inputCol="features",outputCol="features_scaled")
# Fit the DataFrame to the scaler
scaled_df = standardScaler.fit(assembler_df).transform(assembler_df)

```

<hr>

<a name="schema8"></a>

# 8. Building A Machine Learning Model With Spark ML