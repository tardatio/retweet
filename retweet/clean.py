from datetime import datetime

from pyspark.sql import SparkSession

from pyspark.sql.functions import log10
from pyspark.sql.types import FloatType, ArrayType, TimestampType, DoubleType
from pyspark.sql.functions import col, split, udf, to_timestamp

from pyspark.ml.feature import StringIndexer # for days

# for the timestamp
from pyspark.sql.functions import date_format
from pyspark.sql.functions import unix_timestamp

class Clean():

    def cleaning(self, data):
        now = datetime.now()

        data = data.withColumn("days", date_format(unix_timestamp(data["timestamp"],  "EEE MMM dd HH:mm:ss Z yyyy") \
                                                    .cast(TimestampType()),"EEE")) # weekdays

        stringIndexer = StringIndexer().setInputCol("days").setOutputCol("weeks") # weekdays to stringIndex
        data = stringIndexer.fit(data).transform(data)

        data = data.withColumn("hours", date_format(unix_timestamp(data["timestamp"],  "EEE MMM dd HH:mm:ss Z yyyy") \
                                                    .cast(TimestampType()),"HH").cast("int")) # hours of day

        def q_hours(x):
            if x <= 5:
                return 1.0  #0 -> 5
            elif x > 5 and x <=11:
                return 2.0 #6 -> 11
            elif x > 11 and x <=17:
                return 3.0  # 12 -> 17
            else:
                return 4.0 # 18-> 23
        q_hour = udf(q_hours, FloatType())
        data = data.withColumn("q_hours", q_hour(col("hours"))) # 4 type of hours in a day


        def to_log(x):
            if x == None:
                return 0.0
            else:
                return x

        log_to = udf(to_log, DoubleType())

        data = data.withColumn('followers', log_to(log10(col('followers'))))
        data = data.withColumn('friends', log_to(log10(col('friends'))))
        data = data.withColumn('favorites', log_to(log10(col('favorites'))))
        data = data.withColumn('retweets', log_to(log10(col('retweets'))))

        data = data.withColumn("posi_sen", split(data['sentiment'], ' ').getItem(0).cast('float'))
        data = data.withColumn("nega_sen", split(data['sentiment'], ' ').getItem(1).cast('float'))


        def ent_count(row):
            lis = []
            for x in row:
                try:
                    if x[0].isupper() == True:
                        #x = re.sub('[!@#$%]', '', x)
                        lis.append(x)
                except:
                    continue
            if len(lis) == 0:
                return 0.0
            return float(len(lis)) #sum(lis) / len(lis)

        count_ent = udf(ent_count, FloatType())
        data = data.withColumn("count_ent_an", count_ent(split('entities', '[;:]')))

        def enti_score(row):
            lis = []
            for x in row:
                try:
                    lis.append(float(x))
                except:
                    continue
            if len(lis) == 0:
                return 0.0
            return sum(lis) / len(lis)

        cal = udf(enti_score, FloatType())
        data = data.withColumn("ent_score", cal(split('entities', '[;:]')))

        data = data.select("followers","friends","favorites","posi_sen","nega_sen",
                           "count_ent_an","ent_score","days","weeks","hours","q_hours",
                           "retweets")

        print(data.printSchema())
        print(datetime.now()-now)
        return data
