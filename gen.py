from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql as PS
from functools import partial
import random
import sys
import time


def timer(f, *args, **kwargs):
    t = time.time()
    r = f(*args, **kwargs)
    print(f"{time.time() - t}")
    return r


def words_from_file(fn: str) -> list[str]:
    with open(fn) as fn:
        words = fn.read().split()

    return words


# "borrowed from quinn lib"
def array_choice(col: PS.Column):
    index = (F.rand()*F.size(col)).cast("int")
    return col[index]


def random_choice(data: list[str], *args):
    return random.choice(data)


def quinn_cols(spark: PS.SparkSession, numrows: int, partitions: int) -> PS.DataFrame:
    first_names_list = words_from_file("first_names.txt")
    last_names_list = words_from_file("last_names.txt")

    data = [(n,) for n in range(numrows)]
    df = spark.createDataFrame(data, ["id"])
    df = df.repartition(partitions, "id")
    fnc = list(map(lambda c: F.lit(c), first_names_list))
    lnc = list(map(lambda c: F.lit(c), last_names_list))
    df = df.withColumn("first_name", array_choice(F.array(fnc)))
    df = df.withColumn("last_name", array_choice(F.array(lnc)))
    return df


def udf_cols(spark: PS.SparkSession, numrows: int, partitions: int) -> PS.DataFrame:
    first_names_list = words_from_file("first_names.txt")
    last_names_list = words_from_file("last_names.txt")

    first_name_udf = F.udf(partial(random_choice, first_names_list))
    last_name_udf = F.udf(partial(random_choice, last_names_list))

    data = [(n,) for n in range(numrows)]
    df = spark.createDataFrame(data, ["id"])
    df = df.repartition(partitions, "id")
    df = df.withColumn("first_name", first_name_udf())
    df = df.withColumn("last_name", last_name_udf())

    return df


if __name__ == "__main__":
    spark = SparkSession.builder.master("local[1]").appName(
        'name_generator').getOrCreate()

    numrows = int(sys.argv[1])
    partitions = 10

    df = timer(udf_cols, spark, numrows, partitions)
    df.write.option("header", True).csv("udf", "overwrite")
    df = timer(quinn_cols, spark, numrows, partitions)
    df.write.option("header", True).csv("quinn", "overwrite")
    spark.stop()
