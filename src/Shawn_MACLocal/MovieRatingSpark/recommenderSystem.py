#coding=utf-8 

# Programmed by Yang Zhou
# Date: 2016/06/08
# This file fulfills the function of movie rating predictions and is going to be developed into 
# a movie recommendation system based on Python/Flask/Spark

import os
import urllib
import zipfile
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS
from pyspark.mllib.recommendation import MatrixFactorizationModel
import math
from time import time

def quiet_logs( sc ):
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
    logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )


sc = SparkContext(appName="MovieRating")
quiet_logs( sc )
datasets_path = './datasets'
complete_movies_file = os.path.join(datasets_path, 'ml-latest', 'movies.csv')
complete_movies_raw_data = sc.textFile(complete_movies_file)
complete_movies_raw_data_header = complete_movies_raw_data.take(1)[0]

# Parse
# After cache, results are the 'PipelinedRDD' object 
# By default, each transformed RDD may be recomputed each time you run an action on it. However, you may also persist an RDD in memory 
# using the persist (or cache) method, in which case Spark will keep the elements around on the cluster for much faster access the next 
# time you query it. There is also support for persisting RDDs on disk, or replicated across multiple nodes.

# http://stackoverflow.com/questions/28981359/why-do-we-need-to-call-cache-or-persist-on-a-rdd

# >>> print(complete_movies_data.take(2))
# >>> [(1, u'Toy Story (1995)', u'Adventure|Animation|Children|Comedy|Fantasy'), (2, u'Jumanji (1995)', u'Adventure|Children|Fantasy')]
complete_movies_data = complete_movies_raw_data.filter(lambda line: line!=complete_movies_raw_data_header)\
            .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),tokens[1],tokens[2])).cache()


complete_movies_titles = complete_movies_data.map(lambda x: (int(x[0]),x[1]))

print "There are %s movies in the complete dataset" % (complete_movies_titles.count())


def get_counts_and_averages(ID_and_ratings_tuple):
    nratings = len(ID_and_ratings_tuple[1])
    return ID_and_ratings_tuple[0], (nratings, float(sum(x for x in ID_and_ratings_tuple[1]))/nratings)

# loading the complete rating data 
complete_ratings_file = os.path.join(datasets_path, 'ml-latest', 'ratings.csv')
complete_ratings_raw_data = sc.textFile(complete_ratings_file)
complete_ratings_raw_data_header = complete_ratings_raw_data.take(1)[0]

# Parse
complete_ratings_data = complete_ratings_raw_data.filter(lambda line: line!=complete_ratings_raw_data_header)\
            .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),int(tokens[1]),float(tokens[2]))).cache()

print "There are %s recommendations in the complete dataset" % (complete_ratings_data.count())

# x[1] and x[2] are the movieID and ratings
# movie_ID_with_avg_ratings_RDD is used to obtain (movieID, [ratings])
movie_ID_with_ratings_RDD = (complete_ratings_data.map(lambda x: (x[1], x[2])).groupByKey())
movie_ID_with_avg_ratings_RDD = movie_ID_with_ratings_RDD.map(get_counts_and_averages)
movie_rating_counts_RDD = movie_ID_with_avg_ratings_RDD.map(lambda x: (x[0], x[1][0]))

# new user
new_user_ID = 0
# The format of each line is (userID, movieID, rating)
new_user_ratings = [
     (0,260,4), # Star Wars (1977)
     (0,1,3), # Toy Story (1995)
     (0,16,3), # Casino (1995)
     (0,25,4), # Leaving Las Vegas (1995)
     (0,32,4), # Twelve Monkeys (a.k.a. 12 Monkeys) (1995)
     (0,335,1), # Flintstones, The (1994)
     (0,379,1), # Timecop (1994)
     (0,296,3), # Pulp Fiction (1994)
     (0,858,5) , # Godfather, The (1972)
     (0,50,4) # Usual Suspects, The (1995)
    ]
# transform the python list into RDD data
new_user_ratings_RDD = sc.parallelize(new_user_ratings)
# print 'New user ratings: %s' % new_user_ratings_RDD.take(10)

# The previous lines convert the new users' data to RDD format. Union() func add the new data to the original data set
complete_data_with_new_ratings_RDD = complete_ratings_data.union(new_user_ratings_RDD)

# results from MovieRating
best_rank = 8
best_lambda = 0.2
seed = 5L
iterations = 100

# to avoid stackover flow
sc.setCheckpointDir('checkpoint/')

t0 = time()
new_ratings_model = ALS.train(complete_data_with_new_ratings_RDD, best_rank, seed=seed, 
                              iterations=iterations, lambda_=best_lambda)
tt = time() - t0

print "New model trained in %s seconds" % round(tt,3)

# Getting top recommendations
new_user_ratings_ids = map(lambda x: x[1], new_user_ratings) # get just movie IDs; Note here, the map() in Python is used instead of the map() in Spark
# keep just those not on the ID list (thanks Lei Li for spotting the error!)
new_user_unrated_movies_RDD = complete_movies_data.filter(lambda x: x[0] not in new_user_ratings_ids).map(lambda x: (new_user_ID, x[0]))

# Use the input RDD, new_user_unrated_movies_RDD, with new_ratings_model.predictAll() to predict new ratings for the movies
# Need to check how to use function predictAll()
new_user_recommendations_RDD = new_ratings_model.predictAll(new_user_unrated_movies_RDD)


# Transform new_user_recommendations_RDD into pairs of the form (Movie ID, Predicted Rating)
new_user_recommendations_rating_RDD = new_user_recommendations_RDD.map(lambda x: (x.product, x.rating))
new_user_recommendations_rating_title_and_count_RDD = \
    new_user_recommendations_rating_RDD.join(complete_movies_titles).join(movie_rating_counts_RDD)
new_user_recommendations_rating_title_and_count_RDD.take(3)

# Transform into (Title, Rating, Ratings Count).
new_user_recommendations_rating_title_and_count_RDD = \
    new_user_recommendations_rating_title_and_count_RDD.map(lambda r: (r[1][0][1], r[1][0][0], r[1][1]))

# Find the top movies
top_movies = new_user_recommendations_rating_title_and_count_RDD.filter(lambda r: r[2]>=25).takeOrdered(25, key=lambda x: -x[1])

# Python the 格式输出
print ('TOP recommended movies (with more than 25 reviews):\n%s' % '\n'.join(map(str, top_movies)))



'''
# 以下有错 可以提醒作者
my_movie = sc.parallelize([(0, 500)]) # Quiz Show (1994)
# 原文有错 individual_movie_rating_RDD = new_ratings_model.predictAll(new_user_unrated_movies_RDD)
individual_movie_rating_RDD = new_ratings_model.predictAll(my_movie)
individual_movie_rating_RDD.take(1)
'''
# save the model
model_path = os.path.join('.', 'models', 'movie_lens_als')

# Save and load model
new_ratings_model.save(sc, model_path)
same_model = MatrixFactorizationModel.load(sc, model_path)