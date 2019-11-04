import numpy
import matplotlib.pyplot as plt
import pandas
from scipy.spatial import distance

# Specify numer of clusters to find
k = 3

file_string = "iris.txt"
train_data = pandas.read_csv(file_string, sep=" ", header=None)

columns_array = []
total_test_columns = train_data.shape[1]
print('total_test_columns')
print(total_test_columns)
for i in list(range(0, total_test_columns)):
  columns_array.append(i)
columns_array.append('cluster_group')

print(columns_array[0:-1])

train_data.columns = columns_array[0:-1]

# Identify a seed for consistent results
seed = None # set to none for no seed
# Sample K Random Points
train_data_centroids = train_data.sample(n = k, random_state = seed)
train_data_centroids.columns = columns_array[0:-1]

centroids_did_not_change = False

max_iterations = 20
for iteration_index in list(range(0, max_iterations)):
  if (centroids_did_not_change):
    break

  print("{} / {}".format(iteration_index, max_iterations))
  cluster_group_array = []
  for index, row in train_data.iterrows():
    row = numpy.array(row)

    prev_distance = float("inf")
    cluster_group = None

    cluster_group_index = 0

    train_data_coords = []
    for column_label in columns_array:
      if (column_label != 'cluster_group'):
        train_data_coords.append(row[column_label])

    for i, centroid_row in train_data_centroids.iterrows():
      centroid_coords = []
      for column_label in columns_array:
        if (column_label != 'cluster_group'):
          centroid_coords.append(centroid_row[column_label])

      euclidean_distance = distance.euclidean(
          numpy.array(centroid_coords),
          numpy.array(train_data_coords)
      )

      if (euclidean_distance < prev_distance):
          cluster_group = cluster_group_index
          prev_distance = euclidean_distance

      cluster_group_index = cluster_group_index + 1

    cluster_group_array.append(cluster_group)

  data = {}
  for column_label in columns_array:
    if (column_label != 'cluster_group'):
      data[column_label] = train_data[column_label]
  data['cluster_group'] = cluster_group_array

  # This adds the cluster_group, which will be used to filter the data
  train_data = pandas.DataFrame(data)

  # Reassign Centroids
  # With help from https://stackoverflow.com/questions/45418353/get-nearest-coordinates-from-pandas-df-from-centroid-of-coordinates
  centroids = []
  for i in list(range(0, k)):
    train_data_grouped_by_cluster = train_data[train_data['cluster_group'] == i]
    # Get row containing centroid
    centroid = train_data_grouped_by_cluster.loc[[train_data_grouped_by_cluster.sub(train_data_grouped_by_cluster.mean()).pow(2).sum(1).idxmin()]]
    print('centroid')
    print(centroid)

    centroids_points = []
    for column_label in columns_array:
      if (column_label != 'cluster_group'):
        centroids_points.append(centroid.iloc[0][column_label])

    centroids.append(centroids_points)

  old_centroids = []
  for old_centroid_row_index, old_centroid_row in train_data_centroids.iterrows():
    old_centroids_points = []
    for column_label in columns_array:
      if (column_label != 'cluster_group'):
        old_centroids_points.append(old_centroid_row[column_label])

    old_centroids.append(old_centroids_points)

  centroids_euclidean_distance = 0
  for i in list(range(0, len(centroids))):
    centroids_euclidean_distance += distance.euclidean(
        centroids[i],
        old_centroids[i],
    )

  print('δ: centroids_euclidean_distance error')
  print(centroids_euclidean_distance)

  if (centroids_euclidean_distance == 0):
    centroids_did_not_change = True

  train_data_centroids = pandas.DataFrame(
    numpy.array(centroids),
    columns=columns_array[0:-1]
  )

print('Done')

# Remove the following line for a readable output
# pandas.set_option('display.max_rows', None)

for i in list(range(0, k)):
  print("Cluster {}".format(i))
  print(train_data[train_data['cluster_group'] == i])

  print('mean point')
  mean_point = train_data_centroids.iloc[i]
  print(mean_point)
  
  cluster_data = train_data[train_data['cluster_group'] == i]
  print('size of cluster')
  print(cluster_data.shape[0])