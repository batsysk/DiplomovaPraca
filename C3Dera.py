# c3d_best_model-49.data-00000-of-00001, c3d_best_model-49.index, c3d_best_model-49.meta, checkpoint

# import cv2
# import numpy as np
# import tensorflow as tf
# import random
# import os
# import csv
#
# # Dictionary with saved category names
# video_categories = {
#     0: "Baseball",
#     1: "Basketball",
#     2: "Boating",
#     3: "CarRacing",
#     4: "Concert",
#     5: "Conflict",
#     6: "Constructing",
#     7: "Cycling",
#     8: "Fire",
#     9: "Flood",
#     10: "Harvesting",
#     11: "Landslide",
#     12: "Mudslide",
#     13: "NonEvent",
#     14: "ParadeProtest",
#     15: "Party",
#     16: "Ploughing",
#     17: "PoliceChase",
#     18: "PostEarthquake",
#     19: "ReligiousActivity",
#     20: "Running",
#     21: "Soccer",
#     22: "Swimming",
#     23: "TrafficCollision",
#     24: "TrafficCongestion"
# }
#
# # Output for the tested predictions into CSV life
# csv_file_path = "predictionsC3DI.csv"
# fieldnames = ['Video_File', 'Predicted_Category', 'Predictions']
#
# # Creating Tensorflow session to work with the model
# with tf.compat.v1.Session() as sess:
#     # Importing graph of the model and restoring of weights
#     saver = tf.compat.v1.train.import_meta_graph('./model_C3D-Sport1M/c3d_best_model-49.meta')
#     saver.restore(sess, tf.train.latest_checkpoint("./model_C3D-Sport1M"))
#
#     # Definition of input and output tensors
#     input_tensor = sess.graph.get_tensor_by_name('input_x:0')
#     output_tensor = sess.graph.get_tensor_by_name('logits:0')
#
#     # List for saving up tested videos
#     videos_tested = []
#
#     # Opening of CSV file and writing tested data
#     with open(csv_file_path, mode='w', newline='') as csv_file:
#         writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
#         writer.writeheader()
#
#         # Iterating through every category and listing all videos in it
#         for category_index, category_name in video_categories.items():
#             category_path = f'./TestFULL/{category_name}/'
#             video_files = [file for file in os.listdir(category_path) if file.endswith('.mp4')]
#
#             # Check if there are videos in category
#             if len(video_files) < 1:
#                 print(f"Category  {category_name} do not have videos.")
#                 continue
#
#             # Random selection of 15 videos
#             selected_videos = random.sample(video_files, 15)
#             selected_video_paths = [os.path.join(category_path, video) for video in selected_videos]
#
#             # Iterating through every selected video and opening the video
#             for video_file in selected_video_paths:
#                 cap = cv2.VideoCapture(video_file)
#                 videos_tested.append(video_file)
#
#                 # Check for accessing the video
#                 if not cap.isOpened():
#                     print(f"Problem with opening video {video_file} .")
#                     continue
#
#                 # List for all predictions
#                 all_predictions = []
#
#                 # Reading frames from the opened video
#                 while True:
#                     ret, frame = cap.read()
#                     if not ret:
#                         break
#
#                     # Resizing the frame to smaller and creating frames to fit C3D requirements
#                     resized_frame = cv2.resize(frame, (112, 112))
#                     input_frames = np.expand_dims(np.array([resized_frame] * 16), axis=0)
#
#                     # Feeding the frame into the input tensor
#                     feed_dict = {input_tensor: input_frames, 'training:0': False}
#
#                     try:
#                         # Running inference with fed dictionary
#                         output_values = sess.run(output_tensor, feed_dict=feed_dict)
#
#                         # Appending the predictions for each frame
#                         all_predictions.append(output_values)
#
#                     # Check for processing
#                     except Exception as e:
#                         print(f"Error processing video {video_file}: {e}")
#
#                 # Accessing all predictions
#                 if all_predictions:
#                     # Calculating the average
#                     average_predictions = np.mean(all_predictions, axis=0)
#
#                     # Printing the results
#                     print("Predictions for ", video_file, " are: ", average_predictions)
#                     predicted_class_index = np.argmax(average_predictions)
#
#                     # Mapping predicted class index to category
#                     predicted_category = video_categories.get(predicted_class_index, "Unknown")
#                     print("Predicted category: ", predicted_class_index, "- ",  predicted_category)
#
#                     # Writing predictions to CSV file
#                     writer.writerow({
#                         'Video_File': video_file,
#                         'Predicted_Category': predicted_category,
#                         'Predictions': str(average_predictions)
#                     })
#
#                 # Releasing of the video capture
#                 cap.release()
#
#
# # Printing the list of tested videos
# print("Videos tested:", videos_tested)

###############################################################################################

import cv2
import numpy as np
import tensorflow as tf
import os
import csv

# Output for the tested predictions into CSV life
video_categories = {
    0: "Baseball",
    1: "Basketball",
    2: "Boating",
    3: "CarRacing",
    4: "Concert",
    5: "Conflict",
    6: "Constructing",
    7: "Cycling",
    8: "Fire",
    9: "Flood",
    10: "Harvesting",
    11: "Landslide",
    12: "Mudslide",
    13: "NonEvent",
    14: "ParadeProtest",
    15: "Party",
    16: "Ploughing",
    17: "PoliceChase",
    18: "PostEarthquake",
    19: "ReligiousActivity",
    20: "Running",
    21: "Soccer",
    22: "Swimming",
    23: "TrafficCollision",
    24: "TrafficCongestion"
}

csv_file_path = "predictionsC3DUCF.csv"  # You can change the file path as needed
fieldnames = ['Video_File', 'Predicted_Category', 'Predictions']

# Creating Tensorflow session to work with the model
with tf.compat.v1.Session() as sess:
    # Importing graph of the model and restoring of weights
    saver = tf.compat.v1.train.import_meta_graph('./model_C3D_UCF101/c3d_best_model-19.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./model_C3D_UCF101'))

    # Definition of input and output tensors
    input_tensor = sess.graph.get_tensor_by_name('input_x:0')
    output_tensor = sess.graph.get_tensor_by_name('logits:0')

    # List for saving up tested videos
    videos_tested = []
    
    # Randomly selected 15 unique videos from dataset and saved in list
    video_files_list = ['./TestFULL/Baseball/Baseball_008 .mp4', './TestFULL/Baseball/Baseball_035 .mp4', './TestFULL/Baseball/Baseball_005 .mp4', './TestFULL/Baseball/Baseball_021 .mp4', './TestFULL/Baseball/Baseball_012 .mp4', './TestFULL/Baseball/Baseball_050 .mp4', './TestFULL/Baseball/Baseball_019 .mp4', './TestFULL/Baseball/Baseball_022 .mp4', './TestFULL/Baseball/Baseball_001 .mp4', './TestFULL/Baseball/Baseball_046 .mp4', './TestFULL/Baseball/Baseball_028 .mp4', './TestFULL/Baseball/Baseball_026 .mp4', './TestFULL/Baseball/Baseball_023 .mp4', './TestFULL/Baseball/Baseball_044 .mp4', './TestFULL/Baseball/Baseball_017 .mp4', './TestFULL/Basketball/Basketball_047 .mp4', './TestFULL/Basketball/Basketball_043 .mp4', './TestFULL/Basketball/Basketball_001 .mp4', './TestFULL/Basketball/Basketball_009 .mp4', './TestFULL/Basketball/Basketball_024 .mp4', './TestFULL/Basketball/Basketball_020 .mp4', './TestFULL/Basketball/Basketball_040 .mp4', './TestFULL/Basketball/Basketball_005 .mp4', './TestFULL/Basketball/Basketball_029 .mp4', './TestFULL/Basketball/Basketball_044 .mp4', './TestFULL/Basketball/Basketball_017 .mp4', './TestFULL/Basketball/Basketball_002 .mp4', './TestFULL/Basketball/Basketball_003 .mp4', './TestFULL/Basketball/Basketball_034 .mp4', './TestFULL/Basketball/Basketball_008 .mp4', './TestFULL/Boating/Boating_040 .mp4', './TestFULL/Boating/Boating_017 .mp4', './TestFULL/Boating/Boating_048 .mp4', './TestFULL/Boating/Boating_035 .mp4', './TestFULL/Boating/Boating_026 .mp4', './TestFULL/Boating/Boating_005 .mp4', './TestFULL/Boating/Boating_050 .mp4', './TestFULL/Boating/Boating_051 .mp4', './TestFULL/Boating/Boating_032 .mp4', './TestFULL/Boating/Boating_019 .mp4', './TestFULL/Boating/Boating_036 .mp4', './TestFULL/Boating/Boating_018 .mp4', './TestFULL/Boating/Boating_025 .mp4', './TestFULL/Boating/Boating_031 .mp4', './TestFULL/Boating/Boating_008 .mp4', './TestFULL/CarRacing/CarRacing_008 .mp4', './TestFULL/CarRacing/CarRacing_009 .mp4', './TestFULL/CarRacing/CarRacing_012 .mp4', './TestFULL/CarRacing/CarRacing_015 .mp4', './TestFULL/CarRacing/CarRacing_010 .mp4', './TestFULL/CarRacing/CarRacing_002 .mp4', './TestFULL/CarRacing/CarRacing_013 .mp4', './TestFULL/CarRacing/CarRacing_004 .mp4', './TestFULL/CarRacing/CarRacing_017 .mp4', './TestFULL/CarRacing/CarRacing_014 .mp4', './TestFULL/CarRacing/CarRacing_016 .mp4', './TestFULL/CarRacing/CarRacing_003 .mp4', './TestFULL/CarRacing/CarRacing_001 .mp4', './TestFULL/CarRacing/CarRacing_007 .mp4', './TestFULL/CarRacing/CarRacing_019 .mp4', './TestFULL/Concert/Concert_047 .mp4', './TestFULL/Concert/Concert_039 .mp4', './TestFULL/Concert/Concert_040 .mp4', './TestFULL/Concert/Concert_011 .mp4', './TestFULL/Concert/Concert_037 .mp4', './TestFULL/Concert/Concert_008 .mp4', './TestFULL/Concert/Concert_012 .mp4', './TestFULL/Concert/Concert_019 .mp4', './TestFULL/Concert/Concert_042 .mp4', './TestFULL/Concert/Concert_041 .mp4', './TestFULL/Concert/Concert_001 .mp4', './TestFULL/Concert/Concert_016 .mp4', './TestFULL/Concert/Concert_003 .mp4', './TestFULL/Concert/Concert_045 .mp4', './TestFULL/Concert/Concert_046 .mp4', './TestFULL/Conflict/Conflict_017 .mp4', './TestFULL/Conflict/Conflict_019 .mp4', './TestFULL/Conflict/Conflict_025 .mp4', './TestFULL/Conflict/Conflict_001 .mp4', './TestFULL/Conflict/Conflict_013 .mp4', './TestFULL/Conflict/Conflict_003 .mp4', './TestFULL/Conflict/Conflict_009 .mp4', './TestFULL/Conflict/Conflict_002 .mp4', './TestFULL/Conflict/Conflict_007 .mp4', './TestFULL/Conflict/Conflict_004 .mp4', './TestFULL/Conflict/Conflict_012 .mp4', './TestFULL/Conflict/Conflict_023 .mp4', './TestFULL/Conflict/Conflict_010 .mp4', './TestFULL/Conflict/Conflict_011 .mp4', './TestFULL/Conflict/Conflict_020 .mp4', './TestFULL/Constructing/Constructing_058 .mp4', './TestFULL/Constructing/Constructing_027 .mp4', './TestFULL/Constructing/Constructing_010 .mp4', './TestFULL/Constructing/Constructing_044 .mp4', './TestFULL/Constructing/Constructing_034 .mp4', './TestFULL/Constructing/Constructing_038 .mp4', './TestFULL/Constructing/Constructing_039 .mp4', './TestFULL/Constructing/Constructing_054 .mp4', './TestFULL/Constructing/Constructing_032 .mp4', './TestFULL/Constructing/Constructing_020 .mp4', './TestFULL/Constructing/Constructing_047 .mp4', './TestFULL/Constructing/Constructing_033 .mp4', './TestFULL/Constructing/Constructing_011 .mp4', './TestFULL/Constructing/Constructing_041 .mp4', './TestFULL/Constructing/Constructing_043 .mp4', './TestFULL/Cycling/Cycling_025 .mp4', './TestFULL/Cycling/Cycling_039 .mp4', './TestFULL/Cycling/Cycling_024 .mp4', './TestFULL/Cycling/Cycling_015 .mp4', './TestFULL/Cycling/Cycling_044 .mp4', './TestFULL/Cycling/Cycling_019 .mp4', './TestFULL/Cycling/Cycling_016 .mp4', './TestFULL/Cycling/Cycling_053 .mp4', './TestFULL/Cycling/Cycling_026 .mp4', './TestFULL/Cycling/Cycling_006 .mp4', './TestFULL/Cycling/Cycling_048 .mp4', './TestFULL/Cycling/Cycling_047 .mp4', './TestFULL/Cycling/Cycling_051 .mp4', './TestFULL/Cycling/Cycling_005 .mp4', './TestFULL/Cycling/Cycling_045 .mp4', './TestFULL/Fire/Fire_014 .mp4', './TestFULL/Fire/Fire_018 .mp4', './TestFULL/Fire/Fire_034 .mp4', './TestFULL/Fire/Fire_044 .mp4', './TestFULL/Fire/Fire_021 .mp4', './TestFULL/Fire/Fire_036 .mp4', './TestFULL/Fire/Fire_007 .mp4', './TestFULL/Fire/Fire_028 .mp4', './TestFULL/Fire/Fire_053 .mp4', './TestFULL/Fire/Fire_054 .mp4', './TestFULL/Fire/Fire_012 .mp4', './TestFULL/Fire/Fire_048 .mp4', './TestFULL/Fire/Fire_027 .mp4', './TestFULL/Fire/Fire_017 .mp4', './TestFULL/Fire/Fire_024 .mp4', './TestFULL/Flood/Flood_011 .mp4', './TestFULL/Flood/Flood_031 .mp4', './TestFULL/Flood/Flood_032 .mp4', './TestFULL/Flood/Flood_010 .mp4', './TestFULL/Flood/Flood_016 .mp4', './TestFULL/Flood/Flood_020 .mp4', './TestFULL/Flood/Flood_038 .mp4', './TestFULL/Flood/Flood_040 .mp4', './TestFULL/Flood/Flood_001 .mp4', './TestFULL/Flood/Flood_009 .mp4', './TestFULL/Flood/Flood_043 .mp4', './TestFULL/Flood/Flood_041 .mp4', './TestFULL/Flood/Flood_029 .mp4', './TestFULL/Flood/Flood_049 .mp4', './TestFULL/Flood/Flood_021 .mp4', './TestFULL/Harvesting/Harvesting_044 .mp4', './TestFULL/Harvesting/Harvesting_056 .mp4', './TestFULL/Harvesting/Harvesting_008 .mp4', './TestFULL/Harvesting/Harvesting_039 .mp4', './TestFULL/Harvesting/Harvesting_037 .mp4', './TestFULL/Harvesting/Harvesting_021 .mp4', './TestFULL/Harvesting/Harvesting_015 .mp4', './TestFULL/Harvesting/Harvesting_023 .mp4', './TestFULL/Harvesting/Harvesting_033 .mp4', './TestFULL/Harvesting/Harvesting_019 .mp4', './TestFULL/Harvesting/Harvesting_038 .mp4', './TestFULL/Harvesting/Harvesting_047 .mp4', './TestFULL/Harvesting/Harvesting_061 .mp4', './TestFULL/Harvesting/Harvesting_002 .mp4', './TestFULL/Harvesting/Harvesting_017 .mp4', './TestFULL/Landslide/Landslide_028 .mp4', './TestFULL/Landslide/Landslide_012 .mp4', './TestFULL/Landslide/Landslide_046 .mp4', './TestFULL/Landslide/Landslide_007 .mp4', './TestFULL/Landslide/Landslide_008 .mp4', './TestFULL/Landslide/Landslide_033 .mp4', './TestFULL/Landslide/Landslide_027 .mp4', './TestFULL/Landslide/Landslide_017 .mp4', './TestFULL/Landslide/Landslide_032 .mp4', './TestFULL/Landslide/Landslide_030 .mp4', './TestFULL/Landslide/Landslide_018 .mp4', './TestFULL/Landslide/Landslide_040 .mp4', './TestFULL/Landslide/Landslide_001 .mp4', './TestFULL/Landslide/Landslide_049 .mp4', './TestFULL/Landslide/Landslide_037 .mp4', './TestFULL/Mudslide/Mudslide_032 .mp4', './TestFULL/Mudslide/Mudslide_008 .mp4', './TestFULL/Mudslide/Mudslide_044 .mp4', './TestFULL/Mudslide/Mudslide_012 .mp4', './TestFULL/Mudslide/Mudslide_035 .mp4', './TestFULL/Mudslide/Mudslide_021 .mp4', './TestFULL/Mudslide/Mudslide_011 .mp4', './TestFULL/Mudslide/Mudslide_036 .mp4', './TestFULL/Mudslide/Mudslide_050 .mp4', './TestFULL/Mudslide/Mudslide_031 .mp4', './TestFULL/Mudslide/Mudslide_042 .mp4', './TestFULL/Mudslide/Mudslide_039 .mp4', './TestFULL/Mudslide/Mudslide_028 .mp4', './TestFULL/Mudslide/Mudslide_001 .mp4', './TestFULL/Mudslide/Mudslide_048 .mp4', './TestFULL/NonEvent/NonEvent_051 .mp4', './TestFULL/NonEvent/NonEvent_093 .mp4', './TestFULL/NonEvent/NonEvent_133 .mp4', './TestFULL/NonEvent/NonEvent_094 .mp4', './TestFULL/NonEvent/NonEvent_037 .mp4', './TestFULL/NonEvent/NonEvent_042 .mp4', './TestFULL/NonEvent/NonEvent_141 .mp4', './TestFULL/NonEvent/NonEvent_013 .mp4', './TestFULL/NonEvent/NonEvent_146 .mp4', './TestFULL/NonEvent/NonEvent_154 .mp4', './TestFULL/NonEvent/NonEvent_078 .mp4', './TestFULL/NonEvent/NonEvent_019 .mp4', './TestFULL/NonEvent/NonEvent_137 .mp4', './TestFULL/NonEvent/NonEvent_096 .mp4', './TestFULL/NonEvent/NonEvent_038 .mp4', './TestFULL/ParadeProtest/ParadeProtest_022 .mp4', './TestFULL/ParadeProtest/ParadeProtest_013 .mp4', './TestFULL/ParadeProtest/ParadeProtest_028 .mp4', './TestFULL/ParadeProtest/ParadeProtest_029 .mp4', './TestFULL/ParadeProtest/ParadeProtest_021 .mp4', './TestFULL/ParadeProtest/ParadeProtest_026 .mp4', './TestFULL/ParadeProtest/ParadeProtest_017 .mp4', './TestFULL/ParadeProtest/ParadeProtest_031 .mp4', './TestFULL/ParadeProtest/ParadeProtest_003 .mp4', './TestFULL/ParadeProtest/ParadeProtest_049 .mp4', './TestFULL/ParadeProtest/ParadeProtest_023 .mp4', './TestFULL/ParadeProtest/ParadeProtest_018 .mp4', './TestFULL/ParadeProtest/ParadeProtest_042 .mp4', './TestFULL/ParadeProtest/ParadeProtest_009 .mp4', './TestFULL/ParadeProtest/ParadeProtest_019 .mp4', './TestFULL/Party/Party_001 .mp4', './TestFULL/Party/Party_002 .mp4', './TestFULL/Party/Party_009 .mp4', './TestFULL/Party/Party_005 .mp4', './TestFULL/Party/Party_017 .mp4', './TestFULL/Party/Party_015 .mp4', './TestFULL/Party/Party_050 .mp4', './TestFULL/Party/Party_036 .mp4', './TestFULL/Party/Party_016 .mp4', './TestFULL/Party/Party_028 .mp4', './TestFULL/Party/Party_035 .mp4', './TestFULL/Party/Party_039 .mp4', './TestFULL/Party/Party_040 .mp4', './TestFULL/Party/Party_029 .mp4', './TestFULL/Party/Party_007 .mp4', './TestFULL/Ploughing/Ploughing_047 .mp4', './TestFULL/Ploughing/Ploughing_044 .mp4', './TestFULL/Ploughing/Ploughing_030 .mp4', './TestFULL/Ploughing/Ploughing_011 .mp4', './TestFULL/Ploughing/Ploughing_001 .mp4', './TestFULL/Ploughing/Ploughing_021 .mp4', './TestFULL/Ploughing/Ploughing_005 .mp4', './TestFULL/Ploughing/Ploughing_043 .mp4', './TestFULL/Ploughing/Ploughing_023 .mp4', './TestFULL/Ploughing/Ploughing_035 .mp4', './TestFULL/Ploughing/Ploughing_033 .mp4', './TestFULL/Ploughing/Ploughing_024 .mp4', './TestFULL/Ploughing/Ploughing_045 .mp4', './TestFULL/Ploughing/Ploughing_031 .mp4', './TestFULL/Ploughing/Ploughing_041 .mp4', './TestFULL/PoliceChase/PoliceChase_001 .mp4', './TestFULL/PoliceChase/PoliceChase_033 .mp4', './TestFULL/PoliceChase/PoliceChase_014 .mp4', './TestFULL/PoliceChase/PoliceChase_034 .mp4', './TestFULL/PoliceChase/PoliceChase_010 .mp4', './TestFULL/PoliceChase/PoliceChase_047 .mp4', './TestFULL/PoliceChase/PoliceChase_007 .mp4', './TestFULL/PoliceChase/PoliceChase_012 .mp4', './TestFULL/PoliceChase/PoliceChase_031 .mp4', './TestFULL/PoliceChase/PoliceChase_011 .mp4', './TestFULL/PoliceChase/PoliceChase_029 .mp4', './TestFULL/PoliceChase/PoliceChase_039 .mp4', './TestFULL/PoliceChase/PoliceChase_048 .mp4', './TestFULL/PoliceChase/PoliceChase_044 .mp4', './TestFULL/PoliceChase/PoliceChase_021 .mp4', './TestFULL/PostEarthquake/PostEarthquake_030 .mp4', './TestFULL/PostEarthquake/PostEarthquake_028 .mp4', './TestFULL/PostEarthquake/PostEarthquake_007 .mp4', './TestFULL/PostEarthquake/PostEarthquake_029 .mp4', './TestFULL/PostEarthquake/PostEarthquake_015 .mp4', './TestFULL/PostEarthquake/PostEarthquake_003 .mp4', './TestFULL/PostEarthquake/PostEarthquake_004 .mp4', './TestFULL/PostEarthquake/PostEarthquake_038 .mp4', './TestFULL/PostEarthquake/PostEarthquake_012 .mp4', './TestFULL/PostEarthquake/PostEarthquake_010 .mp4', './TestFULL/PostEarthquake/PostEarthquake_005 .mp4', './TestFULL/PostEarthquake/PostEarthquake_042 .mp4', './TestFULL/PostEarthquake/PostEarthquake_047 .mp4', './TestFULL/PostEarthquake/PostEarthquake_039 .mp4', './TestFULL/PostEarthquake/PostEarthquake_016 .mp4', './TestFULL/ReligiousActivity/ReligiousActivity_030 .mp4', './TestFULL/ReligiousActivity/ReligiousActivity_003 .mp4', './TestFULL/ReligiousActivity/ReligiousActivity_033 .mp4', './TestFULL/ReligiousActivity/ReligiousActivity_047 .mp4', './TestFULL/ReligiousActivity/ReligiousActivity_017 .mp4', './TestFULL/ReligiousActivity/ReligiousActivity_048 .mp4', './TestFULL/ReligiousActivity/ReligiousActivity_045 .mp4', './TestFULL/ReligiousActivity/ReligiousActivity_009 .mp4', './TestFULL/ReligiousActivity/ReligiousActivity_029 .mp4', './TestFULL/ReligiousActivity/ReligiousActivity_001 .mp4', './TestFULL/ReligiousActivity/ReligiousActivity_034 .mp4', './TestFULL/ReligiousActivity/ReligiousActivity_018 .mp4', './TestFULL/ReligiousActivity/ReligiousActivity_043 .mp4', './TestFULL/ReligiousActivity/ReligiousActivity_025 .mp4', './TestFULL/ReligiousActivity/ReligiousActivity_010 .mp4', './TestFULL/Running/Running_059 .mp4', './TestFULL/Running/Running_071 .mp4', './TestFULL/Running/Running_084 .mp4', './TestFULL/Running/Running_052 .mp4', './TestFULL/Running/Running_068 .mp4', './TestFULL/Running/Running_060 .mp4', './TestFULL/Running/Running_011 .mp4', './TestFULL/Running/Running_025 .mp4', './TestFULL/Running/Running_051 .mp4', './TestFULL/Running/Running_030 .mp4', './TestFULL/Running/Running_079 .mp4', './TestFULL/Running/Running_037 .mp4', './TestFULL/Running/Running_038 .mp4', './TestFULL/Running/Running_023 .mp4', './TestFULL/Running/Running_004 .mp4', './TestFULL/Soccer/Soccer_022 .mp4', './TestFULL/Soccer/Soccer_038 .mp4', './TestFULL/Soccer/Soccer_056 .mp4', './TestFULL/Soccer/Soccer_001 .mp4', './TestFULL/Soccer/Soccer_043 .mp4', './TestFULL/Soccer/Soccer_004 .mp4', './TestFULL/Soccer/Soccer_025 .mp4', './TestFULL/Soccer/Soccer_041 .mp4', './TestFULL/Soccer/Soccer_011 .mp4', './TestFULL/Soccer/Soccer_031 .mp4', './TestFULL/Soccer/Soccer_035 .mp4', './TestFULL/Soccer/Soccer_029 .mp4', './TestFULL/Soccer/Soccer_014 .mp4', './TestFULL/Soccer/Soccer_036 .mp4', './TestFULL/Soccer/Soccer_023 .mp4', './TestFULL/Swimming/Swimming_048 .mp4', './TestFULL/Swimming/Swimming_009 .mp4', './TestFULL/Swimming/Swimming_031 .mp4', './TestFULL/Swimming/Swimming_010 .mp4', './TestFULL/Swimming/Swimming_020 .mp4', './TestFULL/Swimming/Swimming_024 .mp4', './TestFULL/Swimming/Swimming_014 .mp4', './TestFULL/Swimming/Swimming_050 .mp4', './TestFULL/Swimming/Swimming_021 .mp4', './TestFULL/Swimming/Swimming_012 .mp4', './TestFULL/Swimming/Swimming_042 .mp4', './TestFULL/Swimming/Swimming_027 .mp4', './TestFULL/Swimming/Swimming_025 .mp4', './TestFULL/Swimming/Swimming_011 .mp4', './TestFULL/Swimming/Swimming_007 .mp4', './TestFULL/TrafficCollision/TrafficCollision_022 .mp4', './TestFULL/TrafficCollision/TrafficCollision_030 .mp4', './TestFULL/TrafficCollision/TrafficCollision_012 .mp4', './TestFULL/TrafficCollision/TrafficCollision_043 .mp4', './TestFULL/TrafficCollision/TrafficCollision_031 .mp4', './TestFULL/TrafficCollision/TrafficCollision_040 .mp4', './TestFULL/TrafficCollision/TrafficCollision_046 .mp4', './TestFULL/TrafficCollision/TrafficCollision_023 .mp4', './TestFULL/TrafficCollision/TrafficCollision_020 .mp4', './TestFULL/TrafficCollision/TrafficCollision_016 .mp4', './TestFULL/TrafficCollision/TrafficCollision_037 .mp4', './TestFULL/TrafficCollision/TrafficCollision_001 .mp4', './TestFULL/TrafficCollision/TrafficCollision_005 .mp4', './TestFULL/TrafficCollision/TrafficCollision_033 .mp4', './TestFULL/TrafficCollision/TrafficCollision_024 .mp4', './TestFULL/TrafficCongestion/TrafficCongestion_013 .mp4', './TestFULL/TrafficCongestion/TrafficCongestion_033 .mp4', './TestFULL/TrafficCongestion/TrafficCongestion_044 .mp4', './TestFULL/TrafficCongestion/TrafficCongestion_028 .mp4', './TestFULL/TrafficCongestion/TrafficCongestion_050 .mp4', './TestFULL/TrafficCongestion/TrafficCongestion_014 .mp4', './TestFULL/TrafficCongestion/TrafficCongestion_021 .mp4', './TestFULL/TrafficCongestion/TrafficCongestion_048 .mp4', './TestFULL/TrafficCongestion/TrafficCongestion_043 .mp4', './TestFULL/TrafficCongestion/TrafficCongestion_049 .mp4', './TestFULL/TrafficCongestion/TrafficCongestion_039 .mp4', './TestFULL/TrafficCongestion/TrafficCongestion_004 .mp4', './TestFULL/TrafficCongestion/TrafficCongestion_005 .mp4', './TestFULL/TrafficCongestion/TrafficCongestion_010 .mp4', './TestFULL/TrafficCongestion/TrafficCongestion_023 .mp4']

    # Opening of CSV file and writing tested data
    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        # Iterating through every selected video and opening the video.
        for video_file in video_files_list:
            # Check if there are videos in category
            if not os.path.exists(video_file):
                print(f"Warning: Video file {video_file} not found. Skipping.")
                continue

            cap = cv2.VideoCapture(video_file)
            videos_tested.append(video_file)

            if not cap.isOpened():
                print(f"Error: Couldn't open video file {video_file}")
                continue

            # List for all predictions
            all_predictions = []

            # Read frames from the video
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Reading frames from the opened video and updating the frames for model needs      
                resized_frame = cv2.resize(frame, (112, 112))
                input_frames = np.expand_dims(np.array([resized_frame] * 16), axis=0)

                # Feeding the frame into the input tensor
                feed_dict = {input_tensor: input_frames, 'training:0': False}

                try:
                    # Running inference with fed dictionary
                    output_values = sess.run(output_tensor, feed_dict=feed_dict)

                   # Appending the predictions for each frame
                    all_predictions.append(output_values)

                # Check for processing
                except Exception as e:
                    print(f"Error processing video {video_file}: {e}")

           # Accessing all predictions
            if all_predictions:
                 # Calculating the average
                average_predictions = np.mean(all_predictions, axis=0)

                 # Printing the results
                print("Predictions for ", video_file, " are: ", average_predictions)
                predicted_class_index = np.argmax(average_predictions)

                # Mapping predicted class index to category
                predicted_category = video_categories.get(predicted_class_index, "Unknown")
                print("Predicted category: ", predicted_class_index, "- ", predicted_category)

                # Writing predictions to CSV file
                writer.writerow({
                    'Video_File': video_file,
                    'Predicted_Category': predicted_category,
                    'Predictions': str(average_predictions)
                })

            # Releasing of the video capture
            cap.release()

# Printing the list of tested videos
print("Videos tested:", videos_tested)


