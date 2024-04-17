# Imports of libraries
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import csv

# Dictionary with saved category names
category_dict = {
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
# List of randomly selected 15 unique videos from dataset
video_names = ['./TestFULL/Baseball/Baseball_008 .mp4', './TestFULL/Baseball/Baseball_035 .mp4',
               './TestFULL/Baseball/Baseball_005 .mp4', './TestFULL/Baseball/Baseball_021 .mp4',
               './TestFULL/Baseball/Baseball_012 .mp4', './TestFULL/Baseball/Baseball_050 .mp4',
               './TestFULL/Baseball/Baseball_019 .mp4', './TestFULL/Baseball/Baseball_022 .mp4',
               './TestFULL/Baseball/Baseball_001 .mp4', './TestFULL/Baseball/Baseball_046 .mp4',
               './TestFULL/Baseball/Baseball_028 .mp4', './TestFULL/Baseball/Baseball_026 .mp4',
               './TestFULL/Baseball/Baseball_023 .mp4', './TestFULL/Baseball/Baseball_044 .mp4',
               './TestFULL/Baseball/Baseball_017 .mp4', './TestFULL/Basketball/Basketball_047 .mp4',
               './TestFULL/Basketball/Basketball_043 .mp4', './TestFULL/Basketball/Basketball_001 .mp4',
               './TestFULL/Basketball/Basketball_009 .mp4', './TestFULL/Basketball/Basketball_024 .mp4',
               './TestFULL/Basketball/Basketball_020 .mp4', './TestFULL/Basketball/Basketball_040 .mp4',
               './TestFULL/Basketball/Basketball_005 .mp4', './TestFULL/Basketball/Basketball_029 .mp4',
               './TestFULL/Basketball/Basketball_044 .mp4', './TestFULL/Basketball/Basketball_017 .mp4',
               './TestFULL/Basketball/Basketball_002 .mp4', './TestFULL/Basketball/Basketball_003 .mp4',
               './TestFULL/Basketball/Basketball_034 .mp4', './TestFULL/Basketball/Basketball_008 .mp4',
               './TestFULL/Boating/Boating_040 .mp4', './TestFULL/Boating/Boating_017 .mp4',
               './TestFULL/Boating/Boating_048 .mp4', './TestFULL/Boating/Boating_035 .mp4',
               './TestFULL/Boating/Boating_026 .mp4', './TestFULL/Boating/Boating_005 .mp4',
               './TestFULL/Boating/Boating_050 .mp4', './TestFULL/Boating/Boating_051 .mp4',
               './TestFULL/Boating/Boating_032 .mp4', './TestFULL/Boating/Boating_019 .mp4',
               './TestFULL/Boating/Boating_036 .mp4', './TestFULL/Boating/Boating_018 .mp4',
               './TestFULL/Boating/Boating_025 .mp4', './TestFULL/Boating/Boating_031 .mp4',
               './TestFULL/Boating/Boating_008 .mp4', './TestFULL/CarRacing/CarRacing_008 .mp4',
               './TestFULL/CarRacing/CarRacing_009 .mp4', './TestFULL/CarRacing/CarRacing_012 .mp4',
               './TestFULL/CarRacing/CarRacing_015 .mp4', './TestFULL/CarRacing/CarRacing_010 .mp4',
               './TestFULL/CarRacing/CarRacing_002 .mp4', './TestFULL/CarRacing/CarRacing_013 .mp4',
               './TestFULL/CarRacing/CarRacing_004 .mp4', './TestFULL/CarRacing/CarRacing_017 .mp4',
               './TestFULL/CarRacing/CarRacing_014 .mp4', './TestFULL/CarRacing/CarRacing_016 .mp4',
               './TestFULL/CarRacing/CarRacing_003 .mp4', './TestFULL/CarRacing/CarRacing_001 .mp4',
               './TestFULL/CarRacing/CarRacing_007 .mp4', './TestFULL/CarRacing/CarRacing_019 .mp4',
               './TestFULL/Concert/Concert_047 .mp4', './TestFULL/Concert/Concert_039 .mp4',
               './TestFULL/Concert/Concert_040 .mp4', './TestFULL/Concert/Concert_011 .mp4',
               './TestFULL/Concert/Concert_037 .mp4', './TestFULL/Concert/Concert_008 .mp4',
               './TestFULL/Concert/Concert_012 .mp4', './TestFULL/Concert/Concert_019 .mp4',
               './TestFULL/Concert/Concert_042 .mp4', './TestFULL/Concert/Concert_041 .mp4',
               './TestFULL/Concert/Concert_001 .mp4', './TestFULL/Concert/Concert_016 .mp4',
               './TestFULL/Concert/Concert_003 .mp4', './TestFULL/Concert/Concert_045 .mp4',
               './TestFULL/Concert/Concert_046 .mp4', './TestFULL/Conflict/Conflict_017 .mp4',
               './TestFULL/Conflict/Conflict_019 .mp4', './TestFULL/Conflict/Conflict_025 .mp4',
               './TestFULL/Conflict/Conflict_001 .mp4', './TestFULL/Conflict/Conflict_013 .mp4',
               './TestFULL/Conflict/Conflict_003 .mp4', './TestFULL/Conflict/Conflict_009 .mp4',
               './TestFULL/Conflict/Conflict_002 .mp4', './TestFULL/Conflict/Conflict_007 .mp4',
               './TestFULL/Conflict/Conflict_004 .mp4', './TestFULL/Conflict/Conflict_012 .mp4',
               './TestFULL/Conflict/Conflict_023 .mp4', './TestFULL/Conflict/Conflict_010 .mp4',
               './TestFULL/Conflict/Conflict_011 .mp4', './TestFULL/Conflict/Conflict_020 .mp4',
               './TestFULL/Constructing/Constructing_058 .mp4', './TestFULL/Constructing/Constructing_027 .mp4',
               './TestFULL/Constructing/Constructing_010 .mp4', './TestFULL/Constructing/Constructing_044 .mp4',
               './TestFULL/Constructing/Constructing_034 .mp4', './TestFULL/Constructing/Constructing_038 .mp4',
               './TestFULL/Constructing/Constructing_039 .mp4', './TestFULL/Constructing/Constructing_054 .mp4',
               './TestFULL/Constructing/Constructing_032 .mp4', './TestFULL/Constructing/Constructing_020 .mp4',
               './TestFULL/Constructing/Constructing_047 .mp4', './TestFULL/Constructing/Constructing_033 .mp4',
               './TestFULL/Constructing/Constructing_011 .mp4', './TestFULL/Constructing/Constructing_041 .mp4',
               './TestFULL/Constructing/Constructing_043 .mp4', './TestFULL/Cycling/Cycling_025 .mp4',
               './TestFULL/Cycling/Cycling_039 .mp4', './TestFULL/Cycling/Cycling_024 .mp4',
               './TestFULL/Cycling/Cycling_015 .mp4', './TestFULL/Cycling/Cycling_044 .mp4',
               './TestFULL/Cycling/Cycling_019 .mp4', './TestFULL/Cycling/Cycling_016 .mp4',
               './TestFULL/Cycling/Cycling_053 .mp4', './TestFULL/Cycling/Cycling_026 .mp4',
               './TestFULL/Cycling/Cycling_006 .mp4', './TestFULL/Cycling/Cycling_048 .mp4',
               './TestFULL/Cycling/Cycling_047 .mp4', './TestFULL/Cycling/Cycling_051 .mp4',
               './TestFULL/Cycling/Cycling_005 .mp4', './TestFULL/Cycling/Cycling_045 .mp4',
               './TestFULL/Fire/Fire_014 .mp4', './TestFULL/Fire/Fire_018 .mp4', './TestFULL/Fire/Fire_034 .mp4',
               './TestFULL/Fire/Fire_044 .mp4', './TestFULL/Fire/Fire_021 .mp4', './TestFULL/Fire/Fire_036 .mp4',
               './TestFULL/Fire/Fire_007 .mp4', './TestFULL/Fire/Fire_028 .mp4', './TestFULL/Fire/Fire_053 .mp4',
               './TestFULL/Fire/Fire_054 .mp4', './TestFULL/Fire/Fire_012 .mp4', './TestFULL/Fire/Fire_048 .mp4',
               './TestFULL/Fire/Fire_027 .mp4', './TestFULL/Fire/Fire_017 .mp4', './TestFULL/Fire/Fire_024 .mp4',
               './TestFULL/Flood/Flood_011 .mp4', './TestFULL/Flood/Flood_031 .mp4', './TestFULL/Flood/Flood_032 .mp4',
               './TestFULL/Flood/Flood_010 .mp4', './TestFULL/Flood/Flood_016 .mp4', './TestFULL/Flood/Flood_020 .mp4',
               './TestFULL/Flood/Flood_038 .mp4', './TestFULL/Flood/Flood_040 .mp4', './TestFULL/Flood/Flood_001 .mp4',
               './TestFULL/Flood/Flood_009 .mp4', './TestFULL/Flood/Flood_043 .mp4', './TestFULL/Flood/Flood_041 .mp4',
               './TestFULL/Flood/Flood_029 .mp4', './TestFULL/Flood/Flood_049 .mp4', './TestFULL/Flood/Flood_021 .mp4',
               './TestFULL/Harvesting/Harvesting_044 .mp4', './TestFULL/Harvesting/Harvesting_056 .mp4',
               './TestFULL/Harvesting/Harvesting_008 .mp4', './TestFULL/Harvesting/Harvesting_039 .mp4',
               './TestFULL/Harvesting/Harvesting_037 .mp4', './TestFULL/Harvesting/Harvesting_021 .mp4',
               './TestFULL/Harvesting/Harvesting_015 .mp4', './TestFULL/Harvesting/Harvesting_023 .mp4',
               './TestFULL/Harvesting/Harvesting_033 .mp4', './TestFULL/Harvesting/Harvesting_019 .mp4',
               './TestFULL/Harvesting/Harvesting_038 .mp4', './TestFULL/Harvesting/Harvesting_047 .mp4',
               './TestFULL/Harvesting/Harvesting_061 .mp4', './TestFULL/Harvesting/Harvesting_002 .mp4',
               './TestFULL/Harvesting/Harvesting_017 .mp4', './TestFULL/Landslide/Landslide_028 .mp4',
               './TestFULL/Landslide/Landslide_012 .mp4', './TestFULL/Landslide/Landslide_046 .mp4',
               './TestFULL/Landslide/Landslide_007 .mp4', './TestFULL/Landslide/Landslide_008 .mp4',
               './TestFULL/Landslide/Landslide_033 .mp4', './TestFULL/Landslide/Landslide_027 .mp4',
               './TestFULL/Landslide/Landslide_017 .mp4', './TestFULL/Landslide/Landslide_032 .mp4',
               './TestFULL/Landslide/Landslide_030 .mp4', './TestFULL/Landslide/Landslide_018 .mp4',
               './TestFULL/Landslide/Landslide_040 .mp4', './TestFULL/Landslide/Landslide_001 .mp4',
               './TestFULL/Landslide/Landslide_049 .mp4', './TestFULL/Landslide/Landslide_037 .mp4',
               './TestFULL/Mudslide/Mudslide_032 .mp4', './TestFULL/Mudslide/Mudslide_008 .mp4',
               './TestFULL/Mudslide/Mudslide_044 .mp4', './TestFULL/Mudslide/Mudslide_012 .mp4',
               './TestFULL/Mudslide/Mudslide_035 .mp4', './TestFULL/Mudslide/Mudslide_021 .mp4',
               './TestFULL/Mudslide/Mudslide_011 .mp4', './TestFULL/Mudslide/Mudslide_036 .mp4',
               './TestFULL/Mudslide/Mudslide_050 .mp4', './TestFULL/Mudslide/Mudslide_031 .mp4',
               './TestFULL/Mudslide/Mudslide_042 .mp4', './TestFULL/Mudslide/Mudslide_039 .mp4',
               './TestFULL/Mudslide/Mudslide_028 .mp4', './TestFULL/Mudslide/Mudslide_001 .mp4',
               './TestFULL/Mudslide/Mudslide_048 .mp4', './TestFULL/NonEvent/NonEvent_051 .mp4',
               './TestFULL/NonEvent/NonEvent_093 .mp4', './TestFULL/NonEvent/NonEvent_133 .mp4',
               './TestFULL/NonEvent/NonEvent_094 .mp4', './TestFULL/NonEvent/NonEvent_037 .mp4',
               './TestFULL/NonEvent/NonEvent_042 .mp4', './TestFULL/NonEvent/NonEvent_141 .mp4',
               './TestFULL/NonEvent/NonEvent_013 .mp4', './TestFULL/NonEvent/NonEvent_146 .mp4',
               './TestFULL/NonEvent/NonEvent_154 .mp4', './TestFULL/NonEvent/NonEvent_078 .mp4',
               './TestFULL/NonEvent/NonEvent_019 .mp4', './TestFULL/NonEvent/NonEvent_137 .mp4',
               './TestFULL/NonEvent/NonEvent_096 .mp4', './TestFULL/NonEvent/NonEvent_038 .mp4',
               './TestFULL/ParadeProtest/ParadeProtest_022 .mp4', './TestFULL/ParadeProtest/ParadeProtest_013 .mp4',
               './TestFULL/ParadeProtest/ParadeProtest_028 .mp4', './TestFULL/ParadeProtest/ParadeProtest_029 .mp4',
               './TestFULL/ParadeProtest/ParadeProtest_021 .mp4', './TestFULL/ParadeProtest/ParadeProtest_026 .mp4',
               './TestFULL/ParadeProtest/ParadeProtest_017 .mp4', './TestFULL/ParadeProtest/ParadeProtest_031 .mp4',
               './TestFULL/ParadeProtest/ParadeProtest_003 .mp4', './TestFULL/ParadeProtest/ParadeProtest_049 .mp4',
               './TestFULL/ParadeProtest/ParadeProtest_023 .mp4', './TestFULL/ParadeProtest/ParadeProtest_018 .mp4',
               './TestFULL/ParadeProtest/ParadeProtest_042 .mp4', './TestFULL/ParadeProtest/ParadeProtest_009 .mp4',
               './TestFULL/ParadeProtest/ParadeProtest_019 .mp4', './TestFULL/Party/Party_001 .mp4',
               './TestFULL/Party/Party_002 .mp4', './TestFULL/Party/Party_009 .mp4', './TestFULL/Party/Party_005 .mp4',
               './TestFULL/Party/Party_017 .mp4', './TestFULL/Party/Party_015 .mp4', './TestFULL/Party/Party_050 .mp4',
               './TestFULL/Party/Party_036 .mp4', './TestFULL/Party/Party_016 .mp4', './TestFULL/Party/Party_028 .mp4',
               './TestFULL/Party/Party_035 .mp4', './TestFULL/Party/Party_039 .mp4', './TestFULL/Party/Party_040 .mp4',
               './TestFULL/Party/Party_029 .mp4', './TestFULL/Party/Party_007 .mp4',
               './TestFULL/Ploughing/Ploughing_047 .mp4', './TestFULL/Ploughing/Ploughing_044 .mp4',
               './TestFULL/Ploughing/Ploughing_030 .mp4', './TestFULL/Ploughing/Ploughing_011 .mp4',
               './TestFULL/Ploughing/Ploughing_001 .mp4', './TestFULL/Ploughing/Ploughing_021 .mp4',
               './TestFULL/Ploughing/Ploughing_005 .mp4', './TestFULL/Ploughing/Ploughing_043 .mp4',
               './TestFULL/Ploughing/Ploughing_023 .mp4', './TestFULL/Ploughing/Ploughing_035 .mp4',
               './TestFULL/Ploughing/Ploughing_033 .mp4', './TestFULL/Ploughing/Ploughing_024 .mp4',
               './TestFULL/Ploughing/Ploughing_045 .mp4', './TestFULL/Ploughing/Ploughing_031 .mp4',
               './TestFULL/Ploughing/Ploughing_041 .mp4', './TestFULL/PoliceChase/PoliceChase_001 .mp4',
               './TestFULL/PoliceChase/PoliceChase_033 .mp4', './TestFULL/PoliceChase/PoliceChase_014 .mp4',
               './TestFULL/PoliceChase/PoliceChase_034 .mp4', './TestFULL/PoliceChase/PoliceChase_010 .mp4',
               './TestFULL/PoliceChase/PoliceChase_047 .mp4', './TestFULL/PoliceChase/PoliceChase_007 .mp4',
               './TestFULL/PoliceChase/PoliceChase_012 .mp4', './TestFULL/PoliceChase/PoliceChase_031 .mp4',
               './TestFULL/PoliceChase/PoliceChase_011 .mp4', './TestFULL/PoliceChase/PoliceChase_029 .mp4',
               './TestFULL/PoliceChase/PoliceChase_039 .mp4', './TestFULL/PoliceChase/PoliceChase_048 .mp4',
               './TestFULL/PoliceChase/PoliceChase_044 .mp4', './TestFULL/PoliceChase/PoliceChase_021 .mp4',
               './TestFULL/PostEarthquake/PostEarthquake_030 .mp4', './TestFULL/PostEarthquake/PostEarthquake_028 .mp4',
               './TestFULL/PostEarthquake/PostEarthquake_007 .mp4', './TestFULL/PostEarthquake/PostEarthquake_029 .mp4',
               './TestFULL/PostEarthquake/PostEarthquake_015 .mp4', './TestFULL/PostEarthquake/PostEarthquake_003 .mp4',
               './TestFULL/PostEarthquake/PostEarthquake_004 .mp4', './TestFULL/PostEarthquake/PostEarthquake_038 .mp4',
               './TestFULL/PostEarthquake/PostEarthquake_012 .mp4', './TestFULL/PostEarthquake/PostEarthquake_010 .mp4',
               './TestFULL/PostEarthquake/PostEarthquake_005 .mp4', './TestFULL/PostEarthquake/PostEarthquake_042 .mp4',
               './TestFULL/PostEarthquake/PostEarthquake_047 .mp4', './TestFULL/PostEarthquake/PostEarthquake_039 .mp4',
               './TestFULL/PostEarthquake/PostEarthquake_016 .mp4',
               './TestFULL/ReligiousActivity/ReligiousActivity_030 .mp4',
               './TestFULL/ReligiousActivity/ReligiousActivity_003 .mp4',
               './TestFULL/ReligiousActivity/ReligiousActivity_033 .mp4',
               './TestFULL/ReligiousActivity/ReligiousActivity_047 .mp4',
               './TestFULL/ReligiousActivity/ReligiousActivity_017 .mp4',
               './TestFULL/ReligiousActivity/ReligiousActivity_048 .mp4',
               './TestFULL/ReligiousActivity/ReligiousActivity_045 .mp4',
               './TestFULL/ReligiousActivity/ReligiousActivity_009 .mp4',
               './TestFULL/ReligiousActivity/ReligiousActivity_029 .mp4',
               './TestFULL/ReligiousActivity/ReligiousActivity_001 .mp4',
               './TestFULL/ReligiousActivity/ReligiousActivity_034 .mp4',
               './TestFULL/ReligiousActivity/ReligiousActivity_018 .mp4',
               './TestFULL/ReligiousActivity/ReligiousActivity_043 .mp4',
               './TestFULL/ReligiousActivity/ReligiousActivity_025 .mp4',
               './TestFULL/ReligiousActivity/ReligiousActivity_010 .mp4', './TestFULL/Running/Running_059 .mp4',
               './TestFULL/Running/Running_071 .mp4', './TestFULL/Running/Running_084 .mp4',
               './TestFULL/Running/Running_052 .mp4', './TestFULL/Running/Running_068 .mp4',
               './TestFULL/Running/Running_060 .mp4', './TestFULL/Running/Running_011 .mp4',
               './TestFULL/Running/Running_025 .mp4', './TestFULL/Running/Running_051 .mp4',
               './TestFULL/Running/Running_030 .mp4', './TestFULL/Running/Running_079 .mp4',
               './TestFULL/Running/Running_037 .mp4', './TestFULL/Running/Running_038 .mp4',
               './TestFULL/Running/Running_023 .mp4', './TestFULL/Running/Running_004 .mp4',
               './TestFULL/Soccer/Soccer_022 .mp4', './TestFULL/Soccer/Soccer_038 .mp4',
               './TestFULL/Soccer/Soccer_056 .mp4', './TestFULL/Soccer/Soccer_001 .mp4',
               './TestFULL/Soccer/Soccer_043 .mp4', './TestFULL/Soccer/Soccer_004 .mp4',
               './TestFULL/Soccer/Soccer_025 .mp4', './TestFULL/Soccer/Soccer_041 .mp4',
               './TestFULL/Soccer/Soccer_011 .mp4', './TestFULL/Soccer/Soccer_031 .mp4',
               './TestFULL/Soccer/Soccer_035 .mp4', './TestFULL/Soccer/Soccer_029 .mp4',
               './TestFULL/Soccer/Soccer_014 .mp4', './TestFULL/Soccer/Soccer_036 .mp4',
               './TestFULL/Soccer/Soccer_023 .mp4', './TestFULL/Swimming/Swimming_048 .mp4',
               './TestFULL/Swimming/Swimming_009 .mp4', './TestFULL/Swimming/Swimming_031 .mp4',
               './TestFULL/Swimming/Swimming_010 .mp4', './TestFULL/Swimming/Swimming_020 .mp4',
               './TestFULL/Swimming/Swimming_024 .mp4', './TestFULL/Swimming/Swimming_014 .mp4',
               './TestFULL/Swimming/Swimming_050 .mp4', './TestFULL/Swimming/Swimming_021 .mp4',
               './TestFULL/Swimming/Swimming_012 .mp4', './TestFULL/Swimming/Swimming_042 .mp4',
               './TestFULL/Swimming/Swimming_027 .mp4', './TestFULL/Swimming/Swimming_025 .mp4',
               './TestFULL/Swimming/Swimming_011 .mp4', './TestFULL/Swimming/Swimming_007 .mp4',
               './TestFULL/TrafficCollision/TrafficCollision_022 .mp4',
               './TestFULL/TrafficCollision/TrafficCollision_030 .mp4',
               './TestFULL/TrafficCollision/TrafficCollision_012 .mp4',
               './TestFULL/TrafficCollision/TrafficCollision_043 .mp4',
               './TestFULL/TrafficCollision/TrafficCollision_031 .mp4',
               './TestFULL/TrafficCollision/TrafficCollision_040 .mp4',
               './TestFULL/TrafficCollision/TrafficCollision_046 .mp4',
               './TestFULL/TrafficCollision/TrafficCollision_023 .mp4',
               './TestFULL/TrafficCollision/TrafficCollision_020 .mp4',
               './TestFULL/TrafficCollision/TrafficCollision_016 .mp4',
               './TestFULL/TrafficCollision/TrafficCollision_037 .mp4',
               './TestFULL/TrafficCollision/TrafficCollision_001 .mp4',
               './TestFULL/TrafficCollision/TrafficCollision_005 .mp4',
               './TestFULL/TrafficCollision/TrafficCollision_033 .mp4',
               './TestFULL/TrafficCollision/TrafficCollision_024 .mp4',
               './TestFULL/TrafficCongestion/TrafficCongestion_013 .mp4',
               './TestFULL/TrafficCongestion/TrafficCongestion_033 .mp4',
               './TestFULL/TrafficCongestion/TrafficCongestion_044 .mp4',
               './TestFULL/TrafficCongestion/TrafficCongestion_028 .mp4',
               './TestFULL/TrafficCongestion/TrafficCongestion_050 .mp4',
               './TestFULL/TrafficCongestion/TrafficCongestion_014 .mp4',
               './TestFULL/TrafficCongestion/TrafficCongestion_021 .mp4',
               './TestFULL/TrafficCongestion/TrafficCongestion_048 .mp4',
               './TestFULL/TrafficCongestion/TrafficCongestion_043 .mp4',
               './TestFULL/TrafficCongestion/TrafficCongestion_049 .mp4',
               './TestFULL/TrafficCongestion/TrafficCongestion_039 .mp4',
               './TestFULL/TrafficCongestion/TrafficCongestion_004 .mp4',
               './TestFULL/TrafficCongestion/TrafficCongestion_005 .mp4',
               './TestFULL/TrafficCongestion/TrafficCongestion_010 .mp4',
               './TestFULL/TrafficCongestion/TrafficCongestion_023 .mp4']

# Definition of model architecture based on Inception
class InceptionTRN(nn.Module):
    def __init__(self, pretrained_model_path, num_classes=25):
        super(InceptionTRN, self).__init__()

        # Loading BNInception as the base model
        base_model = models.inception_v3(pretrained=True)

        # Loading of the model
        base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)

        # Loading of the pretrained state dictionary
        pretrained_state_dict = torch.load(pretrained_model_path, map_location=torch.device('cpu'))

        # Filtering out missing keys from the state dictionary
        filtered_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in base_model.state_dict()}

        # Loading of the modified state dictionary
        base_model.load_state_dict(filtered_state_dict, strict=False)

        self.model = base_model

    def forward(self, x):
        return self.model(x)


# Loading of the pretrained model and defining of the number of categories
pretrained_model_path = 'model_TRN-Something-Something-V2/TRN_new_RGB_BNInception_TRNmultiscale_segment21_best.pth.tar'
# pretrained_model_path = 'model_TRN_Moments_In_Time/TRN_new_RGB_InceptionV3_TRNmultiscale_segment21_best.pth.tar'
num_classes = 25
model = InceptionTRN(pretrained_model_path, num_classes)
# Initializing of the model
model.eval()

# Moving model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print("Using device:", device)

# Defining transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Opening of CSV file and writing tested data
with open('predictionsTRNMomentsVar1.csv', 'w', newline='') as csvfile:
    fieldnames = ['Video_File', 'Predicted_Category', 'Predictions']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Iterating through every selected video and opening the video
    for video_name in video_names:
        all_predictions = []

        # Processing the video frames
        video_capture = cv2.VideoCapture(video_name)

        # Check if there are videos in category
        if not video_capture.isOpened():
            print(f"Error: Unable to open video file {video_name}")
            continue

        # Initializing of frame count
        frame_count = 0

        # Read frames from the video
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break

            # Sampling of every 10th frame
            if frame_count % 10 == 0:
                # Converting frame to tensor
                frame_tensor = transform(frame)
                frame_tensor = frame_tensor.unsqueeze(0)
                frame_tensor = frame_tensor.to(device)

                with torch.no_grad():
                    outputs = model(frame_tensor)

                all_predictions.append(outputs.cpu().numpy())

            frame_count += 1

        # Releasing of the video capture
        video_capture.release()

        # Accessing all predictions
        if all_predictions:
            # Calculating the average
            average_predictions = np.mean(all_predictions, axis=0)
            # Printing the results
            print(average_predictions)
            predicted_class_index = np.argmax(average_predictions)
            predicted_category = category_dict.get(predicted_class_index, "Unknown")
            print("Predicted category for", video_name, ":", predicted_category)

            # Writing predictions to CSV file
            writer.writerow({
                'Video_File': video_name,
                'Predicted_Category': predicted_category,
                'Predictions': str(average_predictions)
            })
