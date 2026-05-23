# behavior_data.csv columns:
#pitch, yaw, roll, mar, face_present, head_flag, absence_flag, mouth_flag, label

# label: 0 = normal, 1 = suspicious

# Collect by:
# 1. Running the full system on participants
# 2. Participant sits normally for 5 min (label=0)
# 3. Participant does cued cheating behaviors for 5 min (label=1)
# 4. Save each frame's feature vector with label to CSV