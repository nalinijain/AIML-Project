import os, sys
import numpy as np
import csv, json

def imu_anno_loader(file_name):
    event, start_time, end_time, value = [[], [], [], []]       # Initialization
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                pass
            else:
                event.append(row[2])
                start_time.append(row[4])
                end_time.append(row[5])
                value.append(row[6])
            line_count += 1
    return event, start_time, end_time, value


def imu_loader(file_name):
    t, x, y, z = [[], [], [], []]                               # Initializationi
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                pass
            else:
                t.append(float(row[0])/1000)
                x.append(float(row[1]))
                y.append(float(row[2]))
                z.append(float(row[3]))
            line_count += 1
    data = np.vstack([np.array(t), np.array(x), np.array(y), np.array(z)])
    return data


def open_json_file(file_name):
    id = []
    joints = []
    with open(file_name) as json_file:
        data = json.load(json_file)
        for body in data["bodies"]:
            id.append(body['id'])
            joints.append(np.array(body['joints26']).reshape(1,-1,4))
    try:
        joints = np.vstack(joints)
    except:
        pass

    return id, joints


def refined_kp_loader(path):
    _, _, f = next(os.walk(path))
    number_of_files = len(f)
    
    f.sort()
    joints = [] # Initializing variables
    print("Start loading key-points data.")

    for i in range(number_of_files):
        file_name = f[i]
    
        ids, cur_joints = open_json_file(os.path.join(path, file_name))
        
        for id in ids:
            if len(joints) == 0:     # Before both subjects are detected at least one time
                joints.append(cur_joints[0].reshape(-1, 26, 4))
            elif len(joints) == 1:
                joints.append(cur_joints[1].reshape(-1, 26, 4))
            else: 
                joints[ids.index(id)] = np.vstack([
                        joints[ids.index(id)], cur_joints[ids.index(id)].reshape(-1, 26, 4)])

    print("Complete key-points data loading.")
    return joints, [0, 1]