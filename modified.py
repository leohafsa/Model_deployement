import cv2
import tensorflow as tf
from scipy.interpolate import interp1d
import dlib
import numpy as np
import os
from os.path import join
# import pandas as pd
from scipy import signal
from scipy.signal import butter, lfilter
import time
import torch
from scipy.signal import butter, filtfilt
import pickle
face_detector = dlib.get_frontal_face_detector()
low_freq = 0.7
high_freq = 4.0
def estimate_spo2_green(ppg_signal):
    # fs = fps  # Sampling frequency
    filtered_ppg=ppg_signal

    # Calculate DC and AC components of the PPG signal
    dc_component = np.mean(filtered_ppg)
    # ac_component = np.mean((filtered_ppg - dc_component))
    ac_component = np.sqrt(np.mean((filtered_ppg - dc_component) ** 2))

    # Estimate oxygenated and deoxygenated blood levels
    oxygenated_blood = ac_component
    deoxygenated_blood = dc_component

    spo2 = 125 - 26*(oxygenated_blood / ( oxygenated_blood+deoxygenated_blood))
    lower_bound = 80
    upper_bound = 100
    spo2 = np.clip(spo2, lower_bound, upper_bound)


    return spo2

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def estimate_hr_spo2_given_video_link(video_path,sys_model,dys_model,sugar_model,temperature_model,heart_model,spo_model):
  cap = cv2.VideoCapture(video_path)
  bp_sys_model = tf.keras.models.load_model(sys_model)
  bp_dys_model = tf.keras.models.load_model(dys_model)
  sugar_80_180_model=tf.keras.models.load_model(sugar_model)
  temp_model=tf.keras.models.load_model(temperature_model)
  hr_model=tf.keras.models.load_model(heart_model)
  spo2_model=tf.keras.models.load_model(spo_model)
  start_time=time.time()
  # Get total frames and fps
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  fps = cap.get(cv2.CAP_PROP_FPS)
  fs = fps  # Assuming 30 frames per second

  # print(f'Total frames are {total_frames} and FPS is {fps}')
  frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  green_signal=[]
  GT_Spo2=[]
  GT_HR=[]
  spo2_ssa=[]
  spo2_raw=[]
  spo2_estimate=[]
  hr_estimate=[]
  spo2_temp=[]
  ac_green=[]
  SPO2_pred_gt_video=[]
  HR_pred_gt_video=[]
  ###################################################
  start_time=time.time()

  current_frame = 1

  # Iterate through the frames
  while True:
      # Read the frame
      if current_frame <= np.floor(fps*2):
        ret, frame = cap.read()
        current_frame+=1
        continue

      # print('current_frame is ',current_frame)
      ret, frame = cap.read()

      # Check if frame is successfully read
      if not ret:
          break


      frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      faces = face_detector(gray_frame)

      current_frame += 1

      # Iterate through detected faces
      for face in faces:
          # Extract face coordinates
          # x, y, w, h = face['box']
          (x, y, w, h) = (faces[0].left(), faces[0].top(), faces[0].width(), faces[0].height())

      
          face_region = frame_rgb[y:y+h, x:x+w]
          # resized_face = cv2.resize(face_region, (100, 100))
          # green_signal.append(np.mean(face_region[:,:,1], axis=(0, 1)))
          ac_green.append(np.mean(face_region[:,:,1], axis=(0, 1)))
  ac_green=np.array(ac_green)
  ac_green=(ac_green-np.mean(ac_green))
  ac_green=butter_bandpass_filter(np.array(ac_green), 0.7, 4.0, fps, order=4)
  if len(ac_green)<300:
    print(f'original video length is {len(ac_green)} so interpolation')
    interp_func = interp1d(np.arange(len(ac_green)), ac_green, kind='linear')
    ac_green = interp_func(np.linspace(0, len(ac_green) - 1, 300))
  raw_ppg=ac_green[-300:]  
  raw_ppg_1=np.copy(raw_ppg)
  tf_ppg = tf.convert_to_tensor(raw_ppg_1, dtype=tf.float32)
  min_value = tf.reduce_min(tf_ppg)
  max_value = tf.reduce_max(tf_ppg)
  scaled_ppg = (tf_ppg - min_value) / (max_value - min_value)

  lower_bound_temp = 95
  upper_bound_temp = 102
  lower_bound_spo2=80
  upper_bound_spo2=100
  predicted_sys = bp_sys_model.predict(tf.expand_dims(scaled_ppg, axis=0),verbose=0)
  predicted_dys = bp_dys_model.predict(tf.expand_dims(scaled_ppg, axis=0),verbose=0)
  predicted_sugar=sugar_80_180_model.predict(tf.expand_dims(scaled_ppg, axis=0),verbose=0)
  predicted_temp=temp_model.predict(tf.expand_dims(scaled_ppg, axis=0),verbose=0)
  predicted_hr=hr_model.predict(tf.expand_dims(scaled_ppg, axis=0),verbose=0)
  predicted_spo2=spo2_model.predict(tf.expand_dims(scaled_ppg, axis=0),verbose=0)
  predicted_temp = np.clip(predicted_temp, lower_bound_temp, upper_bound_temp)
  predicted_spo2 = np.clip(predicted_spo2, lower_bound_spo2, upper_bound_spo2)
  print(f'Blood Pressure: Systolic [pred: {predicted_sys} ]  ||| Diastolic [Pred: {predicted_dys} ]')
  print(f'Sugar [pred: {predicted_sugar}, ]')
  print(f'Temperature [pred: {predicted_temp}, ]')
  print(f' HR model: {predicted_hr}')
  print(f'SpO2 model : {predicted_spo2}')
  print('------------------------------------------------------------------------')
  print('------------------------------------------------------------------------')
  
  heart_rate=round(float(predicted_hr[0,0]))
  spo2_rate=round(float(predicted_spo2[0,0]))
  systolic=round(float(predicted_sys[0,0]))
  diastolic=round(float(predicted_dys[0,0]))
  sugar=round(float(predicted_sugar[0,0]))
  temp=round(float(predicted_temp[0,0]))
  print(f'Blood Pressure: Systolic [pred: {systolic} ]  ||| Diastolic [Pred: {diastolic} ]')
  print(f'Sugar [pred: {sugar}, ]')
  print(f'Temperature [pred: {temp}, ]')
  print(f' HR model: {heart_rate}')
  print(f'SpO2 model : {spo2_rate}')
  
  return heart_rate,spo2_rate,systolic,diastolic,sugar,temp


# import os
# from os.path import join
# vid_dir_path='/content/drive/MyDrive/vital/mobile_video_test'
# vid_dir=os.listdir(vid_dir_path)
# bp_sys_model = tf.keras.models.load_model('/content/drive/MyDrive/vital/comb_models_v4/sys_v1_mse30_balance_2.h5')
# bp_dys_model = tf.keras.models.load_model('/content/drive/MyDrive/vital/comb_models_v4/dys_v1_mse30_balance_2.h5')
# sugar_80_180_model=tf.keras.models.load_model('/content/drive/MyDrive/vital/comb_models_v4/sugar_v1_mse30_balance_2.h5')
# temp_model=tf.keras.models.load_model('/content/drive/MyDrive/vital/comb_models_v4/temp_v1_mse30_balance_2.h5')
# hr_model=tf.keras.models.load_model('/content/drive/MyDrive/Colab Notebooks/ahsan/Seq_models/V4/HR_v1_mse30_balance_2.h5')
# spo2_model=tf.keras.models.load_model('/content/drive/MyDrive/Colab Notebooks/ahsan/Seq_models/V4/spo2_v1_mse30_balance_2.h5')
# for vid_path in vid_dir:
#   print('Video name: ',vid_path)
# vid_name=vid_path
# pred_hr_model,pred_spo2_model,pred_sys,pred_dys,pred_sugar,pred_temp,raw_ppg=estimate_hr_spo2_given_video_link(join(vid_dir_path,vid_path),bp_sys_model,bp_dys_model,sugar_80_180_model,temp_model,hr_model,spo2_model)