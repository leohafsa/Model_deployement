from fastapi import FastAPI,Query
from mangum import Mangum
from fastapi.responses import JSONResponse
import uvicorn
import os
import time
app = FastAPI()
handler = Mangum(app)

from face_lib_complete import estimate_hr_spo2_given_video_link
from botos3 import download_video,download_model_sys,download_model_dys,download_model_sugar,download_model_temp,download_model_hr,download_model_spo2
from typing import Union

# from pred_vitals_v2 import estimate_hr_spo2_given_video_link
# # from transform import save_transform
# from botos3 import download_video,download_model_sys,download_model_dys,download_model_sugar,download_model_temp
# from typing import Union

@app.get("/")
def read_item(video_name: str = Query(default=None, description="Video ID from S3 bucket")):
    start_time=time.time()
    print("Get into function")
    download_video(video_name)
    out_path=os.path.join('/tmp',video_name)  
    print("issue in video downloadid function")
    bp_systolic='sys_v1_mse30_balance_2'
    bp_diastolic='dys_v1_mse30_balance_2'
    vital_sugar='sugar_v1_mse30_balance_2'
    vital_temp='temp_v1_mse30_balance_2'
    vital_hr='HR_v1_mse30_balance_2'
    vital_spo2='spo2_v1_mse30_balance_2'
    bp_sys_model=download_model_sys(bp_systolic)
    bp_dys_model=download_model_dys(bp_diastolic)
    sugar_cal_model=download_model_sugar(vital_sugar)
    temp_cal_model=download_model_temp(vital_temp)
    hr_cal_model=download_model_hr(vital_hr)
    spo_cal_model=download_model_spo2(vital_spo2)
    print("Model fetched")
    results=estimate_hr_spo2_given_video_link(out_path,bp_sys_model,bp_dys_model,sugar_cal_model,temp_cal_model,hr_cal_model,spo_cal_model)
    print("video cropped")
    end_time=time.time()
    time_taken=end_time-start_time
    print("video transformed")
    print(results)
    return JSONResponse({"result": results,'timetaken':int(time_taken)})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
   