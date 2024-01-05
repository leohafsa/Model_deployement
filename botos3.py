import boto3
import os
s3_client = boto3.client(service_name='s3', 
    region_name='us-east-2',
    aws_access_key_id='AKIAVAHUPHL6XWW4X7PU',
    aws_secret_access_key='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')

def download_video(video_name):
    with open(os.path.join('/tmp/',video_name), 'wb') as f:
        s3_client.download_fileobj('prof.ezshifa.com', video_name, f)
    print('video downloaded')


def download_model_sys(vital_1):
    # bp_sys=vital_1
    bp_sys='{}.h5'.format(vital_1)
    model_sys=os.path.join('/tmp',bp_sys)
    print(model_sys)
    if not os.path.exists(model_sys):
        print('path not exist')
        with open(model_sys, 'wb') as f:
            s3_client.download_fileobj('ezshifa-vitals', os.path.join('models',bp_sys), f)
            # s3_client.download_fileobj('ezshifa-vitals', os.path.join('models',bp_sys), f)
            print("model one downloaded")
    return model_sys
def download_model_dys(vital_2):
    bp_dys='{}.h5'.format(vital_2)
    model_dys=os.path.join('/tmp',bp_dys)
    # out=vital
    if not os.path.exists(model_dys):
        print('model 2 file dont exist')
        with open(model_dys, 'wb') as f:
            s3_client.download_fileobj('ezshifa-vitals', os.path.join('models',bp_dys), f)
            print("model two downloaded ")
    return model_dys
def download_model_sugar(vital_3):
    smodel='{}.h5'.format(vital_3)
    model_sugar=os.path.join('/tmp',smodel)
    # out=vital
    if not os.path.exists(model_sugar):
        print('model 2 file dont exist')
        with open(model_sugar, 'wb') as f:
            s3_client.download_fileobj('ezshifa-vitals', os.path.join('models',smodel), f)
            print("model two downloaded ")
    return model_sugar
def download_model_temp(vital_4):
    tmodel='{}.h5'.format(vital_4)
    model_temp=os.path.join('/tmp',tmodel)
    # out=vital
    if not os.path.exists(model_temp):
        print('model 2 file dont exist')
        with open(model_temp, 'wb') as f:
            s3_client.download_fileobj('ezshifa-vitals', os.path.join('models',tmodel), f)
            print("model three downloaded ")
    return model_temp
def download_model_hr(vital_5):
    hmodel='{}.h5'.format(vital_5)
    model_hr=os.path.join('/tmp',hmodel)
    # out=vital
    if not os.path.exists(model_hr):
        print('model 2 file dont exist')
        with open(model_hr, 'wb') as f:
            s3_client.download_fileobj('ezshifa-vitals', os.path.join('models',hmodel), f)
            print("model Four downloaded ")
    return model_hr
def download_model_spo2(vital_6):
    spmodel='{}.h5'.format(vital_6)
    model_spo=os.path.join('/tmp',spmodel)
    # out=vital
    if not os.path.exists(model_spo):
        print('model 2 file dont exist')
        with open(model_spo, 'wb') as f:
            s3_client.download_fileobj('ezshifa-vitals', os.path.join('models',spmodel), f)
            print("model two downloaded ")
    return model_spo