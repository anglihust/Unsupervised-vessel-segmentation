from training import *
from predict import *

img_path=os.path.join(os.getcwd(),'infer_img')
def main():
    config =get_config_from_json(os.path.join(os.getcwd(),'config.json'))
    train(config,True)
    img = predict_img(img_path,config)

if __name__=="main":
    main()
