from models import *
from ultis import *
logdir=os.path.join(os.getcwd(),'model_weight/step2')

def input_img(img_path,stride=96,patch=96,normalized=True,iscolor=False,inverse=False):
    oripath=[]
    for ext in('/*.tif','/*.png'):
        oripath.extend(natsorted(glob(img_path+ext)))

    for i in tqdm(range(len(oripath))):
        if iscolor:
            ori_img =cv2.imread(oripath[i])
            ori_img=ori_img[:,:,1]*0.75+ori_img[:,:,0]*0.25
        else:
            ori_img =cv2.imread(oripath[i],0)
        img_patches = window_search_2D(ori_img,patch,stride)
        if i ==0:
            img_patches_array=img_patches
        else:
            img_patches_array= np.append(img_patches_array,img_patches,axis=0)

    img_patches_array=np.expand_dims(img_patches_array,axis=3)

    if inverse:
        img_patches_array=255-img_patches_array

    if normalized:
        img_patches_array = img_patches_array/255*2-1

    return img_patches_array

def predict_img(img_path,config):
    img_patches_array= input_img(img_path,stride=config.stride,patch=config.patch_diameter,normalized=True,iscolor=False,inverse=False)
    ori_img = tf.placeholder(tf.float32,shape=(None,config.patch_diameter,config.patch_diameter,1),name='ori_img')
    bn_train = tf.placeholder(tf.bool, [])
    nn = tfADDA(config)
    if config.multi_fusion:
        up2_t,up1_t,feat_t = nn.t_encoder(ori_img,bn_train,reuse=tf.AUTO_REUSE,trainable=False)
        logits_t = nn.classifier(feat_t,reuse=tf.AUTO_REUSE,trainable=False)
    else:
        feat_t = nn.t_encoder(ori_img,bn_train,reuse=tf.AUTO_REUSE,trainable=False)
        logits_t = nn.classifier(feat_t,reuse=tf.AUTO_REUSE,trainable=False)

    encoder_path = tf.train.latest_checkpoint(os.path.join(logdir,"encoder"))
    classifier_path = tf.train.latest_checkpoint(os.path.join(logdir,"classifier"))
    if encoder_path is None:
        raise ValueError("Don't exits in this dir")
    if classifier_path is None:
        raise ValueError("Don't exits in this dir")
    var_t_g = tf.trainable_variables(scope=nn.t_e)
    var_c_g = tf.global_variables(scope=nn.c)
    encoder_saver = tf.train.Saver(var_list=var_t_g)
    classifier_saver = tf.train.Saver(var_list=var_c_g)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        encoder_saver.restore(sess,encoder_path)
        classifier_saver.restore(sess,classifier_path)
        print("model init successfully!")
        logits_t_ = sess.run(fetches=[logits_t],feed_dict={ori_img:img_patches_array,bn_train:False})

    seg_array = (logits_t_+1)/2*255
    seg_image = recompone_overlap(seg_array,config.patch_diameter,config.stride,config.src_height,config.src_width)
    plt.imshow(seg_image)
