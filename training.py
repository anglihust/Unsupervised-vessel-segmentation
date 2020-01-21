from models import *
from ultis import *
import os

source_logdir=os.path.join(os.getcwd(),'model_weight/step1')
logdir=os.path.join(os.getcwd(),'model_weight/step2')

def train_source_only(datas,input_config,train_type='source'):
    if train_type=='source':
        train_img_gen = datas.train_gen()
        eval_img_gen = datas.val_gen()
    else:
        train_img_gen = datas.target_train_gen()
        eval_img_gen = datas.target_eval_gen()

    bn_train = tf.placeholder(tf.bool, [])
    ori_img = tf.placeholder(tf.float32,shape=(None,input_config.patch_diameter,input_config.patch_diameter,1),name='ori_img')
    seg_mask = tf.placeholder(tf.float32,shape=(None,input_config.patch_diameter,input_config.patch_diameter,2),name='seg_m')
    nn = tfADDA(input_config)
    finalout = nn.s_encoder(ori_img,bn_train,type=input_config.model_type)
    outs = nn.classifier(finalout)
    mask_loss = nn.build_classify_loss(seg_mask,outs)
    final_loss = tf.reduce_mean(mask_loss)
    tr_acc = nn.eval(seg_mask,outs)

    # eval_finalout, eval_edgeout = nn.s_encoder(ori_img_e,bn_train,reuse=True,trainable=False)
    # eval_outs = nn.classifier(eval_finalout,reuse=True,trainable=False)
    # te_acc = nn.eval(seg_mask_e,eval_outs)

    var_s_en = tf.trainable_variables(scope=nn.s_e)
    var_c = tf.trainable_variables(scope=nn.c)
    encoder_saver = tf.train.Saver(max_to_keep=3,var_list=var_s_en)
    classifier_saver = tf.train.Saver(max_to_keep=3,var_list=var_c)

    fresh_dir(source_logdir)
    eval_acc = []
    best_acc = 0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    batch_num=datas.soruce_subsample//input_config.batch_size
    batch_num_e=datas.source_val_subsample//input_config.batch_size
    te_acc_s = np.zeros(batch_num_e)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(0.0001).minimize(final_loss)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(input_config.step1_epochs):
            for j in range(batch_num):
                source_batch=next(train_img_gen)
                _,loss,tr_acc_ = sess.run(fetches=[train_op,final_loss,tr_acc],feed_dict={bn_train:True,ori_img:source_batch[0],seg_mask:source_batch[1]})
                if j % input_config.batch_size == 0:
                    print("epoch:{},batch_id:{},loss:{:.4f},tr_acc:{:.4f}".format(i,j,loss,tr_acc_))

            for j in range(batch_num_e):
                eval_batch = next(eval_img_gen)
                te_acc_s[j]=tr_acc.eval({bn_train:False,ori_img:eval_batch[0],seg_mask:eval_batch[1]})

            te_acc_ = np.average(te_acc_s)
            eval_acc.append(te_acc_)
            if best_acc < te_acc_:
                best_acc = te_acc_
                encoder_saver.save(sess,os.path.join(source_logdir,"encoder/encoder.ckpt"))
                classifier_saver.save(sess,os.path.join(source_logdir,"classifier/classifier.ckpt"))
            print("#+++++++++++++++++++++++++++++++++++#")
            print("epoch:{},test_accuracy:{:.4f},best_acc:{:.4f}".format(i,te_acc_,best_acc))
            print("#+++++++++++++++++++++++++++++++++++#")

def train_adda(datas,input_config,loss_type='AD'):
    batch_num=datas.unsupervised_subsample//input_config.batch_size
    batch_num_e = datas.target_eval_subsample//input_config.batch_size
    s_train_img_gen = datas.train_gen()
    t_train_img_gen = datas.target_train_gen()
    t_eval_img_gen = datas.target_eval_gen()
    t_unsupervised = datas.target_unsurpervised_gen()

    s_ori_img = tf.placeholder(tf.float32,shape=(None,input_config.patch_diameter,input_config.patch_diameter,1),name='sori_img')
    t_ori_img = tf.placeholder(tf.float32,shape=(None,input_config.patch_diameter,input_config.patch_diameter,1),name='tori_img')
    t_seg_mask = tf.placeholder(tf.float32,shape=(None,input_config.patch_diameter,input_config.patch_diameter,2),name='tseg_m')
    bn_train = tf.placeholder(tf.bool, [])

    # create graph
    nn = tfADDA(input_config)
    # for source domain
    if input_config.multi_fusion:
        up2_s,up1_s,feat_s =nn.s_encoder(s_ori_img,bn_train,reuse=tf.AUTO_REUSE,trainable=False,type=input_config.model_type)
        logits_s = nn.classifier(feat_s,reuse=tf.AUTO_REUSE,trainable=False)
        disc_s = nn.discriminator_multi_fusion(up2_s,up1_s,feat_s,reuse=False)

        up2_t,up1_t,feat_t = nn.t_encoder(t_ori_img,bn_train,reuse=False,trainable=True,type=input_config.model_type)
        logits_t = nn.classifier(feat_t,reuse=True,trainable=False)
        disc_t = nn.discriminator_multi_fusion(up2_t,up1_t,feat_t,reuse=True)

    else:
        feat_s = nn.s_encoder(s_ori_img,bn_train,reuse=tf.AUTO_REUSE,trainable=False,type=input_config.model_type)
        logits_s = nn.classifier(feat_s,reuse=tf.AUTO_REUSE,trainable=False)
        disc_s = nn.discriminator(feat_s,bn_train,reuse=False)

        # for target domain
        feat_t = nn.t_encoder(t_ori_img,bn_train,reuse=False,trainable=True,type=input_config.model_type)
        logits_t = nn.classifier(feat_t,reuse=True,trainable=False)
        disc_t = nn.discriminator(feat_t,bn_train,reuse=True)

    if loss_type=="wgan":
    #wgan gradient penalty
        epsilon = tf.random_uniform(shape=[input_config.batch_size, 1, 1, 1], minval=0., maxval=1.)
        interpolated_image = feat_s + epsilon * (feat_t - feat_s)
        d_interpolated = nn.discriminator(interpolated_image,bn_train,reuse=True)
        g_loss,d_loss = nn.build_w_loss(disc_s,disc_t)
        grad_d_interpolated = tf.gradients(d_interpolated, [interpolated_image])[0]
        slopes = tf.sqrt(1e-8 + tf.reduce_sum(tf.square(grad_d_interpolated), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        d_loss += 10 * gradient_penalty
    else:
        g_loss,d_loss = nn.build_ad_loss(disc_s,disc_t)
    acc_tr_t = nn.eval(t_seg_mask,logits_t)
    # create optimizer for two task
    var_t_en = tf.trainable_variables(nn.t_e)
    var_d = tf.trainable_variables(nn.d)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optim_g = tf.train.AdamOptimizer(0.0001,beta1=0.5,beta2=0.999).minimize(g_loss,var_list=var_t_en)
        optim_d = tf.train.AdamOptimizer(0.0001,beta1=0.5,beta2=0.999).minimize(d_loss,var_list=var_d)

    encoder_path = tf.train.latest_checkpoint(os.path.join(source_logdir,"encoder"))
    classifier_path = tf.train.latest_checkpoint(os.path.join(source_logdir,"classifier"))
    if encoder_path is None:
        raise ValueError("Don't exits in this dir")
    if classifier_path is None:
        raise ValueError("Don't exits in this dir")

    source_var = tf.contrib.framework.list_variables(encoder_path)

    var_s_g = tf.global_variables(scope=nn.s_e)
    var_c_g = tf.global_variables(scope=nn.c)
    var_t_g = tf.trainable_variables(scope=nn.t_e)
    t_dict_var={}
    for i in source_var:
        for j in var_s_g:
            if i[0][1:] in j.name[1:]:
                t_dict_var[i[0]]=j
    encoder_saver = tf.train.Saver(var_list=t_dict_var)
    classifier_saver = tf.train.Saver(var_list=var_c_g)
    dict_var={}
    for i in source_var:
        for j in var_t_g:
            if i[0][1:] in j.name[1:]:
                dict_var[i[0]]=j
    fine_turn_saver = tf.train.Saver(var_list = dict_var)
    fresh_dir(logdir)
    best_saver = tf.train.Saver(max_to_keep=3)


    # create a list to record accuracy
    eval_acc = []
    best_acc = 0
    merge = tf.summary.merge_all()
    te_acc_t= np.zeros(batch_num_e)
    # start a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # init t_e and d
        sess.run(tf.global_variables_initializer())
        # init s_e and c
        encoder_saver.restore(sess,encoder_path)
        classifier_saver.restore(sess,classifier_path)
        fine_turn_saver.restore(sess,encoder_path)
        print("model init successfully!")
        filewriter = tf.summary.FileWriter(logdir=logdir,graph=sess.graph)
        for i in range(input_config.step2_epochs):
            for j in range(batch_num_e):
                t_eval_batch=next(t_eval_img_gen)
                te_acc_t[j] = acc_tr_t.eval({bn_train:False,t_ori_img:t_eval_batch[0],t_seg_mask:t_eval_batch[1]})
            t_acc = np.average(te_acc_t)
            if best_acc < t_acc:
                best_acc = t_acc
                best_saver.save(sess,logdir+"/adda_model.ckpt")
            print("epoch: %d, target accuracy: %.4f, best accuracy：%4f"%(i,t_acc,best_acc))
            for j in range(batch_num):
                source_batch=next(s_train_img_gen)
                traget_batch=next(t_train_img_gen)
                unsuper_batch=next(t_unsupervised)
                _,d_loss_,_ = sess.run(fetches=[optim_d,d_loss,acc_tr_t],feed_dict={s_ori_img:source_batch[0],t_ori_img:unsuper_batch,t_seg_mask:traget_batch[1],bn_train:True})
                _,g_loss_,merge_,acc_seg = sess.run(fetches=[optim_g,g_loss,merge,acc_tr_t],feed_dict={s_ori_img:source_batch[0],t_ori_img:unsuper_batch,t_seg_mask:traget_batch[1],bn_train:True})
                print("step:{},g_loss:{:.4f},d_loss:{:.4f},seg_acc:{:.4f}".format(j,g_loss_,d_loss_,acc_seg))
            filewriter.add_summary(merge_,global_step=i)


def train(config,train_1st=True):
    datas= Data_set(config)
    if train_1st:
        train_source_only(datas,config,train_type='source')
    train_adda(datas,config,loss_type='ad')
