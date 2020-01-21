import tensorflow as tf
from keras import backend as K


def swish(x):
    return K.sigmoid(x)*x

class tfADDA():
    def __init__(self,config):
        self.s_e = 's_e'
        self.t_e = 't_e'
        self.c = 'c_'
        self.d = 'd_'
        self.channel_num1 = 64
        self.channel_num2 = 64
        self.channel_num3 = 32
        self.upchannel_num1=32
        self.upchannel_num2=64
        self.multi_fusion = config.multi_fusion

    def SEblock(self,inputs,scope,ratio=8,trainable=True):
        with tf.variable_scope(scope):
            input_channels = inputs.get_shape().as_list()[-1]
            reduced_channel = max(input_channels//ratio,8)
            x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
            x = tf.layers.dense(x,reduced_channel,activation=None,kernel_initializer='he_normal',trainable=trainable)
            x = swish(x)
            x = tf.layers.dense(x,input_channels,activation='sigmoid',kernel_initializer='he_normal',trainable=trainable)
            x = tf.reshape(x,[-1,1,1,input_channels])
        return tf.math.multiply(inputs,x)

    def densedownblock(self,inputs,bn_train,scope,bottom=False,filters=64,trainable=True):
        with tf.variable_scope(scope):
            bn = tf.layers.batch_normalization(inputs,training=bn_train)
            act = tf.keras.activations.relu(bn)
            conv1 = tf.layers.conv2d(act,filters,kernel_size=3,strides=1,activation=None,padding='same',kernel_initializer='he_normal',trainable=trainable)
            inputshape = inputs.get_shape()
            if inputshape[3].value != filters:
                shortcut = tf.layers.conv2d(inputs,filters,kernel_size=1,strides=1,activation=None,padding='same',kernel_initializer='he_normal',trainable=trainable)
            else:
                shortcut = inputs
            result1 = tf.math.add(conv1,shortcut)
            bn = tf.layers.batch_normalization(result1,training=bn_train)
            act = tf.keras.activations.relu(bn)
            conv2 = tf.layers.conv2d(act,filters,kernel_size=3,strides=1,activation=None,padding='same',kernel_initializer='he_normal',trainable=trainable)
            out = tf.math.add(tf.math.add(result1,conv2),shortcut)
            out = tf.keras.activations.relu(out)
            if bottom:
                return out
            else:
                return (out,tf.layers.max_pooling2d(out,pool_size=2,strides=2,padding='same'))

    def denseupblock(self,inputs,cocate_inputs,bn_train,scope,filters=64,trainable=True):
        with tf.variable_scope(scope):
            x =tf.layers.conv2d_transpose(inputs,filters,kernel_size=3,strides=2,activation=None,padding='same',kernel_initializer='he_normal',trainable=trainable)
            x = tf.concat([x,cocate_inputs],axis=-1)
            inputshape = x.get_shape()
            bn = tf.layers.batch_normalization(x,training=bn_train)
            act = tf.keras.activations.relu(bn)
            conv1 = tf.layers.conv2d(act,filters,kernel_size=3,strides=1,activation=None,padding='same',kernel_initializer='he_normal',trainable=trainable)
            if inputshape[3].value != filters:
                shortcut = tf.layers.conv2d(x,filters,kernel_size=1,strides=1,activation=None,padding='same',kernel_initializer='he_normal',trainable=trainable)
            else:
                shortcut = x
            result1 = tf.math.add(conv1, shortcut)
            bn = tf.layers.batch_normalization(result1,training=bn_train)
            act = tf.keras.activations.relu(bn)
            conv2 = tf.layers.conv2d(act,filters,kernel_size=3,strides=1,activation=None,padding='same',kernel_initializer='he_normal',trainable=trainable)
            result = tf.math.add(tf.math.add(result1,conv2),shortcut)
            result = tf.keras.activations.relu(result)
            return result

    def upsamplingblock(self,inputs,cocate_inputs,bn_train,scope,filters=64,SE=False,trainable=True):
        with tf.variable_scope(scope):
            x =tf.layers.conv2d_transpose(inputs,filters,kernel_size=3,strides=2,activation=None,padding='same',kernel_initializer='he_normal',trainable=trainable)
            x = tf.concat([x,cocate_inputs],axis=-1)
            x = tf.layers.conv2d(x,filters,kernel_size=3,strides=1,activation=None,padding='same',kernel_initializer='he_normal',trainable=trainable)
            x = tf.layers.batch_normalization(x,training=bn_train)
            x =swish(x)
            if SE:
                x = self.SEblock(x,scope=scope+'_se',ratio= 8,trainable=trainable)
            x = tf.layers.conv2d(x,filters,kernel_size=3,strides=1,activation=None,padding='same',kernel_initializer='he_normal',trainable=trainable)
            x = tf.layers.batch_normalization(x,training=bn_train)
            x =swish(x)
        return x

    def downsammplingblock(self,inputs,bn_train,scope,bottom=False,filters=64,SE=True,trainable=True):
        with tf.variable_scope(scope):
            x = tf.layers.conv2d(inputs,filters,kernel_size=3,strides=1,padding='same',kernel_initializer='he_normal',trainable=trainable)
            x = tf.layers.batch_normalization(x,training=bn_train)
            x =swish(x)
            if SE:
                x = self.SEblock(x,scope=scope+'_se',ratio=8,trainable=True)
            x = tf.layers.conv2d(x,filters,kernel_size=3,strides=1,padding='same',kernel_initializer='he_normal',trainable=trainable)
            x = tf.layers.batch_normalization(x,training=bn_train)
            x =swish(x)
            if bottom:
                return x
            else:
                return (x,tf.layers.max_pooling2d(x,pool_size=2,strides=2,padding='same'))

    def s_encoder(self,inputs,is_train,reuse=False,trainable=True,type='dense'):
        with tf.variable_scope(self.s_e,reuse=reuse):
            if type=='dense':
                conc_out1, output1= self.densedownblock(inputs,is_train,scope='downsample_layer1',filters=self.upchannel_num1,trainable=trainable)#48/24
                conc_out2, output2= self.densedownblock(output1,is_train,scope='downsample_layer2',filters=self.upchannel_num2,trainable=trainable)#24/12
                conc_out3, output3= self.densedownblock(output2,is_train,scope='downsample_layer3',filters=64,trainable=trainable)#12/6
                output4= self.densedownblock(output3,is_train,scope='downsample_layer4',bottom=True,filters=128,trainable=trainable)#6
                uoutput5 = self.denseupblock(output4,conc_out3,is_train,scope='upsample_layer1',filters=self.channel_num1,trainable=trainable)#12
                uoutput6 = self.denseupblock(uoutput5,conc_out2,is_train,scope='upsample_layer2',filters=self.channel_num2,trainable=trainable)#24
                uoutput7 = self.denseupblock(uoutput6,conc_out1,is_train,scope='upsample_layer3',filters=self.channel_num3,trainable=trainable)#48
            else:
                conc_out1, output1= self.downsammplingblock(inputs,is_train,scope='downsample_layer1',filters=self.upchannel_num1,SE=True,trainable=trainable)#48/24
                conc_out2, output2= self.downsammplingblock(output1,is_train,scope='downsample_layer2',filters=self.upchannel_num2,SE=True,trainable=trainable)#24/12
                conc_out3, output3= self.downsammplingblock(output2,is_train,scope='downsample_layer3',filters=64,SE=True,trainable=trainable)#12/6
                output4= self.downsammplingblock(output3,is_train,scope='downsample_layer4',bottom=True,filters=128,SE=True,trainable=trainable)#6
                uoutput5 = self.upsamplingblock(output4,conc_out3,is_train,scope='upsample_layer1',filters=self.channel_num1,SE=True,trainable=trainable)#12
                uoutput6 = self.upsamplingblock(uoutput5,conc_out2,is_train,scope='upsample_layer2',filters=self.channel_num2,SE=True,trainable=trainable)#24
                uoutput7 = self.upsamplingblock(uoutput6,conc_out1,is_train,scope='upsample_layer3',filters=self.channel_num3,SE=True,trainable=trainable)#48
            out = tf.layers.conv2d(uoutput7,32,kernel_size=3,strides=1,activation='relu',padding='same',kernel_initializer='he_normal',trainable=trainable,name='conv_out')
        if self.multi_fusion:
            return uoutput6, uoutput7, out
        else:
            return out

    def t_encoder(self,inputs,is_train,reuse=False,trainable=True,type='dense'):
        with tf.variable_scope(self.t_e,reuse=reuse):
            if type=='dense':
                conc_out1, output1= self.densedownblock(inputs,is_train,scope='downsample_layer1',filters=self.upchannel_num1,trainable=trainable)#48/24
                conc_out2, output2= self.densedownblock(output1,is_train,scope='downsample_layer2',filters=self.upchannel_num2,trainable=trainable)#24/12
                conc_out3, output3= self.densedownblock(output2,is_train,scope='downsample_layer3',filters=64,trainable=trainable)#12/6
                output4= self.densedownblock(output3,is_train,scope='downsample_layer4',bottom=True,filters=128,trainable=trainable)#6
                uoutput5 = self.denseupblock(output4,conc_out3,is_train,scope='upsample_layer1',filters=self.channel_num1,trainable=trainable)#12
                uoutput6 = self.denseupblock(uoutput5,conc_out2,is_train,scope='upsample_layer2',filters=self.channel_num2,trainable=trainable)#24
                uoutput7 = self.denseupblock(uoutput6,conc_out1,is_train,scope='upsample_layer3',filters=self.channel_num3,trainable=trainable)#48
            else:
                conc_out1, output1= self.downsammplingblock(inputs,is_train,scope='downsample_layer1',filters=self.upchannel_num1,SE=True,trainable=trainable)#48/24
                conc_out2, output2= self.downsammplingblock(output1,is_train,scope='downsample_layer2',filters=self.upchannel_num2,SE=True,trainable=trainable)#24/12
                conc_out3, output3= self.downsammplingblock(output2,is_train,scope='downsample_layer3',filters=64,SE=True,trainable=trainable)#12/6
                output4= self.downsammplingblock(output3,is_train,scope='downsample_layer4',bottom=True,filters=128,SE=True,trainable=trainable)#6
                uoutput5 = self.upsamplingblock(output4,conc_out3,is_train,scope='upsample_layer1',filters=self.channel_num1,SE=True,trainable=trainable)#12
                uoutput6 = self.upsamplingblock(uoutput5,conc_out2,is_train,scope='upsample_layer2',filters=self.channel_num2,SE=True,trainable=trainable)#24
                uoutput7 = self.upsamplingblock(uoutput6,conc_out1,is_train,scope='upsample_layer3',filters=self.channel_num3,SE=True,trainable=trainable)#48
            out = tf.layers.conv2d(uoutput7,32,kernel_size=3,strides=1,activation='relu',padding='same',kernel_initializer='he_normal',trainable=trainable,name='conv_out')
        if self.multi_fusion:
            return uoutput6, uoutput7, out
        else:
            return out


    def classifier(self,inputs,reuse=False,trainable=True):
        with tf.variable_scope(self.c,reuse=reuse):
            finalout = tf.layers.conv2d(inputs,16,kernel_size=1,strides=1,activation='relu',padding='same',kernel_initializer='he_normal',trainable=trainable,name='cl_conv1')
            finalout = tf.layers.conv2d(finalout,2,kernel_size=1,strides=1,activation='softmax',padding='same',kernel_initializer='he_normal',trainable=trainable,name='cl_conv2')
        return finalout

    def discriminator(self,inputs,bn_train,reuse=False,trainable=True):
        with tf.variable_scope(self.d,reuse=reuse):
            conv1 = tf.layers.conv2d(inputs,32,kernel_size=3,strides=2,activation=None,padding='same',kernel_initializer='he_normal',trainable=trainable,name='d_conv1')
            conv1 = tf.nn.leaky_relu(conv1,alpha=0.2)
            conv1 = tf.nn.dropout(conv1,0.5)
            conv2 = tf.layers.conv2d(conv1,32,kernel_size=3,strides=2,activation=None,padding='same',kernel_initializer='he_normal',trainable=trainable,name='d_conv2')
            #conv2 = tf.layers.batch_normalization(conv2,training=bn_train)

            conv2 = tf.nn.leaky_relu(conv2,alpha=0.2)
            conv2 = tf.nn.dropout(conv2,0.5)
            conv3 = tf.layers.conv2d(conv2,64, kernel_size=3,strides=2,activation=None,padding='same',kernel_initializer='glorot_uniform',trainable=trainable,name='d_conv3')
            #conv3 = tf.layers.batch_normalization(conv3,training=bn_train)

            conv3 = tf.nn.leaky_relu(conv3,alpha=0.2)
            conv3 = tf.nn.dropout(conv3,0.5)

            conv4 =tf.layers.conv2d(conv3,128, kernel_size=3,strides=2,activation=None,padding='same',kernel_initializer='glorot_uniform',trainable=trainable,name='d_conv4')
            #conv4 = tf.layers.batch_normalization(conv4,training=bn_train)

            conv4 = tf.nn.leaky_relu(conv4,alpha=0.2)
            conv4 = tf.nn.dropout(conv4,0.5)

            conv5 =tf.layers.conv2d(conv4,128, kernel_size=3,strides=2,activation=None,padding='same',kernel_initializer='glorot_uniform',trainable=trainable,name='d_conv5')
            #conv4 = tf.layers.batch_normalization(conv4,training=bn_train)

            conv5 = tf.nn.leaky_relu(conv5,alpha=0.2)
            conv5 = tf.nn.dropout(conv5,0.5)

            fc1= tf.keras.layers.GlobalAveragePooling2D()(conv5)
            #fc1 = tf.layers.flatten(conv5)
            output = tf.layers.dense(fc1,1,activation=None,kernel_initializer='he_normal',trainable=trainable,name='out_dense')
            return output

    def discriminator_multi_fusion(self,inputs1,inputs2,inputs3,reuse=False,trainable=True):
        with tf.variable_scope(self.d,reuse=reuse):
            conv1 = tf.layers.conv2d(inputs1,32,kernel_size=3,strides=1,activation=None,padding='same',kernel_initializer='he_normal',trainable=trainable,name='d_conv1')
            conv1 = tf.nn.leaky_relu(conv1,alpha=0.2)
            up1 = tf.layers.conv2d_transpose(conv1,self.channel_num3,kernel_size=3,strides=2,activation=None,padding='same',kernel_initializer='he_normal',trainable=trainable,name='de_conv1')
            conv1=tf.math.add(up1,inputs2)

            conv2 = tf.layers.conv2d(conv1,32,kernel_size=3,strides=1,activation=None,padding='same',kernel_initializer='he_normal',trainable=trainable,name='d_conv2')
            conv2 = tf.nn.leaky_relu(conv2,alpha=0.2)
            #up2 = tf.layers.conv2d_transpose(conv2,32,kernel_size=3,strides=2,activation=None,padding='same',kernel_initializer='he_normal',trainable=trainable,name='de_conv2')
            conv2=tf.math.add(conv2,inputs3)

            conv3 =tf.layers.conv2d(conv2,64, kernel_size=3,strides=2,activation=None,padding='same',kernel_initializer='glorot_uniform',trainable=trainable,name='d_conv3')
            conv3 = tf.nn.leaky_relu(conv3,alpha=0.2)
            conv3 = tf.nn.dropout(conv3,0.5)

            conv4 =tf.layers.conv2d(conv3,64, kernel_size=3,strides=2,activation=None,padding='same',kernel_initializer='glorot_uniform',trainable=trainable,name='d_conv4')
            conv4 = tf.nn.leaky_relu(conv4,alpha=0.2)
            conv4 = tf.nn.dropout(conv4,0.5)

            conv5 =tf.layers.conv2d(conv4,64, kernel_size=3,strides=2,activation=None,padding='same',kernel_initializer='glorot_uniform',trainable=trainable,name='d_conv5')
            conv5 = tf.nn.leaky_relu(conv5,alpha=0.2)
            conv5 = tf.nn.dropout(conv5,0.5)

            conv6 =tf.layers.conv2d(conv5,64, kernel_size=3,strides=2,activation=None,padding='same',kernel_initializer='glorot_uniform',trainable=trainable,name='d_conv6')
            conv6 = tf.nn.leaky_relu(conv6,alpha=0.2)
            conv6 = tf.nn.dropout(conv6,0.5)

            conv7 =tf.layers.conv2d(conv6,64, kernel_size=3,strides=2,activation=None,padding='same',kernel_initializer='glorot_uniform',trainable=trainable,name='d_conv7')
            conv7 = tf.nn.leaky_relu(conv7,alpha=0.2)
            conv7 = tf.nn.dropout(conv7,0.5)

            fc1= tf.keras.layers.GlobalAveragePooling2D()(conv7)
            output = tf.layers.dense(fc1,1,activation=None,kernel_initializer='he_normal',trainable=trainable,name='out_dense')
            return output

    def build_classify_loss(self,y_true, y_pred,smooth = 1):
        return 1-self.dice_coef(y_true, y_pred,smooth = 1)

    def dice_coef(self,y_true, y_pred,smooth = 1):
        y_true_f = tf.reshape(y_true,[-1])
        y_pred_f = tf.reshape(y_pred,[-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

    # wgan loss function
    def build_w_loss(self,disc_s,disc_t):
        d_loss = -tf.reduce_mean(disc_s) + tf.reduce_mean(disc_t)
        g_loss = -tf.reduce_mean(disc_t)
        tf.summary.scalar("g_loss",g_loss)
        tf.summary.scalar('d_loss',d_loss)
        return g_loss,d_loss

    def build_ad_loss(self,disc_s,disc_t):
        g_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_t,labels=tf.ones_like(disc_t))
        g_loss = tf.reduce_mean(g_loss)
        d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_s,labels=tf.ones_like(disc_s)))+tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_t,labels=tf.zeros_like(disc_t)))
        tf.summary.scalar("g_loss",g_loss)
        tf.summary.scalar('d_loss',d_loss)
        return g_loss,d_loss

    def build_ad_loss_v2(self,disc_s,disc_t):
        d_loss = -tf.reduce_mean(tf.log(disc_s+ 1e-12)+tf.log(1-disc_t+1e-12))
        g_loss = -tf.reduce_mean(tf.log(disc_t + 1e-12))
        return g_loss,d_loss

    def eval(self,y_true, y_pred):
        return self.dice_coef(y_true, y_pred)
