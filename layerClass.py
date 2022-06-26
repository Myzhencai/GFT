import tensorflow as tf

# class Layer:
#
#     def __init__(self,initializer):
#         self.weightsDict = {}
#         self.biasDict = {}
#         if (initializer == "Xavier"):
#             self.tf_initializer = tf.contrib.layers.xavier_initializer()
#         elif (initializer == "Normal"):
#             self.tf_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
#         elif (initializer == "He"):
#             self.tf_initializer = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
#
#
#     def conv2d(self,inputFeature,filterSize, inputSize, outputSize, name,strides = 1):
#         filter_shape = [filterSize, filterSize, inputSize, outputSize]
#         self.weightName = name + "weight"
#         self.biasName = name + "bias"
#         with tf.compat.v1.variable_scope("variable", reuse=tf.compat.v1.AUTO_REUSE):
#             self.weightsDict[self.weightName] = tf.compat.v1.get_variable(self.weightName, shape=filter_shape,initializer=self.tf_initializer)
#             self.biasDict[self.biasName] = tf.compat.v1.get_variable(self.biasName, shape = outputSize, initializer=self.tf_initializer)
#         convOutput = tf.nn.conv2d(input = inputFeature, filter = self.weightsDict[self.weightName], strides=[1, strides, strides, 1], padding='SAME', name = name)
#         finalOutput = tf.nn.bias_add(convOutput, self.biasDict[self.biasName])
#         # print("finalOutput")
#         # print(finalOutput.shape)
#         return finalOutput
#
#     def avgpool2d(self,inputData):
#         return tf.nn.avg_pool(value = inputData, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')
#
#     def downSamplingBlock(self,x,input_channels,output_channels,down_size,name):
#         print("Downsampling")
#         print (x.shape)
#         conv1 =  self.conv2d(x,3,input_channels,output_channels,name+"conv1",strides = down_size)
#         print (conv1.shape)
#         batchNorm1 = tf.layers.batch_normalization(conv1)
#         x1 = tf.nn.leaky_relu(batchNorm1)
#         return x1
#
#     def upSamplingBlock(self,currentInput,previousInput,input_channels,output_channels,image_width, image_height,name):
#         print ("Upsampling")
#         x = tf.image.resize(images=currentInput,size=[image_width,image_height], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,align_corners=False,preserve_aspect_ratio=False)
#         print (x.shape)
#         conv11 = self.conv2d(x, 1,input_channels,output_channels,name + "conv11First")
#         x_concat = tf.concat([conv11,previousInput],axis=3)
#         print (x_concat.shape)
#         conv11 = self.conv2d(x_concat, 1,input_channels,output_channels,name + "conv11")
#         batchNorm1 = tf.layers.batch_normalization(conv11)
#         x1 = tf.nn.leaky_relu(batchNorm1)
#         return x1
#
#     def runBlock(self,inputData,in_channels=1,out_channels=5,channel_size=32):
#         self.x1 = self.downSamplingBlock(inputData,input_channels=in_channels,output_channels=0.5*channel_size, down_size=1,name = "DownBlock1")
#         self.x2 = self.downSamplingBlock(self.x1,input_channels=0.5*channel_size,output_channels=channel_size, down_size=2,name = "DownBlock2")
#         self.x3 = self.downSamplingBlock(self.x2,input_channels=channel_size,output_channels=2*channel_size, down_size=2,name = "DownBlock3")
#         self.x4 = self.downSamplingBlock(self.x3, input_channels=2*channel_size,output_channels=4*channel_size, down_size=2,name = "DownBlock4")
#         self.x5 = self.downSamplingBlock(self.x4,input_channels=4*channel_size,output_channels=8*channel_size, down_size=2,name = "DownBlock5")
#         self.x6 = self.upSamplingBlock(self.x5, self.x4,input_channels=8*channel_size, output_channels=4*channel_size,image_width = 30,image_height = 40,name = "UpBlock1")
#         self.x7 = self.upSamplingBlock(self.x6, self.x3,input_channels=4*channel_size, output_channels=2*channel_size,image_width = 60,image_height = 80,name = "UpBlock2")
#         self.x8 = self.upSamplingBlock(self.x7, self.x2,input_channels=2*channel_size, output_channels=channel_size,image_width = 120,image_height = 160, name = "UpBlock3")
#         self.x9 = self.upSamplingBlock(self.x8, self.x1,input_channels=channel_size, output_channels=0.5*channel_size,image_width = 240,image_height = 320, name = "UpBlock4")
#         self.out_conv1 = self.conv2d(self.x9,1,0.5*channel_size,out_channels,name = "Inference/Output")
#         return self.out_conv1
        

            


import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
class Layer:

    def __init__(self,initializer):
        self.weightsDict = {}
        self.biasDict = {}
        if (initializer == "Xavier"):
            # self.tf_initializer = tf.initializers.GlorotUniform()
            # self.tf_initializer = tf.keras.initializers.GlorotUniform()
            self.tf_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        elif (initializer == "Normal"):
            self.tf_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        elif (initializer == "He"):
            self.tf_initializer = tf.keras.initializers.he_normal()

    def deconv2d(self,input_tensor, filter_size, output_size,output_size1, out_channels, in_channels, name, outputName,strides = [1, 1, 1, 1]):
        dyn_input_shape = tf.shape(input_tensor)
        batch_size = dyn_input_shape[0]
        out_shape = tf.stack([batch_size, output_size, output_size1, out_channels])
        filter_shape = [filter_size, filter_size, out_channels, in_channels]
        w = tf.compat.v1.get_variable(name=name, shape=filter_shape)
        h1 = tf.compat.v1.nn.conv2d_transpose(input_tensor, w, out_shape, strides, padding='SAME',name = outputName)
        return h1

    def conv2d(self,inputFeature,filterSize, inputSize, outputSize, name,strides = 1):
        filter_shape = [filterSize, filterSize, inputSize, outputSize]
        self.weightName = name + "weight"
        self.biasName = name + "bias"
        with tf.compat.v1.variable_scope("variable", reuse=tf.compat.v1.AUTO_REUSE):
            self.weightsDict[self.weightName] = tf.compat.v1.get_variable(self.weightName, shape=filter_shape, dtype = tf.float32, initializer=self.tf_initializer)
            self.biasDict[self.biasName] = tf.compat.v1.get_variable(self.biasName, shape = outputSize,dtype = tf.float32, initializer=self.tf_initializer)
        convOutput = tf.compat.v1.nn.conv2d(input = inputFeature, filter = (self.weightsDict[self.weightName]), strides=[1, strides, strides, 1], padding='SAME', name = name)
        finalOutput = tf.compat.v1.nn.bias_add(convOutput, (self.biasDict[self.biasName]))
        return finalOutput

    def depthWiseConvolution(self, inputFeature, inputChannels, filterSize, input_multiplier, strides, name):
        filter_shape = [filterSize, filterSize, inputChannels, input_multiplier]
        self.weightName = name + "weight"
        self.biasName = name + "bias"
        with tf.compat.v1.variable_scope("variable", reuse=tf.compat.v1.AUTO_REUSE):
            self.weightsDict[self.weightName] = tf.compat.v1.get_variable(self.weightName, shape=filter_shape,dtype = tf.float32,initializer=self.tf_initializer)

        out = tf.compat.v1.nn.depthwise_conv2d(input = inputFeature, filter = (self.weightsDict[self.weightName]), strides=[1,strides,strides,1], padding='SAME', name = name)
        return out

    def avgpool2d(self,inputData):
        return tf.compat.v1.nn.max_pool(value = inputData, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')

    def downSamplingBlock(self,x,isTrain,input_channels,output_channels,down_size,name):
        #print (name)
        if down_size == 2:
            x = self.avgpool2d(x)
            multiplier = 2
        if down_size == 1:
            multiplier = 8

        depth1 = self.depthWiseConvolution(x,input_channels,3,multiplier,1,name + "depthconv1")
       	conv1 =  self.conv2d(depth1,1,output_channels,output_channels,name+"conv1",strides = 1)
        print ("Convolution layer shape is ", conv1.shape)
        batchNorm1 = tf.compat.v1.layers.batch_normalization(conv1, training=isTrain)
        x1 = tf.nn.relu(batchNorm1)
        #drop_out = tf.nn.dropout(x1, keep_prob)
        return x1

    def upSamplingBlock(self,currentInput,previousInput,isTrain,deconvFilterSize,input_channels,output_channels,image_width, image_height,name):
        #x = self.deconv2d(currentInput, deconvFilterSize, image_width,image_height,  output_channels, input_channels, name + "deconv", name + "Deconv1",strides=[1, 2, 2, 1]) # 32
        x = tf.compat.v1.image.resize_images(images=currentInput,size=[image_width,image_height], method=tf.image.ResizeMethod.BILINEAR,align_corners=False,preserve_aspect_ratio=False)
        #x = tf.image.convert_image_dtype(x,dtype = tf.float16)
        #x = tf.image.resize(images=currentInput,size=[image_width,image_height])
        print ("Upsampled image shape",x.shape)
        #conv11 = self.conv2d(x, 1,input_channels,output_channels,name + "conv11First")
        x_concat = tf.concat([x,previousInput],axis=3)
        #x_concat = tf.cast(x_concat,dtype = tf.float32)
        conv12 = self.conv2d(x_concat,1,input_channels + output_channels,output_channels,name + "conv12")
        depth12 = self.depthWiseConvolution(conv12,output_channels,3,1,1,name + "depthconv12")
       	conv122 =  self.conv2d(depth12,1,output_channels,output_channels,name+"conv122",strides = 1)
        print ("Convolution layer shape is ", conv122.shape)
        batchNorm1 = tf.compat.v1.layers.batch_normalization(conv122, training=isTrain)
        x1 = tf.nn.relu(batchNorm1)
        return x1

    def fullyConnectedBlock(self, currentInput, inputshape, outputShape, name):
        filter_shape = [inputshape, outputShape]
        self.weightName = name + "weight"
        self.biasName = name + "bias"
        with tf.compat.v1.variable_scope("variable", reuse=tf.compat.v1.AUTO_REUSE):
            self.weightsDict[self.weightName] = tf.compat.v1.get_variable(self.weightName, shape=filter_shape, dtype = tf.float32, initializer=self.tf_initializer)
            self.biasDict[self.biasName] = tf.compat.v1.get_variable(self.biasName, shape = outputShape, dtype = tf.float32, initializer=self.tf_initializer)
        fc1 = tf.matmul(currentInput,self.weightsDict[self.weightName]) +  self.biasDict[self.biasName]
        fc1Activation = tf.nn.relu(fc1,name = name + 'relu')
        return fc1Activation

    def runBlock(self,inputData,isTrain, in_channels=1,out_channels=5,channel_size=16):
        self.x1 = self.downSamplingBlock(inputData,isTrain,input_channels=in_channels,output_channels=int(0.5*channel_size), down_size=1,name = "DownBlock1")
        self.x2 = self.downSamplingBlock(self.x1,isTrain, input_channels=int(0.5*channel_size),output_channels=channel_size, down_size=2,name = "DownBlock2")
        self.x3 = self.downSamplingBlock(self.x2,isTrain, input_channels=channel_size,output_channels=int(2*channel_size), down_size=2,name = "DownBlock3")
        self.x4 = self.downSamplingBlock(self.x3,isTrain, input_channels=2*channel_size,output_channels=4*channel_size, down_size=2,name = "DownBlock4")
        self.x5 = self.downSamplingBlock(self.x4,isTrain,input_channels=4*channel_size,output_channels=8*channel_size, down_size=2,name = "DownBlock5")
        self.flatten = tf.compat.v1.layers.flatten(self.x5)
        print (self.flatten.shape)
        self.fc = self.fullyConnectedBlock(self.flatten, 38400, 300 ,name = "fc")
        self.fc1 = self.fullyConnectedBlock(self.fc, 300, 38400 ,name = "fc1")
        self.encoderOutput = tf.reshape(self.fc1,[-1,15,20,128],name = "Inference/EncoderOutput")
        self.x6 = self.upSamplingBlock(self.x5, self.x4,isTrain,deconvFilterSize = 1 , input_channels=8*channel_size, output_channels=4*channel_size,image_width = 30,image_height = 40,name = "UpBlock1")
        self.x7 = self.upSamplingBlock(self.x6, self.x3,isTrain,deconvFilterSize = 2, input_channels=4*channel_size, output_channels=2*channel_size,image_width = 60,image_height = 80,name = "UpBlock2")
        self.x8 = self.upSamplingBlock(self.x7, self.x2,isTrain, deconvFilterSize = 2, input_channels=2*channel_size, output_channels=channel_size,image_width = 120,image_height = 160, name = "UpBlock3")
        self.x9 = self.upSamplingBlock(self.x8, self.x1,isTrain,deconvFilterSize = 2, input_channels=channel_size, output_channels= int(0.5*channel_size),image_width = 240,image_height = 320, name = "UpBlock4")
        # self.out_conv1 = self.conv2d(self.x9,1,int(0.5*channel_size),out_channels,name = "Output")
        # #self.out_conv1 = self.deconv2d(self.x9, 1, 240,320, out_channels, int(0.5*channel_size), "Output" + "deconv", "Output" + "Deconv1",strides=[1, 1, 1, 1])
        # finalOutput = tf.reshape(self.out_conv1,[-1,240,320,1], name = "Inference/Output")
        self.out_conv1 = self.conv2d(self.x9, 1, 0.5 * channel_size, out_channels, name="Inference/Output")
        return self.out_conv1








            

	