# -*- coding : utf-8 -*-
# 原始CNN
# 2018_03_21
#===================================================
import numpy as np 
from activators import ReluActivator , IdentityActivator

#获取卷积区域
def get_patch(input_array , i , j , filter_width , filter_height , stride):
	'''
	从输入数组中获取本次卷积的区域，自动适配输入为2D和3D的情况
	'''
	start_i = i * stride
	start_j = j * stride
	if input_array.dim == 2:
		return input_array[start_i : start_i + filter_height , start_j : start_j + filter_width]
	elif input_array.dim == 3:
		return input_array[: , start_i : start_i + filter_height , start_j : start_j + filter_width]

#获取一个2D区域的最大值的索引
def get_max_index(array):
	max_i = 0
	max_j = 0
	max_value = array[0 , 0]
	for i in range(array.shape[0]):
		for j in range(array.shape[1]):
			if(array[i , j] > max_value):
				max_value = array[i , j]
				max_i , max_j = i , j
	return max_i , max_j

#计算卷积，是互相关操作
def conv(input_array , kernel_array , output_array , stride , bias):
	'''
	计算卷积，自动适配输入为2D和3D的情况
	'''
	channel_number = input_array.ndim
	output_width = output_array.shape[1]
	output_height = output_array.shape[0]
	kernel_width = kernel_array.shape[1]
	kernel_height = kernel_array.shape[0]
	for i in range(output_height):
		for j in range(output_width):
			output_array[i][j] = (
				get_patch(input_array , i , j , kernel_width ,
					kernel_height , stride) * kernel_array).sum() + bias

#对数组增加zero padding
def padding(input_array , zp):
	'''
	对数组增加zero padding，自动适配2D的情况
	'''
	if zp == 0:
		retrun input_array
	else:
		if(input_array.ndim == 3):
			input_width = input_array.shape[2]
			input_height = input_array.shape[1]
			input_depth = input_array.shape[0]
			padded_array = np.zeros((input_depth , input_height + 2 * zp , input_width + 2 * zp))
			padded_array[: , zp : zp + input_height , zp : zp + input_width] = input_array
			return padded_array
		elif (input_array.ndim == 2):
			input_width = input_array.shape[1]
			input_height = input_array.shape[0]
			padded_array = np.zeros((input_height + 2 * zp , input_width + 2 * zp))
			padded_array[zp : zp + input_height , zp : zp + input_width] = input_array
			return padded_array

#对numpy数组进行element wise操作
#??????????????????????????????????????
def element_wise_op(array , op):
	for i in np.nditer(array , op_flags = ['readwrite']):
		i[...] = op(i)

#卷积核类
class Filter(object):
	def __init__(self , width , height , depth):
		self.weights = np.random.uniform(-1e-4 , 1e-4 , (depth , height , width))
		self.bias = 0
		self.weights_grad = np.zeros(self.weights.shape)
		self.bias_grad = 0

	def __repr__(self):
		return 'filter weights : \n%s\nbias : \n%s' % (repr(self.weights) , repr(self.bias))

	def get_weights(self):
		return self.weights

	def get_bias(self):
		return self.bias

	def update(self , learning_rate):
		self.weights -= learning_rate * self.weights_grad
		self.bias -= learning_rate * self.bias_grad

#卷积层类
class ConvLayer(object):

	def __init__(self , input_width , input_height , channel_number,
		         filter_width , filter_height , filter_number ,
		         zero_padding , stride , activator , learning_rate):
		self.input_width = input_width
		self.input_height = input_height
		self.channel_number = channel_number
		self.filter_width = filter_width
		self.filter_height = filter_height
		self.filter_number = filter_number
		self.zero_padding = zero_padding
		self.stride = stride
		self.output_width = ConvLayer.calculate_output_size(self.input_width , filter_width , zero_padding , stride)
		self.output_height = ConvLayer.calculate_output_size(self.input_height , filter_height , zero_padding , stride)
		self.output_array = np.zeros((self.filter_number , self.output_height , self.output_width))
		self.filters = []
		for i in range(filter_number):
			self.filters.append(Filter(filter_width , filter_height , filter_number))
		self.activator = activator
		self.learning_rate = learning_rate

	def forward(self , input_array):
		'''
		计算卷积层的输出
		输出结果保存在self.output_array
		'''
		self.input_array = input_array
		self.padded_input_array = padding(input_array , self.zero_padding)
		for f in range(self.filter_number):
			filter = self.filters[f]
			conv(self.padded_input_array , filter.get_weights() , self.output_array[f] , self.stride , filter.get_bias())
			element_wise_op(self.output_array , self.activator.forward)  #对输出的每一个元素做激活操作

	def backward(self , input_array , sensitivity_array , activator):
		'''
		计算传递给前一层的误差项，以及计算每个权重的梯度
		前一层的误差项保存在self.delta_array，梯度保存在Filter对象的weights_grad中
		'''
		self.forward(input_array)
		self.bp_sensitivity_map(sensitivity_array , activator)
		self.bp_gradient(sensitivity_array)

	def update(self):
		'''
		按照梯度下降，更新权重
		'''
		for filter in self.filters:
			filter.update(self.learning_rate)

	def bp_sensitivity_map(self , sensitivity_array , activator):
		'''
		计算传递到上一层的sensitivity_map
		sensitivity_array：本层的sensitivity map
		activator：上一层的激活函数
		'''
		#处理卷积步长，对原始sensitivity map进行扩展
		expanded_array = self.expand_sensitivity_map(sensitivity_array)
		#full卷积，对sensitivity map进行zero padding
		#虽然原始输入的zero padding单元也会获得残差，但这个残差不需要继续向上传播，因此就不计算了
		expanded_width = expanded_array.shape[2]
		#zero padding的值
		zp = (self.input_width + self.filter_width - 1 - expanded_width) / 2
		padded_array = padding(expanded_array , zp)
		#初始化delta_array,用于保存传递到上一层的sensitivity map
		self.delta_array = self.create_delta_array()
		#对于具有多个filter的卷积层来说，最终传递到上一层的sensitivity map相当于所有filter的sensitivity map之和
		#注意：这里的求和只是针对所有的num求和，而不是针对所有的channel求和；
		for f in range(self.filter_number):
			filter = self.filters[f]
			#将filter的权重翻转180度
			filpped_weights = np.array(map(lambda i : np.rot90(i , 2) , filter.get_weights()))
			#计算与一个filter对应的delta_array
			delta_array = self.create_delta_array()
			for d in range(delta_array.shape[0]):
				conv(padded_array[f] , filpped_weights[d] , delta_array[d] , 1 , 0)

