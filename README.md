# Denosing Dirty Document

## Prerequisite
We use a modified version of [keras](https://github.com/zxytim/keras) which add a generic Maxout layer.

## Config file description
a config file must provide with two functions:
+ get() which returns a dict of 'model', 'data_train' and 'data_val'
+ get_data(list_path)
	The content of file located list_path is two image paths a line, name
	X and y respectively. This function should return a list of numpy array
	whose axes is (c, h, w), representing a multi-channel image.
	We run the model given by get() on (1, c, h, w) image and it should
	produce a image whose size is identical to y

