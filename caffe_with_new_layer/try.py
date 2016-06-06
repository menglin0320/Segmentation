import caffe
net = caffe.Net('/home/menglin/caffe-master/menglin_try/trying.prototxt', '/home/menglin/caffe-master/menglin_try/snapshot_iter_10000.caffemodel', caffe.TEST)

