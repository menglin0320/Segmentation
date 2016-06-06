weights = './snapshot_iter_10000.caffemodel';
model = './cardata_clean.prototxt';
net = caffe.Net(model, weights, 'train');
%solver = get_solver('snapshot_iter_10000.solverstate');
a = 1;