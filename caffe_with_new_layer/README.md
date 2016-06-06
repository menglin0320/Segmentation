Name of the new layer: pca_get_coord_layer

pca_get_coord_layer.cpp test_pca_get_coord_layer.cpp and pca_get_coord_layer.hpp added

CPU version tested but GPU version not tested.(not able to print out gpu_data)

function for print out sigma is commented, if you want to try test backpropagation,uncomment those line and uncomment codes in test_pca_get_coord_layer.cpp

No parameters needed for pca layer

bottom[0] should be u and it's shape should be num_images*t*num_marks*2
bottom[1] should be b and it's shape should be num_images*t
bottom[2] should be avg_q and it's shape should be num_images*num_marks*2
bottom[T] should be u and it's shape should be num_images*2*3(make sure the data is row major, otherwise
[a b c]   will be read as [a d b]
[d e f]			  [e c f]
