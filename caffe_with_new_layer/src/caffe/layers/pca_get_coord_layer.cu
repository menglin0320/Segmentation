#include <cfloat>
#include <vector>

#include "caffe/layers/pca_get_coord_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PcaGetCoordLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  //initialize sigma_ copy from q avg

  Blob<Dtype>& sigma_ = *this->blobs_[0]; 
  const Dtype* Udata = bottom[0]->gpu_data();
  const Dtype* Bdata = bottom[1]->gpu_data();
  const Dtype* Tdata = bottom[3]->gpu_data();
  int numImages = bottom[3]->shape(0);
  t = bottom[0]->shape(0);
  int numCoords = 2*bottom[0]->shape(1); 
  Dtype extended_coord[] = {1.,1.,1.};
  for (int i = 0; i < numImages; ++i)
  {
     caffe_gpu_gemv<Dtype>(CblasTrans, t,\
     numCoords, (Dtype) 1., Udata,&Bdata[i* t],\
     (Dtype)1.,&sigma_.mutable_gpu_data()[i* numCoords]);
      // std::cout<<"sigma0_: "<<sigma_.gpu_data()[0]<<std::endl;
      // std::cout<<"sigma1_: "<<sigma_.gpu_data()[1]<<std::endl;
      // std::cout<<"sigma2_: "<<sigma_.gpu_data()[2]<<std::endl;
      // std::cout<<"sigma3_: "<<sigma_.gpu_data()[3]<<std::endl;
    for (int j = 0; j < numCoords/2; ++j){
      memcpy(extended_coord,sigma_.gpu_data()+j*2+i*numCoords,2*sizeof(Dtype));
      // memcpy(extended_coord + 3,sigma_.gpu_data()+2*j+i*numCoords + 2,2*sizeof(Dtype));
      caffe_gpu_gemv<Dtype>(CblasNoTrans, 2,\
      3, (Dtype)1., &Tdata[i * 6],extended_coord,\
      (Dtype)0., &top[0]->mutable_gpu_data()[i * numCoords + j*2]);
    }
  }
}

template <typename Dtype>
void PcaGetCoordLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  Blob<Dtype>& sigma_ = *this->blobs_[0];   
  const Dtype* top_diff = top[0]->gpu_diff();

  const Dtype* Udata = bottom[0]->gpu_data();
  const Dtype* Tdata = bottom[3]->gpu_data();
  int numImages = bottom[3]->shape(0);
  int t = bottom[0]->shape(0);
  int numCoords = 2*bottom[0]->shape(1); 

  Dtype * dEdT = bottom[3]->mutable_gpu_diff();
  Dtype * dEdSigma = sigma_.mutable_gpu_diff();
  Dtype * dEdB = bottom[1]->mutable_gpu_diff();
  Dtype rotate_part[4];
  for (int i = 0; i < numImages; ++i)
  {
    for(int j = 0; j < numCoords/2;++j)
    {
      caffe_gpu_gemm<Dtype>(CblasNoTrans,\
      CblasNoTrans, 1,2,1,\
      (Dtype)1.,&top_diff[i*numCoords + j*2],&sigma_.gpu_data()[i*numCoords + j*2],(Dtype)1.,\
      &dEdT[i *6]);

      dEdT[i *6+2] += top_diff[i *numCoords + j*2];

      //second row
      caffe_gpu_gemm<Dtype>(CblasNoTrans,\
      CblasNoTrans, 1,2,1,\
      (Dtype)1.,&top_diff[i*numCoords + j*2+1],&sigma_.gpu_data()[i*numCoords + j*2],(Dtype)1.,\
      &dEdT[i * 6 + 3]);

      dEdT[i * 6 + 5] += top_diff[i *numCoords + j*2+1];

      //for dEdsigma
      rotate_part[0] = Tdata[i * 6];
      rotate_part[1] = Tdata[i * 6+1];
      rotate_part[2] = Tdata[i * 6+3];
      rotate_part[3] = Tdata[i * 6+4];
      caffe_gpu_gemm<Dtype>(CblasNoTrans,\
      CblasNoTrans, 1, 2, 2,\
      (Dtype)1., &top_diff[i*numCoords + j*2],rotate_part,(Dtype)0.,&dEdSigma[i*numCoords + j*2]);
    }

    caffe_gpu_gemv<Dtype>(CblasNoTrans, t,\
    numCoords, (Dtype) 1., Udata,&dEdSigma[i* numCoords],\
    (Dtype)0.,&dEdB[i* t]); 
  }
INSTANTIATE_LAYER_GPU_FUNCS(PcaGetCoordLayer);

}  // namespace caffe
