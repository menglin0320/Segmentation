#include <cfloat>
#include <vector>

#include "caffe/layers/pca_get_coord_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PcaGetCoordLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  this->blobs_.resize(1);
  this->blobs_[0].reset(new Blob<Dtype>());

}

template <typename Dtype>
void PcaGetCoordLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //reshape top
  int top_shape_arr[3] = {bottom[3]->shape(0),bottom[0]->shape(1),bottom[0]->shape(2)};
  vector<int> top_shape(top_shape_arr, top_shape_arr + 3);
  top[0]->Reshape(top_shape);

  Blob<Dtype> & sigma_ = *this->blobs_[0]; 
  Blob<Dtype> * avg_q = bottom[2];
  sigma_.Reshape(top_shape);
  for(int i = 0; i < top_shape[0];++i)
  {
    switch (Caffe::mode()) {
    case Caffe::GPU:
      caffe_copy(avg_q->count(), avg_q->gpu_data(),
          static_cast<Dtype*>(&sigma_.mutable_gpu_data()[i * avg_q->count()]));
      break;
    case Caffe::CPU:  
      caffe_copy(avg_q->count(), avg_q->cpu_data(),
          static_cast<Dtype*>(&sigma_.mutable_cpu_data()[i * avg_q->count()]));
      break;
    default:
      LOG(FATAL) << "Unknown caffe mode.";      
    }
  }

  memset(bottom[3]->mutable_cpu_diff(),0,bottom[3]->count() * sizeof(Dtype));
}

template <typename Dtype>
void PcaGetCoordLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  //initialize sigma_ copy from q avg

  Blob<Dtype>& sigma_ = *this->blobs_[0]; 
  const Dtype* Udata = bottom[0]->cpu_data();
  const Dtype* Bdata = bottom[1]->cpu_data();
  const Dtype* Tdata = bottom[3]->cpu_data();
  int numImages = bottom[3]->shape(0);
  t = bottom[0]->shape(0);
  int numCoords = 2*bottom[0]->shape(1); 
  Dtype extended_coord[] = {1.,1.,1.};
  for (int i = 0; i < numImages; ++i)
  {
     caffe_cpu_gemv<Dtype>(CblasTrans, t,\
     numCoords, (Dtype) 1., Udata,&Bdata[i* t],\
     (Dtype)1.,&sigma_.mutable_cpu_data()[i* numCoords]);
      // std::cout<<"sigma0_: "<<sigma_.cpu_data()[0]<<std::endl;
      // std::cout<<"sigma1_: "<<sigma_.cpu_data()[1]<<std::endl;
      // std::cout<<"sigma2_: "<<sigma_.cpu_data()[2]<<std::endl;
      // std::cout<<"sigma3_: "<<sigma_.cpu_data()[3]<<std::endl;
    for (int j = 0; j < numCoords/2; ++j){
      memcpy(extended_coord,sigma_.cpu_data()+j*2+i*numCoords,2*sizeof(Dtype));
      // memcpy(extended_coord + 3,sigma_.cpu_data()+2*j+i*numCoords + 2,2*sizeof(Dtype));
      caffe_cpu_gemv<Dtype>(CblasNoTrans, 2,\
      3, (Dtype)1., &Tdata[i * 6],extended_coord,\
      (Dtype)0., &top[0]->mutable_cpu_data()[i * numCoords + j*2]);
    }
  }
}

template <typename Dtype>
void PcaGetCoordLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  Blob<Dtype>& sigma_ = *this->blobs_[0];   
  const Dtype* top_diff = top[0]->cpu_diff();

  const Dtype* Udata = bottom[0]->cpu_data();
  const Dtype* Tdata = bottom[3]->cpu_data();
  int numImages = bottom[3]->shape(0);
  int t = bottom[0]->shape(0);
  int numCoords = 2*bottom[0]->shape(1); 

  Dtype * dEdT = bottom[3]->mutable_cpu_diff();
  Dtype * dEdSigma = sigma_.mutable_cpu_diff();
  Dtype * dEdB = bottom[1]->mutable_cpu_diff();
  Dtype rotate_part[4];
  for (int i = 0; i < numImages; ++i)
  {
    for(int j = 0; j < numCoords/2;++j)
    {
      caffe_cpu_gemm<Dtype>(CblasNoTrans,\
      CblasNoTrans, 1,2,1,\
      (Dtype)1.,&top_diff[i*numCoords + j*2],&sigma_.cpu_data()[i*numCoords + j*2],(Dtype)1.,\
      &dEdT[i *6]);

      dEdT[i *6+2] += top_diff[i *numCoords + j*2];

      //second row
      caffe_cpu_gemm<Dtype>(CblasNoTrans,\
      CblasNoTrans, 1,2,1,\
      (Dtype)1.,&top_diff[i*numCoords + j*2+1],&sigma_.cpu_data()[i*numCoords + j*2],(Dtype)1.,\
      &dEdT[i * 6 + 3]);

      dEdT[i * 6 + 5] += top_diff[i *numCoords + j*2+1];

      //for dEdsigma
      rotate_part[0] = Tdata[i * 6];
      rotate_part[1] = Tdata[i * 6+1];
      rotate_part[2] = Tdata[i * 6+3];
      rotate_part[3] = Tdata[i * 6+4];
      caffe_cpu_gemm<Dtype>(CblasNoTrans,\
      CblasNoTrans, 1, 2, 2,\
      (Dtype)1., &top_diff[i*numCoords + j*2],rotate_part,(Dtype)0.,&dEdSigma[i*numCoords + j*2]);
    }

    caffe_cpu_gemv<Dtype>(CblasNoTrans, t,\
    numCoords, (Dtype) 1., Udata,&dEdSigma[i* numCoords],\
    (Dtype)0.,&dEdB[i* t]); 
  }
}
// template <typename Dtype>
// int PcaGetCoordLayer<Dtype>::check_dim_exist(int ind,vector<int>& shape)
// {
//   int size = shape.size();
//   if(ind >= size)
//      return 1;
//   else
//       return shape[size -1 - ind];
// }

// template <typename Dtype>
// void PcaGetCoordLayer<Dtype>::print_sigma_matrix(bool type)
// {
//   vector<int> shape = this->blobs_[0]->shape();
//   int num =  check_dim_exist(3,shape);
//   int channels = check_dim_exist(2,shape);
//   int height = check_dim_exist(1,shape);
//   int width = check_dim_exist(0,shape);
//   for(int i = 0; i < num;++i)
//   {
//     printf("(%d,:,:,:)\n",i);

//     for(int j = 0; j < channels;++j)
//     {
//       printf("(%d,%d,:,:)\n",i,j);
//       for(int k = 0; k < height;++k) 
//       {
//         for(int z = 0; z < width;++z)
//         {
//          if (type){
//             std::cout<<this->blobs_[0]->cpu_data()[i * channels*height*width + j * height * width\
//             +k * width + z] <<" ";
//           }
//           else{
//             std::cout<<this->blobs_[0]->cpu_diff()[i * channels*height*width + j * height * width\
//             +k * width + z] <<" ";
//           } 
//         }
//         printf("\n");
//       }
//     }
//   }
// }

#ifdef CPU_ONLY
STUB_GPU(PcaGetCoordLayer);
#endif

INSTANTIATE_CLASS(PcaGetCoordLayer);
REGISTER_LAYER_CLASS(PcaGetCoord);

}  // namespace caffe
