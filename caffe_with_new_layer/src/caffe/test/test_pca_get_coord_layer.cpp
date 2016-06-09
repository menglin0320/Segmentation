#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/pca_get_coord_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"


namespace caffe {

template <typename TypeParam>
class PcaGetCoordLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
 	 PcaGetCoordLayerTest()
 	 	: numimages(4),
 	 	  T(3),
 	 	  numlabels(5),
 	 	  blob_bottom_a_(new Blob<Dtype>()),
 	 	  blob_bottom_b_(new Blob<Dtype>()),
 	 	  blob_bottom_c_(new Blob<Dtype>()),
 	 	  blob_bottom_d_(new Blob<Dtype>()),
 	 	  blob_top_(new Blob<Dtype>()) {
        int Ushape_arr[] = {T,numlabels,2};
        vector<int>  Ushape(Ushape_arr,Ushape_arr + 3);
        blob_bottom_a_->Reshape(Ushape);

			  int Bshape_arr[] = {numimages,T};
 	 	  	vector<int>  Bshape(Bshape_arr,Bshape_arr + 2);
 	 	  	blob_bottom_b_->Reshape(Bshape);
 	 	
 	 		  int Qshape_arr[] = {numlabels,2};
 	 	  	vector<int>  Qshape(Qshape_arr,Qshape_arr + 2);
 	 	  	blob_bottom_c_->Reshape(Qshape);
 	 	
 	 	  	int Tshape_arr[] = {numimages,2,3};
 	 	  	vector<int>  Tshape(Tshape_arr,Tshape_arr + 3);
 	 	  	blob_bottom_d_->Reshape(Tshape);	
 	 		
 	 		Caffe::set_random_seed(1701);
 	 		FillerParameter filler_param;
    		UniformFiller<Dtype> filler(filler_param);
    		filler.Fill(this->blob_bottom_a_);
    		filler.Fill(this->blob_bottom_b_);
	    	filler.Fill(this->blob_bottom_c_);
        filler.Fill(this->blob_bottom_d_);
	    	blob_bottom_vec_.push_back(blob_bottom_a_);
	    	blob_bottom_vec_.push_back(blob_bottom_b_);
	    	blob_bottom_vec_.push_back(blob_bottom_c_);
	    	blob_bottom_vec_.push_back(blob_bottom_d_);
	    	blob_top_vec_.push_back(blob_top_);
    }

    int check_dim_exist(int ind,vector<int>& shape)
    {
      int size = shape.size();
      if(ind >= size)
        return 1;
      else
        return shape[size -1 - ind];
    }
    void print_blob(Blob<Dtype>* Blob_)
    {
       vector<int> shape = Blob_->shape();
       int num =  check_dim_exist(3,shape);
       int channels = check_dim_exist(2,shape);
       int height = check_dim_exist(1,shape);
       int width = check_dim_exist(0,shape);
       for(int i = 0; i < num;++i)
       {
          printf("(%d,:,:,:)\n",i);

          for(int j = 0; j < channels;++j)
          {
            printf("(%d,%d,:,:)\n",i,j);
            for(int k = 0; k < height;++k) 
            {

              for(int z = 0; z < width;++z)
              {
                std::cout<<Blob_->cpu_data()[i * channels*height*width + j * height * width\
                  +k * width + z] <<" "; 
              }
              printf("\n");
            }
          }
       }
    }

    void print_diff(Blob<Dtype>* Blob_)
    {
       vector<int> shape = Blob_->shape();
       int num =  check_dim_exist(3,shape);
       int channels = check_dim_exist(2,shape);
       int height = check_dim_exist(1,shape);
       int width = check_dim_exist(0,shape);
       for(int i = 0; i < num;++i)
       {
          printf("(%d,:,:,:)\n",i);

          for(int j = 0; j < channels;++j)
          {
            printf("(%d,%d,:,:)\n",i,j);
            for(int k = 0; k < height;++k) 
            {

              for(int z = 0; z < width;++z)
              {
                if(Caffe::mode() == Caffe::CPU)
                  std::cout<<Blob_->cpu_diff()[i * channels*height*width + j * height * width\
                  +k * width + z] <<" ";
                else{
                  # if __CUDA_ARCH__>=200
                   std::cout<<Blob_->gpu_diff()[i * channels*height*width + j * height * width\
                  +k * width + z] <<" ";
                  #endif
                }   
              }
              printf("\n");
            }
          }
       }
    }
 	 	virtual ~PcaGetCoordLayerTest(){
 	 		delete blob_bottom_a_;
 	 		delete blob_bottom_b_;
 	 		delete blob_bottom_c_;
 	 		delete blob_bottom_d_;
 	 		delete blob_top_;
 	 	}
 	int numimages;
 	int T;
 	int numlabels;
 	  Blob<Dtype>* const blob_bottom_a_;
  	Blob<Dtype>* const blob_bottom_b_;
  	Blob<Dtype>* const blob_bottom_c_;
  	Blob<Dtype>* const blob_bottom_d_;
  	Blob<Dtype>* const blob_top_;
  	vector<Blob<Dtype>*> blob_bottom_vec_;
  	vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(PcaGetCoordLayerTest, TestDtypesAndDevices);


TYPED_TEST(PcaGetCoordLayerTest, TestProd) {

  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  printf("before setup\n");
  shared_ptr<PcaGetCoordLayer<Dtype> > layer(
  new PcaGetCoordLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  printf("pass setup\n");
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  
  printf("-------------Udata-----------------\n");
  this->print_blob(&*this->blob_bottom_a_);
  printf("\n\n\n");
  printf("-------------Bdata-----------------\n");
  this->print_blob(&*this->blob_bottom_b_);
  printf("\n\n\n");
  printf("-------------avg_qdata-----------------\n");
  this->print_blob(&*this->blob_bottom_c_);
  printf("\n\n\n");
  printf("-------------Tdata-----------------\n");
  this->print_blob(&*this->blob_bottom_d_);
  // printf("-------------sigma-----------------\n");
  // layer->print_sigma_matrix(true);
  printf("-------------coord-----------------\n");
  printf("\n\n\n");
  //this->print_blob(&*layer->blobs_[0]);
  this->print_blob(&*this->blob_top_);
}

// TYPED_TEST(PcaGetCoordLayerTest, TestBack) {
//   typedef typename TypeParam::Dtype Dtype;
//   LayerParameter layer_param;

//   shared_ptr<PcaGetCoordLayer<Dtype> > layer(
//   new PcaGetCoordLayer<Dtype>(layer_param));
//   layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//   layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

//   vector<bool> propagate_down(4,true);

//   Blob<Dtype>* const diff = new Blob<Dtype>();
//   diff->ReshapeLike(*this->blob_top_);
//   {
//      FillerParameter filler_param;
//      UniformFiller<Dtype> filler(filler_param);
//      filler.Fill(diff);
//   }
//   caffe_copy(this->blob_top_vec_[0]->count(),
//   diff->cpu_data(),
//   this->blob_top_vec_[0]->mutable_cpu_diff());
//   layer->Backward(this->blob_top_vec_,
//         propagate_down,
//         this->blob_bottom_vec_);
  
//   printf("-------------top_diff-----------------\n");
//   this->print_diff(&*this->blob_top_);

//   printf("-------------Sigma-----------------\n");
//   layer->print_sigma_matrix(true);

//   printf("-------------Udata-----------------\n");
//   this->print_blob(&*this->blob_bottom_a_);
//   printf("\n\n\n");

//   printf("-------------SigmaDiff-----------------\n");
//   layer->print_sigma_matrix(false);

//   printf("-------------Bdiff-----------------\n");
//   this->print_diff(&*this->blob_bottom_b_);

//   printf("-------------Tdiff-----------------\n");
//   this->print_diff(&*this->blob_bottom_d_);  
// }
}