#include <iostream>
#include <fstream>
#include <random>
#include <vector>

#include <opencv2/opencv.hpp>
using namespace std;

class Mnist{
public:
    int reverseInt (int i) {
        unsigned char c1, c2, c3, c4;

        c1 = i & 255;
        c2 = (i >> 8) & 255;
        c3 = (i >> 16) & 255;
        c4 = (i >> 24) & 255;

        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    }

    vector<vector<double> > readTrainingFile(string filename){
        ifstream ifs(filename.c_str(),std::ios::in | std::ios::binary);
        int magic_number = 0;
        int number_of_images = 0;
        int rows = 0;
        int cols = 0;

        //ヘッダー部より情報を読取る。
        ifs.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);
        ifs.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);
        ifs.read((char*)&rows,sizeof(rows));
        rows= reverseInt(rows);
        ifs.read((char*)&cols,sizeof(cols));
        cols= reverseInt(cols);

        vector<vector<double> > images(number_of_images);
        cout << magic_number << " " << number_of_images << " " << rows << " " << cols << endl;

        for(int i = 0; i < number_of_images; i++){
            images[i].resize(rows * cols);

            for(int row = 0; row < rows; row++){
                for(int col = 0; col < cols; col++){
                    unsigned char temp = 0;
                    ifs.read((char*)&temp,sizeof(temp));
                    images[i][rows*row+col] = (double)temp;
                }
            }
        }
        return images;
    }

    vector<double> readLabelFile(string filename){
        ifstream ifs(filename.c_str(),std::ios::in | std::ios::binary);
        int magic_number = 0;
        int number_of_images = 0;

        //ヘッダー部より情報を読取る。
        ifs.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);
        ifs.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);

        vector<double> label(number_of_images);

        cout << number_of_images << endl;

        for(int i = 0; i < number_of_images; i++){
            unsigned char temp = 0;
            ifs.read((char*)&temp,sizeof(temp));
            label[i] = (double)temp;
        }
        return label;
    }
};

class DenceLayer{
public:
    int data_size = 0;
    int input_width = 0;
    int output_width = 0;
    vector<vector<double> > W; // [in][out]
    vector<double> B; // [out]
    
    DenceLayer(int data_size, int input_width, int output_width){
        this->data_size = data_size;
        this->input_width = input_width;
        this->output_width = output_width;

        random_device seed_gen;
        default_random_engine engine(seed_gen());
        std::normal_distribution<> dist(0.0, 1.0 / sqrt(input_width));
        for (int i = 0; i < this->input_width; ++i){
            for (int j = 0; j < this->output_width; ++j){
                W[i][j] = dist(engine);
            }
        }
        for (int i = 0; i < this->output_width; ++i){
            B[i] = dist(engine);
        }
    }

    vector<vector<double> > forward(vector<vector<double> > in, bool is_training){
        vector<vector<double>> out(this->data_size, vector<double>(this->output_width, 0));
        for (int i = 0; i < this->data_size; ++i){
            for (int j = 0; j < this->output_width; ++j){
                for (int k = 0; k < this->input_width; ++k){
                    out[i][j] += in[i][k] * W[k][j];
                }
                out[i][j] += B[j];
                out[i][j] = out[i][j] > 0 ? out[i][j] : 0;                
            }
        }
        return out;
    }
        
    // def backward(self, grad_z){
    //     grad_y = self._back_activation(grad_z)
    //     grad_x = np.dot(grad_y, self.w.T)
    //     self.w += self.optimizer_w.pop(np.dot(self.x.T, grad_y))
    //     self.b += self.optimizer_b.pop(grad_y.sum(axis=0))
    //     return grad_x
    // }
        
};

int main() {
    // cv::Mat img = cv::imread("lenna.png", -1);
    // if(img.empty()) {
    //     return -1;
    // }
    // cv::namedWindow("Example", cv::WINDOW_AUTOSIZE);
    // cv::imshow("Example", img);
    // cv::waitKey(0);
    // cv::destroyWindow("Example");

    Mnist mnist;
    auto trainX = mnist.readTrainingFile("./data/train-images.idx3-ubyte");
    auto trainY = mnist.readLabelFile("./data/train-labels.idx1-ubyte");

    DenceLayer(trainX.size(), trainX[0].size(), 3).forward(trainX, false);

    /* print */
    // cv::Mat img = cv::Mat::zeros(28, 28, CV_8U);
    // auto tmp = trainX[7];
    // for (int i = 0; i < 28; ++i){
    //     for (int j = 0; j < 28; ++j){
    //         img.data[i*28+j] = tmp[i * 28 + j];
    //         cout << tmp[i * 28 + j] << " ";
    //     } cout << endl;
    // }
    // cv::resize(img, img, cv::Size(), 32, 32, cv::INTER_NEAREST);
    // cv::namedWindow("Example", cv::WINDOW_AUTOSIZE);
    // cv::imshow("Example", img);
    // cv::waitKey(0);
    // cv::destroyWindow("Example");
    // cout << "label: " << trainY[7] << endl;

    


    return 0;
}