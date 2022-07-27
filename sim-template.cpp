#include <iostream>
#include <random>
#include <vector>
#include "./dataset.cpp"
using namespace std;


#define IN2 (32)
#define IN3 (32)


int main() {

    Mnist mnist;
    // vector<vector<double> > trainX = mnist.readTrainingFile("/mnt/c/Users/kuryu/project/oyo/dataset/train-images-idx3-ubyte");
    // vector<double> trainY = mnist.readLabelFile("/mnt/c/Users/kuryu/project/oyo/dataset/train-labels-idx1-ubyte");
    vector<vector<double> > testX = mnist.readTrainingFile("/mnt/c/Users/kuryu/project/oyo/dataset/t10k-images-idx3-ubyte");
    vector<double> testY = mnist.readLabelFile("/mnt/c/Users/kuryu/project/oyo/dataset/t10k-labels-idx1-ubyte");

    // for (int i = 0; i < 28; ++i){
    //     for (int j = 0; j < 28; ++j){
    //         cout << (testX[0][i * 28 + j] >= 64 ? 1 : 0);
    //     }
    //     cout << endl;
    // }
    // cout << testY[0] << endl;

    mt19937 mt(0);
    bool in1[784];
    bool W1[IN2][784] = {

    };
    bool b1[IN2] = {

    };
    bool in2[IN2];
    bool W2[IN3][IN2] = {

    };
    bool b2[IN3] = {

    };
    bool in3[IN3];
    bool W3[10][IN3] = {

    };
    bool b3[10] = {

    };


    int test_num = 10000, cnt = 0;
    for (int test_itr = 0; test_itr < test_num; ++test_itr){
        // int idx = mt() % 10000;
        int idx = test_itr % 10000;

        // in
        for (int in_itr = 0; in_itr < 28*28; ++in_itr){
            in1[in_itr] = (testX[idx][in_itr] > 64); 
        }

        // dence:1
        for (int out_itr = 0; out_itr < 16; ++out_itr){
            uint16_t t = 0;
            for (int in_itr = 0; in_itr < 784; ++in_itr){
                if (W1[out_itr][in_itr] == in1[in_itr]) t++;
            }
            t += b1[out_itr];
            in2[out_itr] = (t >= 784/2);
        }

        // dence:2
        for (int out_itr = 0; out_itr < 16; ++out_itr){
            uint16_t t = 0;
            for (int in_itr = 0; in_itr < 16; ++in_itr){
                if (W2[out_itr][in_itr] == in2[in_itr]) t++;
            }
            t += b2[out_itr];
            in3[out_itr] = (t >= 16/2);
        }

        // dence:3
        int max_itr = 0, max_val = 0; 
        for (int out_itr = 0; out_itr < 10; ++out_itr){
            uint16_t t = 0;
            for (int in_itr = 0; in_itr < 16; ++in_itr){
                if (W3[out_itr][in_itr] == in3[in_itr]) t++;
            }
            t += b3[out_itr];

            if (t > max_val){
                max_val = t;
                max_itr = out_itr;
            }
            // cout << out_itr << ": " << t << endl;
        }
        // cout << "max_itr:" << max_itr  << " testY:" << testY[idx] << endl;
        if (max_itr == testY[idx]) cnt++;
    }
    cout << cnt << "/" << test_num << " [" << 100.0 * cnt / test_num << "%]" << endl; 


    return 0;
}