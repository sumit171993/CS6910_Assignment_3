# CS6910_Assignment_3
In this Project, a Transiliteration(Sequence2Sequence) learning model is built, for transliterate one language to another. 
In this project, this model implemented based on Encoder-Decoder network, in which different combinations of RNN(Recurrent_Neural Network)-RNN, RNN-LSTM(Long_Short_Term_Memory), RNN-GRU(Gated_Recurrent_Unit), GRU_GRU are being analysed. 
WandB platform is used for analysizing based on different hyper-parametres tuning, such as Learning_rate, number of Encoder_Decoder Layers, hidden_size, batch_size, drop_out, Cell_type etc. 
The model is implemented for transliterate the English Language to Hindi Language. 
If we have input embedding size of '**m**' and have'**N**' characters in the Input Sequence then the Total number of Computations is "**m x N**"
Total number of Parameters for the computatation are: 
a: For the Input layer: V x m are the total compuatations at the input layer where "V" is the Vocabulary Size and "m" is the input embedding. 
b: The Total number of Computations done by the network is: T * m (input layer) + T * (3k + m) (encoder) + T * (4k + mVk) (decoder)
How To Run the Code: 
For Training, need to run all the cells. 
Last section is for sweeping. 
There are two colab notebooks prepared, one with out attention consideration
 and the other one with attention into consideration. 
 Link to Colab notebook with out attention: https://colab.research.google.com/drive/1p9E4bPSAYsd6WjyHuNYkPy4Pg9F1R801?usp=sharing
 Link to Colab notebook with attention: https://colab.research.google.com/drive/19pa2CPXl_uM16TLuhVjNAcVuc1CSxlxB?usp=sharing
 First need to put all the three files attached in this repo with corresponding codes in the same location i.e. hin_train.csv, hin_test.csv and hin_valid.csv with corresponding codes need to put in same location before running. 
 
 
 
 In Case of Google colab, need to upload these three csv files before running. 
 
 
 
 
 
 
 
 For python implementation: 
 python deep_final.py              (Without_attention implementation)
 python deep_final_with_attention.py  (With attention implementation)     
 
