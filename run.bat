set exe=D:\code\caffe\Build\x64\Release\vehicle_retrieval_kCNNs.exe
set image_folder=.\data\images
set file_list_xml=.\data\image_list.xml
set models=googlenet_softmax,googlenet_up,googlenet_down_model_id,googlenet_triplet
set layers=pool5/7x7_s1,pool5/7x7_s1,pool5/7x7_s1,pool5/7x7_s1
set feature_folder=.\feature
set result_folder=.\result_

%exe% -submit %image_folder% %file_list_xml% ^
%models% %layers% %feature_folder% %result_folder% 

