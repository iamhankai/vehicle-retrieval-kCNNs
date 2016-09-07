#ifndef VEHICLE_RETRIEVAL_KCNNS_H
#define VEHICLE_RETRIEVAL_KCNNS_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>
#include <hash_map>
#include <direct.h> 
#include "caffe_feature.h"
#include "common_tool.h"

using std::string;
using std::vector;
using std::hash_map;
using std::cout;
using std::endl;
using namespace cv;

class ImageRetrievalSubmit{
public:
	// init
	ImageRetrievalSubmit(const string& input_image_folder,
		const string& input_feature_folder,
		const vector<string>& input_model_files,
		const vector<string>& input_trained_files,
		const vector<string>& input_mean_files,
		const vector<string>& input_model_names,
		const vector<string>& input_layer_names);

	// submit
	void FeatureExtractionSubmit(const string& file_list_xml, const string& ref_or_query);
	void RetrievalSubmit(const string& file_list_xml, const string& result_save_path);
	void FeatureExtractionVal(const string& file_list_txt);
	void RetrievalVal(const string& query_list_path, const string& ref_list_path, const string& result_save_path);

private:
	void FeatureExtractionInner(const vector<string>& flist);
	void RetrievalInner(const vector<string>& qlist, const vector<string>& rlist
		, const vector<string>& q_label_list, const vector<string>& r_label_list
		, hash_map<string, int>& label_count_map, const string& result_save_path);

private:
	string image_folder;
	string feature_folder;

	vector<string> model_files;
	vector<string> trained_files;
	vector<string> mean_files;

	vector<string> model_names;
	vector<string> layer_names;

	// image list
	vector<string> qlist;
	vector<string> q_label_list;
	vector<string> rlist;
	vector<string> r_label_list;
};

ImageRetrievalSubmit::ImageRetrievalSubmit(const string& input_image_folder,
	const string& input_feature_folder,
	const vector<string>& input_model_files,
	const vector<string>& input_trained_files,
	const vector<string>& input_mean_files,
	const vector<string>& input_model_names,
	const vector<string>& input_layer_names){

	image_folder = input_image_folder;
	feature_folder = input_feature_folder;
	model_files = input_model_files;
	trained_files = input_trained_files;
	mean_files = input_mean_files;
	model_names = input_model_names;
	layer_names = input_layer_names;

	// 创建文件夹
	if (_mkdir(feature_folder.c_str()) == 0)
		std::cout << "folder created: " + feature_folder << std::endl;
	else
		std::cout << "folder exists: " + feature_folder << std::endl;
}

/************************************************************************/
/*  feature extraction  */
/************************************************************************/
void ImageRetrievalSubmit::FeatureExtractionSubmit(const string& file_list_xml, const string& ref_or_query){
	//read file list
	vector<string> flist = read_image_list_from_xml(file_list_xml, ref_or_query);
	cout << flist.size() << " image_names loaded!" << endl;
	//
	FeatureExtractionInner(flist);
}

void ImageRetrievalSubmit::FeatureExtractionVal(const string& file_list_txt){
	//read file list
	std::ifstream f_list_file(file_list_txt.c_str());
	vector<string> flist;
	string temp;
	while (getline(f_list_file, temp))
	{
		vector<string> fname_label = str_split(temp, " ");
		vector<string> fname_jpg = str_split(fname_label[0], ".");
		flist.push_back(fname_jpg[0]);
	}
	cout << "file list loaded!" << endl;
	//
	FeatureExtractionInner(flist);
}

void ImageRetrievalSubmit::FeatureExtractionInner(const vector<string>& flist){

	vector<FeatureExtractor> FeatureExtractor_list;
	for (int i = 0; i < model_files.size(); i++){
		FeatureExtractor FeatureExtractor_i(model_files[i], trained_files[i], mean_files[i]);
		FeatureExtractor_list.push_back(FeatureExtractor_i);
		cout << "CNN: " << model_names[i] << " loaded successfully!" << endl;
	}

	//extract feature for every picture
	for (int i = 0; i < flist.size(); i++){
		string file_path = image_folder + "/" + flist[i] + ".jpg";
		// read img
		cv::Mat img = cv::imread(file_path);
		CHECK(!img.empty()) << "Unable to decode image " << file_path;	
		cv::Mat img_float;
		img.convertTo(img_float, CV_32FC3);
		// split to 2 parts
		int mid_height = cvFloor(img.rows / 2);
		cv::Mat up_part = img(Rect(0, 0, img.cols, mid_height));
		cv::Mat down_part = img(Rect(0, mid_height, img.cols, img.rows - mid_height));
		cv::Mat up_part_float;
		up_part.convertTo(up_part_float, CV_32FC3);
		cv::Mat down_part_float;
		down_part.convertTo(down_part_float, CV_32FC3);
		//cout << i << "th image read!" << endl;
		// cnn features concat
		cv::Mat all_fea;
		for (int f = 0; f < FeatureExtractor_list.size(); f++){
			vector<string> layer_names_f = { layer_names[f] };
			vector<vector<float> > cnn_vecs;
			if (model_names[f] == "googlenet_up")
				cnn_vecs = FeatureExtractor_list[f].Predict(up_part_float, layer_names_f);
			else if (model_names[f] == "googlenet_down")
				cnn_vecs = FeatureExtractor_list[f].Predict(down_part_float, layer_names_f);
			else
				cnn_vecs = FeatureExtractor_list[f].Predict(img_float, layer_names_f);
			for (int l = 0; l < cnn_vecs.size(); l++){
				vector<float> vec_l = cnn_vecs[l];
				float* array_l = &vec_l[0];
				cv::Mat fea_l = cv::Mat(vec_l.size(), 1, CV_32FC1, array_l);
				// normalize
				cv::normalize(fea_l, fea_l, 1.0, 0, NORM_L2);
				// concat
				all_fea.push_back(fea_l);
			}
		}
		// write the feature vector
		vector<string> str_list = str_split(file_path, "/");
		string filename = *(str_list.end() - 1);
		string model_file = feature_folder + "/" + filename + ".xml";
		CvMat fea_vec = all_fea;
		cvSave(model_file.c_str(), &fea_vec);
		if(i%50==0) cout << i << "th image feature-extraction successfully!" << endl;
	}
	cout << "all images feature-extraction successfully!" << endl;
}
/************************************************************************/
/*  retrieval  */
/************************************************************************/
void ImageRetrievalSubmit::RetrievalSubmit(const string& file_list_xml, const string& result_save_path){
	//read query list
	vector<string> qlist = read_image_list_from_xml(file_list_xml, "ref");
	vector<string> q_label_list(qlist.size(),"0");
	//cout << qlist.size() << " query name list read!" << endl;
	//read ref list
	vector<string> rlist = read_image_list_from_xml(file_list_xml, "query");
	vector<string> r_label_list(rlist.size(),"0");
	hash_map<string, int> label_count_map;
	//cout << rlist.size() << " ref name list read!" << endl;
	
	//
	RetrievalInner(qlist, rlist, q_label_list, r_label_list, label_count_map, result_save_path);
}

void ImageRetrievalSubmit::RetrievalVal(const string& query_list_path, const string& ref_list_path, const string& result_save_path){
	//read query list
	std::ifstream q_list_file(query_list_path.c_str());
	vector<string> qlist;
	vector<string> q_label_list;
	string temp;
	while (getline(q_list_file, temp))
	{
		vector<string> fname_label = str_split(temp, " ");
		q_label_list.push_back(fname_label[1]);
		vector<string> fname_jpg = str_split(fname_label[0], ".");
		qlist.push_back(fname_jpg[0]);
	}
	cout << qlist.size() << "query name list read!" << endl;
	//read ref list
	std::ifstream r_list_file(ref_list_path.c_str());
	vector<string> rlist;
	vector<string> r_label_list;
	hash_map<string, int> label_count_map;
	while (getline(r_list_file, temp))
	{
		vector<string> fname_label = str_split(temp, " ");
		r_label_list.push_back(fname_label[1]);
		if (label_count_map.find(fname_label[1]) != label_count_map.end()){
			label_count_map[fname_label[1]] += 1;
		}
		else{
			label_count_map[fname_label[1]] = 1;
		}
		vector<string> fname_jpg = str_split(fname_label[0], ".");
		rlist.push_back(fname_jpg[0]);
	}
	cout << rlist.size() << "ref name list read!" << endl;
	//
	RetrievalInner(qlist, rlist, q_label_list, r_label_list, label_count_map,result_save_path);
}

void ImageRetrievalSubmit::RetrievalInner(const vector<string>& qlist, const vector<string>& rlist
	, const vector<string>& q_label_list, const vector<string>& r_label_list
	, hash_map<string, int>& label_count_map, const string& result_save_path){

	//ref feature list
	vector<Mat> r_features;
	for (int i = 0; i < rlist.size(); i++){
		string ri_path = feature_folder + "/" + rlist[i] + ".jpg.xml";
		CvMat* r_feature_temp = (CvMat*)cvLoad(ri_path.c_str()); //1*315
		cv::Mat r_feature(r_feature_temp->rows, r_feature_temp->cols, r_feature_temp->type, r_feature_temp->data.fl);
		r_features.push_back(r_feature);
	}
	cout << "ref features loaded!" << endl;
	/**************************计算每个query与所有test的相似度**************************/
	vector<vector<string>> result_list_list;
	double map = 0;
	double p1 = 0;
	for (int i = 0; i < qlist.size(); i++){
		string qi_path = feature_folder + "/" + qlist[i] + ".jpg.xml";
		CvMat* q_feature_temp = (CvMat*)cvLoad(qi_path.c_str()); //1*315
		cv::Mat q_feature(q_feature_temp->rows, q_feature_temp->cols, q_feature_temp->type, q_feature_temp->data.fl);
		//std::cout << "feature0: " << q_feature.at<float>(0, 0) << std::endl;
		// Sim or Dist
		vector<sim_idx> sim_list(rlist.size());
		for (int j = 0; j < rlist.size(); j++){
			cv::Mat simMat = q_feature.t()*r_features[j];
			double sim = simMat.at<float>(0, 0);
			sim_list[j].sim = sim;
			sim_list[j].idx = j;
		}
		// 释放内存
		cvReleaseMat(&q_feature_temp);
		/********************************** get result **************************************/
		//降序排列
		sort(sim_list.begin(), sim_list.end(), sim_compare);
		// map@k
		int k = 200;
		vector<string> result_label_list;
		for (int j = 0; j < std::min(k, (int)sim_list.size()); j++){
			result_label_list.push_back(r_label_list[sim_list[j].idx]);
		}
		string q_label = q_label_list[i];
		int label_count = label_count_map[q_label];
		double ap = ap_k(q_label_list[i], result_label_list, std::max(1, label_count));
		//std::cout << ap << std::endl;
		map += ap;
		// map@1		
		if (q_label_list[i] == r_label_list[sim_list[0].idx])
			p1++;
		// save
		vector<string> result_list;
		for (int j = 0; j < rlist.size(); j++){
			result_list.push_back(rlist[sim_list[j].idx]);
		}
		result_list_list.push_back(result_list);

		std::cout << i << "th query retrieval successfully!" << std::endl;
	}
	map /= qlist.size();
	p1 /= qlist.size();
	//std::cout << "map@200: " << map << std::endl;
	//std::cout << "map@1: " << p1 << std::endl;
	// write
	save_result_to_xml(qlist, result_list_list, result_save_path);
}

#endif