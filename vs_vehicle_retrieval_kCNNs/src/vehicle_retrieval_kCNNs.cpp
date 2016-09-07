/************************************************************************/
// vehicle retrieval
// author: kaihana@163.com
// date: 2016.07
/************************************************************************/
#include "vehicle_retrieval_kCNNs.h"

/**  Ö÷º¯Êý  **/
int main(int argc, char** argv) {
	if (argc < 3) {
		std::cerr << "Usage: " << argv[0]
			<< " refer to run.bat" << std::endl;
		return 1;
	}
	::google::InitGoogleLogging(argv[0]);

	// args
	string val_or_submit = argv[1];
	string image_folder = argv[2];
	string file_list_xml = argv[3];		
	vector<string> model_names = str_split(argv[4],",");
	vector<string> layer_names = str_split(argv[5], ",");
	string feature_folder = argv[6];
	string result_save_path = argv[7];

	// init param	
	string submit_id;
	vector<string> model_files;
	vector<string> trained_files;
	vector<string> mean_files;
	for (int i = 0; i < model_names.size(); i++){
		model_files.push_back("./model/" + model_names[i] + "/deploy.prototxt");
		trained_files.push_back("./model/" + model_names[i] + "/deploy.caffemodel");
		mean_files.push_back("./model/" + model_names[i] + "/deploy.binaryproto");
		submit_id += (model_names[i] + "_");
	}
	
	string layer_names_str;
	for (int i = 0; i < layer_names.size(); i++){
		string layer_i = layer_names[i];
		for (int j = 0; j < layer_i.size(); j++){
			if (layer_i[j] == '/')
				layer_i.erase(j, 1);
		}
		layer_names_str += layer_i;
	}
	submit_id += (layer_names_str + val_or_submit);

	// feature and retrieval
	if (val_or_submit == "-val"){
		string query_list_path = "E:/dataset/smart_city_car/train/valid2000.txt";
		string ref_list_path = "E:/dataset/smart_city_car/train/train-2000.txt";
		// init
		ImageRetrievalSubmit ImageRetrieval(image_folder, feature_folder + "/" + submit_id
			, model_files, trained_files, mean_files, model_names, layer_names);
		// feature extraction and retrieval
		time_t start_time, stop_time;
		start_time = time(NULL);
		std::cout << "hello, start modeling" << std::endl;
		ImageRetrieval.FeatureExtractionVal(query_list_path);
		ImageRetrieval.FeatureExtractionVal(ref_list_path);
		stop_time = time(NULL);
		printf("Use Time:%ld\n", (stop_time - start_time));

		time_t start_time2, stop_time2;
		start_time2 = time(NULL);
		std::cout << "hello, start retrievaling" << std::endl;
		ImageRetrieval.RetrievalVal(query_list_path, ref_list_path, result_save_path + submit_id + ".xml");
		stop_time2 = time(NULL);
		printf("Use Time:%ld\n", (stop_time2 - start_time2));
	}
	else if (val_or_submit == "-submit"){
		// init
		ImageRetrievalSubmit ImageRetrieval(image_folder, feature_folder + "/" + submit_id
			, model_files, trained_files, mean_files, model_names, layer_names);
		// feature extraction and retrieval
		time_t start_time, stop_time;
		start_time = time(NULL);
		std::cout << "----- feature exatraction stage -----" << std::endl;
		ImageRetrieval.FeatureExtractionSubmit(file_list_xml, "ref");
		ImageRetrieval.FeatureExtractionSubmit(file_list_xml, "query");
		stop_time = time(NULL);
		//printf("Use Time:%ld\n", (stop_time - start_time));

		time_t start_time2, stop_time2;
		start_time2 = time(NULL);
		std::cout << "----- retrieval stage -----" << std::endl;
		ImageRetrieval.RetrievalSubmit(file_list_xml, result_save_path + submit_id + ".xml");
		stop_time2 = time(NULL);
		//printf("Use Time:%ld\n", (stop_time2 - start_time2));
	}
	// error 
	else{
		throw("val_or_submit command error!\n");
	}
	std::cout << "----- congratulations! result has generated -----" << std::endl;
	cout << result_save_path + submit_id + ".xml" << endl;

	return 0;
}
