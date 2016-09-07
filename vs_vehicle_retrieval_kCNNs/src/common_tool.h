#ifndef COMMON_TOOL_H
#define COMMON_TOOL_H

#include <fstream>
#include <iosfwd>
#include <memory>
#include <string>
#include <algorithm>
#include <utility>
#include <vector>
#include "TinyXML/tinyxml.h"

using std::string;
using std::vector;

/************************************************************************/
/*  获取文件夹下所有文件名 */
/************************************************************************/
//void getJustCurrentFile(string path, vector<string>& files)
//{
//	//文件句柄  
//	long   hFile = 0;
//	//文件信息  
//	struct _finddata_t fileinfo;
//	string p;
//	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
//	{
//		do
//		{
//			if ((fileinfo.attrib &  _A_SUBDIR))
//			{
//				;
//			}
//			else
//			{
//				files.push_back(fileinfo.name);
//				//files.push_back(p.assign(path).append("\\").append(fileinfo.name) );  
//			}
//		} while (_findnext(hFile, &fileinfo) == 0);
//		_findclose(hFile);
//	}
//}
/************************************************************************/
/*  字符串分割函数  */
/************************************************************************/
std::vector<std::string> str_split(std::string str, std::string pattern)
{
	std::string::size_type pos;
	std::vector<std::string> result;
	str += pattern;//扩展字符串以方便操作
	int size = str.size();

	for (int i = 0; i<size; i++)
	{
		pos = str.find(pattern, i);
		if (pos<size)
		{
			std::string s = str.substr(i, pos - i);
			result.push_back(s);
			i = pos + pattern.size() - 1;
		}
	}
	return result;
}
/************************************************************************/
/*  相似度数据结构  */
/************************************************************************/
struct sim_idx{
	double sim;
	int idx;
};
bool sim_compare(sim_idx a1, sim_idx a2){
	return a1.sim > a2.sim;
}
/************************************************************************/
/*  AP@K */
/************************************************************************/
double ap_k(string query_label, vector<string> result_label_list, int true_num)
{
	double ap = 0; // average precision
	double precision_i = 0;
	int right_count = 0;
	for (int i = 0; i < result_label_list.size(); i++){
		if (result_label_list[i] == query_label){
			right_count++;
			precision_i = right_count*1.0 / (i + 1);
			ap += precision_i;
		}
	}
	ap /= true_num;
	return ap;
}
/************************************************************************/
/*  XML读取  */
/************************************************************************/
vector<string> read_image_list_from_xml(string xml_file, string ref_or_query) {
	using namespace std;
	vector<string> image_list;
	//const char * xml_file = "E:/dataset/smart_city_car/val/val_list.xml";
	TiXmlDocument doc;
	if (doc.LoadFile(xml_file.c_str())) {
		//doc.Print();
	}
	else {
		std::cout << "can not parse xml" << std::endl;
		return image_list;
	}
	TiXmlElement* root_element = doc.RootElement();  // root: Message  
	TiXmlElement* info_element = root_element->FirstChildElement();  // Info
	// ref
	TiXmlElement* ref_element = info_element->NextSiblingElement();  // Ref Items  
	TiXmlAttribute* attribute_of_ref = ref_element->FirstAttribute();  // Attribute	
	if (strcmp(attribute_of_ref->Value(), ref_or_query.c_str())){
		//cout << attribute_of_ref->Name() << " : " << attribute_of_ref->Value() << std::endl;
		TiXmlElement* ref_item_element = ref_element->FirstChildElement(); // item0
		for (; ref_item_element != NULL; ref_item_element = ref_item_element->NextSiblingElement()) {
			TiXmlAttribute* attribute_of_item = ref_item_element->FirstAttribute();  // Attribute
			for (; attribute_of_item != NULL; attribute_of_item = attribute_of_item->Next()) {
				//cout << attribute_of_item->Name() << " : " << attribute_of_item->Value() << std::endl;
				string image_name = attribute_of_item->Value();
				image_list.push_back(image_name);
			}
		}
	}
	// query
	TiXmlElement* query_element = ref_element->NextSiblingElement();  // query Items  
	TiXmlAttribute* attribute_of_query = query_element->FirstAttribute();  // Attribute
	if (strcmp(attribute_of_query->Value(), ref_or_query.c_str())){
		cout << attribute_of_query->Name() << " : " << attribute_of_query->Value() << std::endl;
		TiXmlElement* query_item_element = query_element->FirstChildElement(); // item0
		for (; query_item_element != NULL; query_item_element = query_item_element->NextSiblingElement()) {
			TiXmlAttribute* attribute_of_item = query_item_element->FirstAttribute();  // Attribute
			for (; attribute_of_item != NULL; attribute_of_item = attribute_of_item->Next()) {
				//cout << attribute_of_item->Name() << " : " << attribute_of_item->Value() << std::endl;
				string image_name = attribute_of_item->Value();
				image_list.push_back(image_name);
			}
		}
	}

	return image_list;
}
/************************************************************************/
/*  XML写入  */
/************************************************************************/
bool save_result_to_xml(const vector<string> query_list, const vector<vector<string>> result_list_list, const string save_path)
{

	if (0 == query_list.size()){
		return false;
	}
	else{

	}

	TiXmlDocument doc;
	TiXmlDeclaration * decl = NULL;
	TiXmlElement * messageElement = NULL;
	TiXmlElement * infoElement = NULL;
	TiXmlElement * itemsElement = NULL;

	//header
	decl = new TiXmlDeclaration("1.0", "gb2312", "");

	messageElement = new TiXmlElement("Message");
	messageElement->SetAttribute("Version", "1.0");

	infoElement = new TiXmlElement("Info");
	const int BUFFER_SIZE = 256;
	char evaluateType[BUFFER_SIZE];
	memset(evaluateType, 0, BUFFER_SIZE);
	sprintf_s(evaluateType, "%d", int(6));
	infoElement->SetAttribute("evaluateType", evaluateType);
	infoElement->SetAttribute("mediaFile", "VehicleRetrieval");

	itemsElement = new TiXmlElement("Items");

	for (int i = 0; i < query_list.size(); i++){
		string query_name = query_list[i];
		vector<string> result_list = result_list_list[i];
		string result_str;
		for (int j = 0; j < std::min(200, (int)result_list.size()); j++)
			result_str += (result_list[j] + " ");

		TiXmlElement *itemElement = new TiXmlElement("Item");
		itemElement->SetAttribute("imageName", query_name.c_str());
		itemsElement->LinkEndChild(itemElement);
		itemElement->LinkEndChild(new TiXmlText(result_str.c_str()));
	}

	messageElement->LinkEndChild(infoElement);
	messageElement->LinkEndChild(itemsElement);
	doc.LinkEndChild(decl);
	doc.LinkEndChild(messageElement);
	doc.SaveFile(save_path.c_str());
	doc.Clear();

	return true;
}

#endif