#ifndef _CONLL_READER_
#define _CONLL_READER_

#include "Reader.h"
#include "N3LDG.h"
#include <sstream>

using namespace std;
/*
 this class reads conll-format data (10 columns, no srl-info)
 */
class InstanceReader : public Reader {
public:
	InstanceReader() {
	}
	~InstanceReader() {
	}

	Instance *getNext() {
		m_instance.clear();
		string strLine1, strLine2;
		if (!my_getline(m_inf, strLine1))
			return NULL;
		if (!my_getline(m_inf, strLine2))
			return NULL;
		if (strLine1.empty())
			return NULL;

		
		vector<string> vecInfo;
		split_bychars(strLine1, vecInfo, "\t");
		for (int i = 0; i < vecInfo.size();i++) cout << vecInfo[i] << " ";
		m_instance.m_label = vecInfo.back();
		if (m_instance.m_label != "NONE" && m_instance.m_label != "FAVOR" && m_instance.m_label != "AGAINST") {
			cout << "label load error!!!" << endl;
			abort();
		}
		//m_instance.m_label = vecInfo[0];
		vecInfo.pop_back();
		//split_bychar(vecInfo[0], m_instance.m_words, ' ');
		//split_bychar(strLine2, m_instance.m_target, ' ');
		//split_bychar(vecInfo[0], m_instance.m_target, ' ');
		int start = 0;
		if (vecInfo[0] == "Atheism") {
			m_instance.m_target = { "atheism" };
			start = 1;
		}
		else if (vecInfo[0] == "Climate") {
			m_instance.m_target = { "climate", "change", "is", "a", "real", "concern" };
			start = 6;
		}
		else if (vecInfo[0] == "Feminist") {
			m_instance.m_target = { "feminist" ,"movement" };
			start = 2;
		}
		else if (vecInfo[0] == "Hillary") {
			m_instance.m_target = { "hillary", "clinton" };
			start = 2;
		}
		else if (vecInfo[0] == "Legalization") {
			m_instance.m_target = { "legalization", "of" ,"abortion" };
			start = 3;
		}
		else if (vecInfo[0] == "Donald") {
			m_instance.m_target = { "donald", "trump" };
			start = 2;
		}
		else {
			std::cout <<"this word: "<< vecInfo[0] <<" is unlegal !"<< std::endl;
			abort();
		}
		for (int i = start; i < vecInfo.size(); i++) {
				m_instance.m_words.push_back(vecInfo[i]);
		}

		return &m_instance;
	}
};

#endif

