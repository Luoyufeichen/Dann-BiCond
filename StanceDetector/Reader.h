#ifndef _JST_READER_
#define _JST_READER_

#pragma once

#include <fstream>
#include <iostream>
#include "Utf.h"
#include "Instance.h"
using namespace std;
class Reader
{
public:
	Reader()
	{
	}

	virtual ~Reader()
	{
		if (m_inf.is_open()) m_inf.close();
	}
	int startReading(const char *filename) {
		if (m_inf.is_open()) {
			m_inf.close();
			m_inf.clear();
		}
		m_inf.open(filename);

    if (!m_inf.is_open()) {
			cout << "Reader::startReading() open file err: " << filename << endl;
			return -1;
		}

		return 0;
	}

	void finishReading() {
		if (m_inf.is_open()) {
			m_inf.close();
			m_inf.clear();
		}
	}

	virtual Instance *getNext() = 0;
protected:
	ifstream m_inf;

	int m_numInstance;

	Instance m_instance;
};
vector<string> readLines(const string &fullFileName) {
	vector<string> lines;
	std::ifstream input(fullFileName);
	for (std::string line; getline(input, line);) {
		lines.push_back(line);
	}
	return lines;
}

void readLineToInstance(const string &line, Instance *instance) {

	vector<string>vec;
	vec.clear();
	string sub = "";
	bool is_space = false;
	for (int i = 0; i < line.length(); i++) {
		if (line[i] == ' ' || line[i] == 9) {
			if (is_space) continue;
			vec.push_back(sub);
			sub = "";
			is_space = true;
			continue;
		}
		sub = sub + line[i];
		is_space = false;
	}
	vec.push_back(sub);
	instance->m_label= vec.back();
	vec.pop_back();
	int start = 1;
	if (vec[0] == "Atheism") {
		instance->m_target = {"#atheism" };
		start = 1;
	}
	else if (vec[0] == "Climate") {
		instance->m_target = {"#climatechange"};
		start = 6;
	}
	else if (vec[0] == "Feminist") {
		instance->m_target = {"#feminism"};
		start = 2;
	}
	else if (vec[0] == "Hillary") {
		instance->m_target = {"#hillaryclinton"};
		start = 2;
	}
	else if (vec[0] == "Legalization") {
		instance->m_target = {"#prochoice"};
		start = 3;
	}
	else if (vec[0] == "Donald") {
		instance->m_target = {"#donaldtrump"};
		start = 2;
	}
	else {
		std::cout << "this word: " << vec[0] << " is unlegal!" << std::endl;
		std::cout << "this sentenceis : " << line << std::endl;
		abort();
	}
	for (int i = start; i < vec.size();i++) {
            string word = normalize_to_lowerwithdigit(vec[i]);
         //   if (word[0] == '@') continue;
	    instance->m_words.push_back(word);
	}

}
vector<Instance> readInstancesFromFile(const string &fullFileName) {
	vector<string> lines = readLines(fullFileName);
	vector<Instance> instances;

	using std::move;
	for (int i = 0; i < lines.size(); ++i) {
		Instance ins;
		readLineToInstance(lines.at(i), &ins);
		instances.push_back(move(ins));
	}

	return instances;
}

#endif

