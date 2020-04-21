#ifndef _EXAMPLE_H_
#define _EXAMPLE_H_

#include <iostream>
#include <vector>

using namespace std;

class Feature
{
public:
	vector<string> m_words;
	vector<string> m_target;
public:
	void clear()
	{
		m_words.clear();
		m_target.clear();
	}
};

class Example
{
public:
	Feature m_feature;
	string m_label;

public:
	void clear()
	{
		m_feature.clear();
		m_label.clear();
	}
};

#endif /*_EXAMPLE_H_*/