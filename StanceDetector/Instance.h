#ifndef _INSTANCE_H_
#define _INSTANCE_H_

#include <iostream>
using namespace std;

class Instance
{
public:
	void clear()
	{
		m_words.clear();
		m_label.clear();
		m_target.clear();
	}

	string evaluate(const string& predict_label, Metric& eval) const
	{
		if (predict_label == m_label)
			eval.correct_label_count++;
		//eval.overall_label_count++;
		return m_label;
	}

	void copyValuesFrom(const Instance& anInstance)
	{
		allocate(anInstance.size());
		m_label = anInstance.m_label;
		m_words = anInstance.m_words;
		m_target = anInstance.m_target;
	}

	void assignLabel(const string& resulted_label) {
		m_label = resulted_label;
	}

	int size() const {
		return m_words.size();
	}

	void allocate(int length)
	{
		clear();
		m_words.resize(length);
	}

public:
	vector<string> m_words;
	vector<string> m_target;
	string m_label;
};
void printStanceCount(const vector<Instance> &instances) {
	int favorCount = 0;
	int againstCount = 0;
	int neutralCount = 0;
	for (const Instance &ins : instances) {
		if (ins.m_label == "FAVOR") {
			favorCount++;
		}
		else if (ins.m_label == "AGAINST") {
			againstCount++;
		}
		else if (ins.m_label == "NONE") {
			neutralCount++;
		}
		else {
			abort();
		}
	}

	std::cout << "favor: " << favorCount << " against: " << againstCount << " neutral: " << neutralCount << std::endl;
}
#endif /*_INSTANCE_H_*/
