#include "StanceDetector.h"
#include "Instance.h"
#include <chrono>
#include "Argument_helper.h"
#include "Reader.h"
#include <unordered_set>

wzStanceDetector::wzStanceDetector(int memsize) :m_driver(memsize) {
	// TODO Auto-generated constructor stub
}

wzStanceDetector::~wzStanceDetector() {
	// TODO Auto-generated destructor stub
}

int wzStanceDetector::createAlphabet(const vector<Instance>& vecInsts) {
	if (vecInsts.size() == 0) {
		std::cout << "training set empty" << std::endl;
		return -1;
	}
	cout << "Creating Alphabet..." << endl;

        int numInstance;
	for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
		const Instance *pInstance = &vecInsts[numInstance];

		const vector<string> &words = pInstance->m_words;
		const vector<string> &target = pInstance->m_target;

		const string &label = pInstance->m_label;

                m_label_stats[label]++;
		int words_num = words.size();
		for (int i = 0; i < words_num; i++)
		{
			string curword = normalize_to_lowerwithdigit(words[i]);
			m_word_stats[curword]++;
		}
		int target_num = target.size();
		for (int i = 0; i < target_num; i++)
		{
			string curtarget = normalize_to_lowerwithdigit(target[i]);
			m_target_stats[curtarget]++;
		}
  
		if ((numInstance + 1) % m_options.verboseIter == 0) {
			cout << numInstance + 1 << " ";
			if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
				cout << std::endl;
			cout.flush();
		}

		if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
			break;
	}
	cout << numInstance << " " << endl;

	cout << "Label num: " << m_label_stats.size() << endl;
	cout << "Target num: " << m_target_stats.size() << endl;
	cout << "Word num: " << m_word_stats.size() << endl;

	return 0;
}

int wzStanceDetector::addTestAlpha(const vector<Instance>& vecInsts) {
	cout << "Adding word Alphabet..." << endl;

	int numInstance;
	for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
		const Instance *pInstance = &vecInsts[numInstance];
		const vector<string> &words = pInstance->m_words;
		int curInstSize = words.size();
		for (int i = 0; i < curInstSize; ++i) {
			string curword = normalize_to_lowerwithdigit(words[i]);
			if (!m_options.wordEmbFineTune)m_word_stats[curword]++;
		}

		if ((numInstance + 1) % m_options.verboseIter == 0) {
			cout << numInstance + 1 << " ";
			if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
				cout << std::endl;
			cout.flush();
		}

		if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
			break;
	}
	cout << numInstance << " " << endl;

	return 0;
}


void wzStanceDetector::extractFeature(Feature& feat, const Instance* pInstance) {
	feat.clear();
	feat.m_words = pInstance->m_words;
	feat.m_target = pInstance->m_target;
}
void wzStanceDetector::convert2Example(const Instance* pInstance, Example& exam) {
	exam.clear();
	exam.m_label = pInstance->m_label;
	Feature feat;
	extractFeature(feat, pInstance);
	exam.m_feature = feat;
}

void wzStanceDetector::initialExamples(const vector<Instance>& vecInsts, vector<Example>& vecExams) {
	int numInstance;
	for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
		const Instance *pInstance = &vecInsts[numInstance];
		Example curExam;
		convert2Example(pInstance, curExam);
		vecExams.push_back(curExam);

		if ((numInstance + 1) % m_options.verboseIter == 0) {
			cout << numInstance + 1 << " ";
			if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
				cout << std::endl;
			cout.flush();
		}
		if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
			break;
	}
	cout << numInstance << " " << endl;
}

void wzStanceDetector::train(const string& trainFile, const string& devFile, const string& testFile, const string& modelFile, const string& optionFile) {
	if (optionFile != "")
		m_options.load(optionFile);
	m_options.showOptions();

	vector<Instance> trainInsts = readInstancesFromFile(trainFile);
	std::cout << "train instances:" << std::endl;
	printStanceCount(trainInsts);

	vector<Instance> devInsts = readInstancesFromFile(devFile);
	std::cout << "dev instances:" << std::endl;
	printStanceCount(devInsts);

	vector<Instance> testInsts = readInstancesFromFile(testFile);
	std::cout << "test instances:" << std::endl;
	printStanceCount(testInsts);


	createAlphabet(trainInsts);
	addTestAlpha(devInsts);
	addTestAlpha(testInsts);

	m_driver._hyperparams.labelSize = m_label_stats.size();

	vector<Example> trainExamples, devExamples, testExamples;

	initialExamples(trainInsts, trainExamples);
	initialExamples(devInsts, devExamples);
	initialExamples(testInsts, testExamples);

	m_word_stats[unknownkey] = m_options.wordCutOff + 1;
	m_driver._modelparams.wordAlpha.init(m_word_stats, m_options.wordCutOff);
	m_target_stats[unknownkey] = m_options.targetCutOff + 1;
	m_driver._modelparams.targetAlpha.init(m_target_stats, m_options.targetCutOff);
	cout<<"wordfile: "<< m_options.wordFile<<endl;
        if (m_options.wordFile != "") {
		m_driver._modelparams.words.init(m_driver._modelparams.wordAlpha, m_options.wordFile, m_options.wordEmbFineTune);
	}
	else {
		m_driver._modelparams.words.init(m_driver._modelparams.wordAlpha, m_options.wordEmbSize, m_options.wordEmbFineTune);
	}

	m_driver._hyperparams.setRequared(m_options);
	m_driver.initial();


	dtype bestDIS = 0;

	int inputSize = trainExamples.size();

	int batchBlock = inputSize / m_options.batchSize;
	if (inputSize % m_options.batchSize != 0)
		batchBlock++;

	std::vector<int> indexes;
	for (int i = 0; i < inputSize; ++i)
		indexes.push_back(i);

	static Metric eval;
	static Metric eval_t;
	static vector<Example> subExamples;
	int devNum = devExamples.size(), testNum = testExamples.size();
	float max = 0.0, bestFavorDev = 0.0, bestAgainstDev = 0.0;
	int step = 0, best_iter = 0;
        float mingoal = 0.0, testgoal = 0.0, devgoal = 0.0;
	for (int iter = 0; iter < m_options.maxIter; ++iter) {
		std::cout << "##### Iteration " << iter << std::endl;

		random_shuffle(indexes.begin(), indexes.end());
		eval.reset();
		eval_t.reset();
		auto time_start = std::chrono::high_resolution_clock::now();

		for (int updateIter = 0; updateIter < batchBlock; updateIter++) {
                        Graph graph;
			subExamples.clear();
			int start_pos = updateIter * m_options.batchSize;
			int end_pos = (updateIter + 1) * m_options.batchSize;
			if (end_pos > inputSize)
				end_pos = inputSize;

			for (int idy = start_pos; idy < end_pos; idy++) {
				subExamples.push_back(trainExamples[indexes[idy]]);
			}

			int curUpdateIter = iter * batchBlock + updateIter;
			//cout << subExamples[0].m_label<<"dfdfdfdf"<< m_driver._modelparams.labelAlpha.from_string(subExamples[0].m_label);
			dtype cost = m_driver.train(graph, subExamples, curUpdateIter);
			eval.overall_label_count += m_driver._eval.overall_label_count;
			eval.correct_label_count += m_driver._eval.correct_label_count;
              
                        eval_t.overall_label_count += m_driver._eval_t.overall_label_count;
			eval_t.correct_label_count += m_driver._eval_t.correct_label_count;

			if ((curUpdateIter + 1) % m_options.verboseIter == 0) {
				//m_driver.checkgrad(subExamples, curUpdateIter + 1);
				std::cout << "current: " << updateIter + 1 << ", total block: " << batchBlock << std::endl;
				std::cout << "Cost = " << cost << ", Tag Correct(%) = " << eval.getAccuracy() << ", Target correct(%) = " << eval_t.getAccuracy() << ", ";

				auto step_time = std::chrono::high_resolution_clock::now();
				std::cout << "Time : " << std::chrono::duration<double>(step_time - time_start).count() << "s" << std::endl;
			}
			m_driver.updateModel();

		}

		auto time_end = std::chrono::high_resolution_clock::now();
		std::cout << "Train finished. Total time taken is: " << std::chrono::duration<double>(time_end - time_start).count() << "s" << std::endl;

		float accuracy = static_cast<float>(eval.correct_label_count) /
			(eval.overall_label_count);
		std::cout << "train set acc:" << accuracy << std::endl;



		float devAvg = 0.0;
		if (devNum > 0) {
			Metric favorDevMetric, noneDevMetric, againstDevMetric;
			favorDevMetric.reset();
			noneDevMetric.reset();
			againstDevMetric.reset();
			auto time_start = std::chrono::high_resolution_clock::now();

			for (int idx = 0; idx < devExamples.size(); idx++) {
				int result = predict(devExamples[idx].m_feature);
				string count;
				switch (result) {
				case 0:
					count = devInsts[idx].evaluate("AGAINST", againstDevMetric);
					againstDevMetric.predicated_label_count++;
					break;
				case 1:
					count = devInsts[idx].evaluate("FAVOR", favorDevMetric);
					favorDevMetric.predicated_label_count++;
					break;
				case 2:
					count = devInsts[idx].evaluate("NONE", noneDevMetric);
					noneDevMetric.predicated_label_count++;
					break;
				default: cout << "no such stance" << endl;
                                         cout<<result<<endl;
					abort();
				}

				if (count == "AGAINST") againstDevMetric.overall_label_count++;
				if (count == "FAVOR") favorDevMetric.overall_label_count++;
				if (count == "NONE") noneDevMetric.overall_label_count++;

			}


			auto time_end = std::chrono::high_resolution_clock::now();
			std::cout << "Dev finished. Total time taken is: " << std::chrono::duration<double>(time_end - time_start).count() << "s" << std::endl;
			std::cout << "dev favor:" << std::endl;
			favorDevMetric.print();

			std::cout << "dev none:" << std::endl;

			noneDevMetric.print();

			std::cout << "dev against:" << std::endl;

			againstDevMetric.print();


			float F_a = againstDevMetric.getAccuracy();
			float F_f = favorDevMetric.getAccuracy();
                        devgoal = againstDevMetric.getAccuracy() + favorDevMetric.getAccuracy();
			float devAcc = F_a + F_f;
			if (devAcc > max) {
				max = devAcc;
				step = 0;
				bestAgainstDev = F_a;
				bestFavorDev = F_f;
				best_iter = iter;
			}
			else {
				step++;

				if (step == 10) {
					std::cout << "dev set is good enough, stop" << std::endl;
					std::cout << "the best dev favor F is: " << bestFavorDev << std::endl;
					std::cout << "the best dev against F is: " << bestAgainstDev << std::endl;
					std::cout << "the iter is: " << best_iter << std::endl;
					exit(0);
				}
			}


		}
              
                bool reserve = false;
                if(iter == 2) reserve = true;
		float testAvg = 0.0;
		if (testNum > 0) {
			Metric favorTestMetric, noneTestMetric, againstTestMetric;
			favorTestMetric.reset();
			noneTestMetric.reset();
			againstTestMetric.reset();
			auto time_start = std::chrono::high_resolution_clock::now();

			for (int idx = 0; idx < testExamples.size(); idx++) {
				if(reserve){
					cout<<testExamples[idx].m_feature.m_target.at(0)<<" ";
					for (int j = 0; j < testExamples[idx].m_feature.m_words.size(); j++)
					    cout<<testExamples[idx].m_feature.m_words.at(j)<<" ";
					cout<<testExamples[idx].m_label<<endl;
					
				}
				int result = predict(testExamples[idx].m_feature);
				string count;
				switch (result) {
				case 0:
					count = testInsts[idx].evaluate("AGAINST", againstTestMetric);
					againstTestMetric.predicated_label_count++;
					if(reserve) cout<<"AGAINST"<<endl;
					break;
				case 1:
					count = testInsts[idx].evaluate("FAVOR", favorTestMetric);
					favorTestMetric.predicated_label_count++;
					if(reserve) cout<<"FAVOR"<<endl;
					break;
				case 2:
					count = testInsts[idx].evaluate("NONE", noneTestMetric);
					noneTestMetric.predicated_label_count++;
					if(reserve) cout<<"NONE"<<endl;
					break;
				default:
					cout << "unlegal label" << endl;
					abort();
				}
				if (count == "AGAINST") againstTestMetric.overall_label_count++;
				if (count == "FAVOR") favorTestMetric.overall_label_count++;
				if (count == "NONE") noneTestMetric.overall_label_count++;

			}
			auto time_end = std::chrono::high_resolution_clock::now();
			std::cout << "Test finished. Total time taken is: " << std::chrono::duration<double>(time_end - time_start).count() << "s" << std::endl;
			std::cout << "test favor:" << std::endl;
			favorTestMetric.print();
			std::cout << "test none:" << std::endl;
			noneTestMetric.print();
			std::cout << "test against:" << std::endl;
			againstTestMetric.print();
                        
                        testgoal = againstTestMetric.getAccuracy() + favorTestMetric.getAccuracy();

		}
		if(iter == 2){
			cout<<"thats enough!!!"<<endl;
			abort();
		}
                if(devgoal > mingoal){
                    mingoal = devgoal;
                    cout<<"laozhongyi_"<<min(devgoal, testgoal)<<endl;
                }
		// Clear gradients
	}
}

int wzStanceDetector::predict(const Feature &feature) {
	//assert(features.size() == words.size());
	int stance;
        Graph graph;
	m_driver.predict(graph, feature, stance);
	return stance;
}

int main(int argc, char* argv[])
{
	std::string trainFile = "", devFile = "", testFile = "", modelFile = "", optionFile = "";
	std::string outputFile = "";
	dsr::Argument_helper ah;
	int memsize = 0;

	ah.new_named_string("train", "trainCorpus", "named_string", "training corpus to train a model, must when training", trainFile);
	ah.new_named_string("dev", "devCorpus", "named_string", "development corpus to train a model, optional when training", devFile);
	ah.new_named_string("test", "testCorpus", "named_string",
		"testing corpus to train a model or input file to test a model, optional when training and must when testing", testFile);
	ah.new_named_string("model", "modelFile", "named_string", "model file, must when training and testing", modelFile);
	ah.new_named_string("option", "optionFile", "named_string", "option file to train a model, optional when training", optionFile);
	ah.new_named_string("output", "outputFile", "named_string", "output file to test, must when testing", outputFile);
	ah.new_named_int("memsize", "memorySize", "named_int", "This argument decides the size of static memory allocation", memsize);

	srand(0);
	ah.process(argc, argv);
	if (memsize < 0)
		memsize = 0;
	wzStanceDetector the_stancedetector(memsize);

	the_stancedetector.train(trainFile, devFile, testFile, modelFile, optionFile);
	
	return 0;
}
