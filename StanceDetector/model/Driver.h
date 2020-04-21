/*
 * Driver.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_Driver_H_
#define SRC_Driver_H_

#include <iostream>
#include "ComputionGraph.h"

//A native neural network classfier using only word embeddings

class Driver{
	public:
		Driver(int memsize) {

		}

		~Driver() {

		}

	public:
		ModelParams _modelparams;  // model parameters
		HyperParams _hyperparams;
		Metric _eval;
		Metric _eval_t;
		Metric _eval_;
                vector<GraphBuilder> _builders;
		vector<UniNode*> _output;
		vector<LinearNode*> _output_atheism;
		vector<LinearNode*> _output_abortion;
		vector<LinearNode*> _output_climate;
		vector<LinearNode*> _output_feminism;

		ModelUpdate _ada;  // model update


	public:
		//embeddings are initialized before this separately.
		inline void initial() {
			if (!_hyperparams.bValid()){
				std::cout << "hyper parameter initialization Error, Please check!" << std::endl;
				return;
			}
			if (!_modelparams.initial(_hyperparams)){
				std::cout << "model parameter initialization Error, Please check!" << std::endl;
				return;
			}
			_modelparams.exportModelParams(_ada);
			//_modelparams.exportCheckGradParams(_checkgrad);

			_hyperparams.print();


			setUpdateParameters(_hyperparams.nnRegular, _hyperparams.adaAlpha, _hyperparams.adaEps);
		}




		inline dtype train(Graph& graph, const vector<Example>& examples, int iter) {
			_eval.reset();
			_eval_t.reset();
			_eval_.reset();
		        _builders.clear();
		        _output.clear();
                        _output_atheism.clear();
                        _output_abortion.clear();
                        _output_climate.clear();
                        _output_feminism.clear();
			int example_num = examples.size();

			dtype cost = 0.0;
			
			for (int count = 0; count < example_num; count++) {
				const Example& example = examples[count];
	
				//forward
				
                                std::pair<UniNode*, vector<LinearNode*>>pair = _builders[count].forward(graph, _modelparams, _hyperparams, example.m_feature, true);
                                _output.push_back(pair.first);
                                _output_atheism.push_back(pair.second.at(0));
                                _output_abortion.push_back(pair.second.at(1));
                                _output_climate.push_back(pair.second.at(2));
                                _output_feminism.push_back(pair.second.at(3));
			}
			graph.compute();

			for (int count = 0; count < example_num; count++) {
				const Example& example = examples[count];
                            
				cost += softMaxLoss(_output.at(count), example.m_label, _eval, example_num);
                               softMaxLoss_binary(_output_atheism.at(count), example.m_feature.m_target[0], _eval_t, example_num, _hyperparams.targetLoss);
                               softMaxLoss_binary(_output_abortion.at(count), example.m_feature.m_target[0], _eval_, example_num, _hyperparams.targetLoss);
                               softMaxLoss_binary(_output_climate.at(count), example.m_feature.m_target[0], _eval_, example_num, _hyperparams.targetLoss);
                               softMaxLoss_binary(_output_feminism.at(count), example.m_feature.m_target[0], _eval_, example_num, _hyperparams.targetLoss);
			}

			graph.backward();

			if (_eval.getAccuracy() < 0) {
				std::cout << "strange" << std::endl;
			}

			return cost;
		}

		inline void predict(Graph& graph, const Feature& feature, int& result) {
                    std::pair<UniNode*, vector<LinearNode*>>_P = _builders[0].forward(graph, _modelparams, _hyperparams, feature);
			graph.compute();
			bool bTargetInTweet = IsTargetIntweet(feature);
			predictLoss(_P.first, result, bTargetInTweet);
		}

		inline bool IsTargetIntweet(const Feature& feature) {
			string words = "";
			for (int i = 0; i < feature.m_words.size(); i++)
				words = words + feature.m_words[i];
			string::size_type idx;
			if (feature.m_target[0] == "#hillaryclinton") {
				idx = words.find("hillary");
				if (idx != string::npos) return true;
				idx = words.find("clinton");
				if (idx != string::npos) return true;

			}
			if (feature.m_target[0] == "#donaldtrump") {
				idx = words.find("trump");
				if (idx != string::npos) return true;
				idx = words.find("donald");
				if (idx != string::npos) return true;

			}
			if (feature.m_target[0] == "#climatechange") {
				idx = words.find("climate");
				if (idx != string::npos) return true;
			}
			if (feature.m_target[0] == "#feminism") {
				idx = words.find("feminism");
				if (idx != string::npos) return true;
				idx = words.find("feminist");
				if (idx != string::npos) return true;

			}
			if (feature.m_target[0] == "#prochoice") {
				idx = words.find("abortion");
				if (idx != string::npos) return true;
				idx = words.find("aborting");
				if (idx != string::npos) return true;

			}
			if (feature.m_target[0] == "#atheism") {
				idx = words.find("atheism");
				if (idx != string::npos) return true;
				idx = words.find("atheist");
				if (idx != string::npos) return true;

			}
			return false;
		}


		void updateModel() {
			//_ada.update();
			//_ada.update(5.0);
			//_ada.update(10);
			_ada.updateAdam(10);
		}

		void checkgrad(const vector<Example>& examples, int iter){
			ostringstream out;
			out << "Iteration: " << iter;
		}




	private:
		inline void resetEval() {
			_eval.reset();
		}


		inline void setUpdateParameters(dtype nnRegular, dtype adaAlpha, dtype adaEps){
			_ada._alpha = adaAlpha;
			_ada._eps = adaEps;
			_ada._reg = nnRegular;
		}

};

#endif /* SRC_Driver_H_ */
