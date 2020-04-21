#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"
#include "HyperParams.h"


// Each model consists of two parts, building neural graph and defining output losses.
struct GraphBuilder{
	public:
		const static int max_sentence_length = 1024;

	public:
		//LSTM1Builder _lstm_left;
		//LSTM1Builder _lstm_right;
                
		Graph *_pcg;

                

	public:
		//allocate enough nodes 
		inline void createNodes(int sent_length){

		//	_lstm_concat.resize(sent_length * 2);
			
		}

		inline void clear(){
		}



	public:
		// some nodes may behave different during training and decode, for example, dropout
                std::pair<UniNode*, vector<LinearNode*>> forward(Graph &_pcg, ModelParams& model, HyperParams& opts, const Feature& feature, bool bTrain = false){
			//forward
                vector<Node *> encoder_lookups_target;
                vector<Node *> encoder_lookups_l2r;
                vector<Node *> encoder_lookups_r2l;
                vector<Node *> encoder_lookups_tweet;

		UniNode *_neural_output = new UniNode;
		UniNode *_transfer_output = new UniNode;
		UniNode *_target_output = new UniNode;
	        LinearNode *_atheism_output = new LinearNode;
		LinearNode *_abortion_output = new LinearNode;
		LinearNode *_climate_output = new LinearNode;
		LinearNode *_feminist_output = new LinearNode;
		GrlNode *_grl = new GrlNode;
                DynamicLSTMBuilder _lstm_l2r;
                DynamicLSTMBuilder _lstm_r2l;

		vector<ConcatNode*> _lstm_concat;
                
                BucketNode *hidden_bucket_l2r = new BucketNode;

                BucketNode *hidden_bucket_r2l = new BucketNode;

                BucketNode *word_bucket = new BucketNode;

                ConcatNode *_concat = new ConcatNode;

		MaxPoolNode *_max_pooling = new MaxPoolNode
;
                AttentionBuilder *_attention = new AttentionBuilder;

                dtype _drop;

			_max_pooling->init(opts.hiddenSize * 2);
                        _grl->init(opts.hiddenSize * 2);
                        _grl->initGrl(opts.grl);
			_neural_output->setParam(&model.olayer_linear);
			_neural_output->init(opts.labelSize);

                        _transfer_output->setParam(&model.transfer_linear);
                        _transfer_output->setFunctions(&frelu, &drelu);
			_transfer_output->init(opts.hiddenSize * 2);

                        //_target_output->setParam(&model.target_linear);
			//_target_output->init(4);
                        _atheism_output->setParam(&model.atheism_linear);
			_atheism_output->init(2);
                        _abortion_output->setParam(&model.abortion_linear);
			_abortion_output->init(2);

                        _climate_output->setParam(&model.climate_linear);
			_climate_output->init(2);
                        _feminist_output->setParam(&model.feminist_linear);
			_feminist_output->init(2);

                        _drop = opts.dropProb;


			hidden_bucket_l2r->init(opts.hiddenSize);
                        hidden_bucket_l2r->forward(_pcg);
                        
                        hidden_bucket_r2l->init(opts.hiddenSize);
                        hidden_bucket_r2l->forward(_pcg);

                        word_bucket->init(opts.wordDim);
                        word_bucket->forward(_pcg);


                        int words_num = feature.m_words.size();
			int target_num = feature.m_target.size();
			int all_num = words_num + target_num;

                        _attention->init(model._attention_params);

                        if (all_num > max_sentence_length)
				all_num = max_sentence_length;
			for (int i = 0; i < target_num; i++) {
                                LookupNode* input_lookup(new LookupNode);
                                input_lookup->init(opts.wordDim);
                                input_lookup->setParam(model.words);
                                input_lookup->forward(_pcg, feature.m_target[i]);
                         
                                DropoutNode *dropout_node(new DropoutNode(_drop, bTrain));
                                dropout_node->init(opts.wordDim);
                                dropout_node->forward(_pcg, *input_lookup);
                                    
                                encoder_lookups_target.push_back(dropout_node);
                                encoder_lookups_l2r.push_back(dropout_node);

			}
			for (int i = 0; i < words_num; i++) {
                                LookupNode* input_lookup(new LookupNode);
                                input_lookup->init(opts.wordDim);
                                input_lookup->setParam(model.words);
                                input_lookup->forward(_pcg, feature.m_words[i]);
                         
                                DropoutNode *dropout_node(new DropoutNode(_drop, bTrain));
                                dropout_node->init(opts.wordDim);
                                dropout_node->forward(_pcg, *input_lookup);
                                    
                                encoder_lookups_tweet.push_back(dropout_node);
                                encoder_lookups_l2r.push_back(dropout_node);

			}

			for (Node* node : encoder_lookups_l2r) {
                                _lstm_l2r.forward(_pcg, model.lstm_target_left_params, *node, *hidden_bucket_l2r, *hidden_bucket_l2r,  _drop, bTrain);
                        }

                        vector<Node*>::iterator iter = encoder_lookups_target.end();

                        while (iter != encoder_lookups_target.begin()) {
                                _lstm_r2l.forward(_pcg, model.lstm_target_right_params, **(--iter), *hidden_bucket_r2l, *hidden_bucket_r2l,  _drop, bTrain);
                        }


                        vector<Node*>::iterator ite = encoder_lookups_tweet.end();
                        
                        while (ite != encoder_lookups_tweet.begin()) {
                                _lstm_r2l.forward(_pcg, model.lstm_tweet_right_params, **(--ite), *hidden_bucket_r2l, *hidden_bucket_r2l,  _drop, bTrain);
                        }


		
                        for (int i = 0; i <  words_num; i++) {
                                ConcatNode *concat = new ConcatNode;
				concat->init(opts.hiddenSize * 2); 
                                concat->forward(_pcg, {_lstm_l2r._hiddens.at(i + target_num), _lstm_r2l._hiddens.at(all_num - i - 1)});
                                _lstm_concat.push_back(concat);
                        }
                       

			vector<Node*> node_c = toNodePointers(_lstm_concat);
			//vector<Node*> node_c = _lstm_concat;
                        ConcatNode *_target_concat = new ConcatNode;
                        _target_concat->init(opts.hiddenSize * 2);
                        _target_concat->forward(_pcg, {_lstm_l2r._hiddens.at(target_num - 1), _lstm_r2l._hiddens.at(target_num - 1)});
                        _attention->forward(_pcg, node_c, *_target_concat);
                        //_max_pooling->forward(&_pcg, node_c);
                       
                        ////_concat.forward(_pcg, &_lstm_left._hiddens[words_num - 1], &_lstm_right._hiddens[target_num]);
                        _grl->forward(&_pcg, _attention->_hidden);
                        _transfer_output->forward(&_pcg, _attention->_hidden);
                        //_grl->forward(&_pcg, _transfer_output);
                        _neural_output->forward(&_pcg, _transfer_output);
                        //_target_output->forward(&_pcg, _grl);

                        vector<LinearNode*>_tv;
                        _atheism_output->forward(_pcg, *_grl);
                            _tv.push_back(_atheism_output);
                        _abortion_output->forward(_pcg, *_grl);
                            _tv.push_back(_abortion_output);
                        _climate_output->forward(_pcg, *_grl);
                            _tv.push_back(_climate_output);
                        _feminist_output->forward(_pcg, *_grl);
                            _tv.push_back(_feminist_output);
                        return std::pair<UniNode*, vector<LinearNode*>>(_neural_output, _tv);
                        
                       
		}
};

#endif /* SRC_ComputionGraph_H_ */
