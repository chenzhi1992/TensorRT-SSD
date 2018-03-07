#include "common.h"
std::string locateFile(const std::string& input, const std::vector<std::string> & directories)
{
    std::string file;
	const int MAX_DEPTH{10};
    bool found{false};
    for (auto &dir : directories)
    {
        file = dir + input;
	std::cout << file << std::endl;
        for (int i = 0; i < MAX_DEPTH && !found; i++)
        {
            std::ifstream checkFile(file);
            found = checkFile.is_open();
            if (found) break;
            file = "../" + file;
        }
        if (found) break;
        file.clear();
    }
    std::cout << file << std::endl;
    assert(!file.empty() && "Could not find a file due to it not existing in the data directory.");
    return file;
}

void readPGMFile(const std::string& fileName,  uint8_t *buffer, int inH, int inW)
{
	std::ifstream infile(fileName, std::ifstream::binary);
    assert(infile.is_open() && "Attempting to read from a file that is not open.");
	std::string magic, h, w, max;
	infile >> magic >> h >> w >> max;
	infile.seekg(1, infile.cur);
	infile.read(reinterpret_cast<char*>(buffer), inH*inW);
}

/*********************************/
/* Updated date： 2018.3.7
/*This is my own implementation of the detectout layer code， because I met a mistake with the detectout api of 
/*tensorrt3.0 a few months ago. You can use the detectout api of tensorrt3.0 correctly by adding an extra output 
/*in the deploy prototxt file. Please refer to my deploy prototxt.
/********************************/
// Retrieve all location predictions.
void GetLocPredictions(const float* loc_data,
                       const int num_preds_per_class, const int num_loc_classes,
                       std::vector<std::vector<float> >* loc_preds) {
        for (int p = 0; p < num_preds_per_class; ++p) {
            int start_idx = p * num_loc_classes * 4;
            vector<float> labelbbox;
            for (int c = 0; c < num_loc_classes; ++c) {
                labelbbox.push_back(loc_data[start_idx + c * 4]);
                labelbbox.push_back(loc_data[start_idx + c * 4 + 1]);
                labelbbox.push_back(loc_data[start_idx + c * 4 + 2]);
                labelbbox.push_back(loc_data[start_idx + c * 4 + 3]);

                loc_preds->push_back(labelbbox);
            }

        }
}

// Retrieve all confidences.
void GetConfidenceScores(const float* conf_data,
                         const int num_preds_per_class, const int num_classes,
                         vector<vector<float> >* conf_preds) {
        for (int p = 0; p < num_preds_per_class; ++p) {
            int start_idx = p * num_classes;
            vector<float> conf_classes;
            for (int c = 0; c < num_classes; ++c) {
                conf_classes.push_back(conf_data[start_idx + c]);
            }
            conf_preds->push_back(conf_classes);
        }
}

// Retrieve all prior bboxes. bboxes and variances
void GetPriorBBoxes(const float* prior_data, const int num_priors,
                    vector<vector<float> >* prior_bboxes,
                    vector<vector<float> >* prior_variances) {
    for (int i = 0; i < num_priors; ++i) {
        int start_idx = i * 4;
        vector<float> prior_bbox;
        prior_bbox.push_back(prior_data[start_idx]);
        prior_bbox.push_back(prior_data[start_idx + 1]);
        prior_bbox.push_back(prior_data[start_idx + 2]);
        prior_bbox.push_back(prior_data[start_idx + 3]);
        prior_bboxes->push_back(prior_bbox);
    }

    for (int i = 0; i < num_priors; ++i) {
        int start_idx = (num_priors + i) * 4;
        vector<float> prior_variance;
        vector<float> var;
        for (int j = 0; j < 4; ++j) {
            prior_variance.push_back(prior_data[start_idx + j]);
        }
        prior_variances->push_back(prior_variance);
    }
}

/* code_type: 0 = CORNER; 1 = CENTER_SIZE; 2 = CORNER_SIZE
 *
 */
void DecodeBBox(
        const vector<float>& prior_bbox, const vector<float>& prior_variance,
        const int code_type, const bool variance_encoded_in_target,
        const bool clip_bbox, const vector<float>& bbox,
        vector<float>* decode_bbox) {
    if (0 == code_type) {
        if (variance_encoded_in_target) {
            // variance is encoded in target, we simply need to add the offset
            // predictions.
            decode_bbox->push_back(prior_bbox[0] + bbox[0]);
            decode_bbox->push_back(prior_bbox[1] + bbox[1]);
            decode_bbox->push_back(prior_bbox[2] + bbox[2]);
            decode_bbox->push_back(prior_bbox[3] + bbox[3]);
        } else {
            // variance is encoded in bbox, we need to scale the offset accordingly.
            decode_bbox->push_back(
                    prior_bbox[0]+ prior_variance[0] * bbox[0]);
            decode_bbox->push_back(
                    prior_bbox[1] + prior_variance[1] * bbox[1]);
            decode_bbox->push_back(
                    prior_bbox[2] + prior_variance[2] * bbox[2]);
            decode_bbox->push_back(
                    prior_bbox[3] + prior_variance[3] * bbox[3]);
        }
    } else if (1 == code_type) {
        float prior_width = prior_bbox[2] - prior_bbox[0];
        //CHECK_GT(prior_width, 0);
        float prior_height = prior_bbox[3] - prior_bbox[1];
        //CHECK_GT(prior_height, 0);
        float prior_center_x = (prior_bbox[0] + prior_bbox[2]) / 2.;
        float prior_center_y = (prior_bbox[1] + prior_bbox[3]) / 2.;

        float decode_bbox_center_x, decode_bbox_center_y;
        float decode_bbox_width, decode_bbox_height;
        if (variance_encoded_in_target) {
            // variance is encoded in target, we simply need to retore the offset
            // predictions.
            decode_bbox_center_x = bbox[0] * prior_width + prior_center_x;
            decode_bbox_center_y = bbox[1] * prior_height + prior_center_y;
            decode_bbox_width = exp(bbox[2]) * prior_width;
            decode_bbox_height = exp(bbox[3]) * prior_height;
        } else {
            // variance is encoded in bbox, we need to scale the offset accordingly.
            decode_bbox_center_x =
                    prior_variance[0] * bbox[0] * prior_width + prior_center_x;
            decode_bbox_center_y =
                    prior_variance[1] * bbox[1] * prior_height + prior_center_y;
            decode_bbox_width =
                    exp(prior_variance[2] * bbox[2]) * prior_width;
            decode_bbox_height =
                    exp(prior_variance[3] * bbox[3]) * prior_height;
        }

        decode_bbox->push_back(decode_bbox_center_x - decode_bbox_width / 2.);
        decode_bbox->push_back(decode_bbox_center_y - decode_bbox_height / 2.);
        decode_bbox->push_back(decode_bbox_center_x + decode_bbox_width / 2.);
        decode_bbox->push_back(decode_bbox_center_y + decode_bbox_height / 2.);
    } else if (2 == code_type) {
        float prior_width = prior_bbox[2] - prior_bbox[0];
        //CHECK_GT(prior_width, 0);
        float prior_height = prior_bbox[3] - prior_bbox[1];
        //CHECK_GT(prior_height, 0);
        if (variance_encoded_in_target) {
            // variance is encoded in target, we simply need to add the offset
            // predictions.
            decode_bbox->push_back(prior_bbox[0] + bbox[0] * prior_width);
            decode_bbox->push_back(prior_bbox[1] + bbox[1] * prior_height);
            decode_bbox->push_back(prior_bbox[2] + bbox[2] * prior_width);
            decode_bbox->push_back(prior_bbox[3] + bbox[3] * prior_height);
        } else {
            // variance is encoded in bbox, we need to scale the offset accordingly.
            decode_bbox->push_back(
                    prior_bbox[0] + prior_variance[0] * bbox[0] * prior_width);
            decode_bbox->push_back(
                    prior_bbox[1] + prior_variance[1] * bbox[1] * prior_height);
            decode_bbox->push_back(
                    prior_bbox[2] + prior_variance[2] * bbox[2] * prior_width);
            decode_bbox->push_back(
                    prior_bbox[3] + prior_variance[3] * bbox[3] * prior_height);
        }
    } else {
        std::cout<< "Unknown LocLossType."<<std::endl;
    }
    //clip_bbox = false, 所以没实现
    /*if (clip_bbox) {
        ClipBBox(*decode_bbox, decode_bbox);
    }*/
}


void DecodeBBoxes(
        const vector<vector<float> >& prior_bboxes,
        const vector<vector<float> >& prior_variances,
        const int code_type, const bool variance_encoded_in_target,
        const bool clip_bbox, const vector<vector<float> >& bboxes,
        vector<vector<float> >* decode_bboxes) {
    //CHECK_EQ(prior_bboxes.size(), prior_variances.size());
    //CHECK_EQ(prior_bboxes.size(), bboxes.size());
    int num_bboxes = prior_bboxes.size();
    
    for (int i = 0; i < num_bboxes; ++i) {
        vector<float> decode_bbox;
        DecodeBBox(prior_bboxes[i], prior_variances[i], code_type,
                   variance_encoded_in_target, clip_bbox, bboxes[i], &decode_bbox);
        decode_bboxes->push_back(decode_bbox);
    }
}

//
void ConfData(const float* data, const int num_classes, const int num_prior, float* new_data) {
    int idx = 0;
    for (int c = 0; c < num_classes; ++c) {
        for (int p = 0; p < num_prior; ++p) {
            new_data[idx] = data[p*num_classes + c];
            idx++;  
        }
    }
    //softmax
    for (int p = 0; p < num_prior; ++p) {
        int sum = 0;
        float _max = new_data[p];//new_data[0*num_prior + p]
	for (int c = 1; c < num_classes; ++c) {
            _max = std::max(_max, new_data[c*num_prior + p]);
        }
        for (int c = 0; c < num_classes; ++c) {
            sum += exp(new_data[c*num_prior + p]-_max);
        }
	for (int j = 0; j < num_classes; ++j) {
            new_data[j*num_prior + p] =  exp(new_data[j*num_prior + p]-_max)/sum; 
        }
    }

}

template <typename Dtype>
void DecodeBBoxes_2(const Dtype* loc_data, const Dtype* prior_data,
                        const int code_type, const bool variance_encoded_in_target,
                        const int num_priors, const bool share_location,
                        const int num_loc_classes, const int background_label_id,
                        const bool clip_bbox, Dtype* bbox_data) {

    if(code_type == 0){
        for(int p = 0; p < num_priors; p++) {
            if (variance_encoded_in_target) {
                for (int i = 0; i < 4; i++) {
                    bbox_data[4 * p + i] = prior_data[4 * p + i] + loc_data[4 * p + i];
                }
            } else {
                for (int i = 0; i < 4; i++) {
                bbox_data[4 * p + i] = prior_data[4 * p + i] + prior_data[4 * num_priors + 4 * p + i] + loc_data[4 * p + i];
                }
            }
        }
    }else if(code_type == 1){
        for(int p = 0; p < num_priors; p++) {
            float prior_width = prior_data[4 * p + 2] - prior_data[4 * p + 0];
            float prior_height = prior_data[4 * p + 3] - prior_data[4 * p + 1];
            float prior_center_x = (prior_data[4 * p + 0] + prior_data[4 * p + 2]) / 2.;
            float prior_center_y = (prior_data[4 * p + 1] + prior_data[4 * p + 3]) / 2.;
            float decode_bbox_center_x, decode_bbox_center_y;
            float decode_bbox_width, decode_bbox_height;;
            if (variance_encoded_in_target) {
                decode_bbox_center_x = loc_data[4 * p + 0] * prior_width + prior_center_x;
                decode_bbox_center_y = loc_data[4 * p + 1] * prior_height + prior_center_y;
                decode_bbox_width = exp(loc_data[4 * p + 2]) * prior_width;
                decode_bbox_height = exp(loc_data[4 * p + 3]) * prior_height;
            }else{
                decode_bbox_center_x = prior_data[4 * num_priors + 4 * p + 0] * loc_data[4 * p + 0] * prior_width + prior_center_x;
                decode_bbox_center_y = prior_data[4 * num_priors + 4 * p + 1] * loc_data[4 * p + 1] * prior_height + prior_center_y;
                decode_bbox_width = exp(prior_data[4 * num_priors + 4 * p + 2] * loc_data[4 * p + 2]) * prior_width;
                decode_bbox_height = exp(prior_data[4 * num_priors + 4 * p + 3] * loc_data[4 * p + 3]) * prior_height;
            }
            bbox_data[4 * p + 0] = (decode_bbox_center_x - decode_bbox_width / 2.);
            bbox_data[4 * p + 1] = (decode_bbox_center_y - decode_bbox_height / 2.);
            bbox_data[4 * p + 2] = (decode_bbox_center_x + decode_bbox_width / 2.);
            bbox_data[4 * p + 3] = (decode_bbox_center_y + decode_bbox_height / 2.);
        }

    }else if(code_type == 2){
        for(int p = 0; p < num_priors; p++) {
            float prior_width = prior_data[4 * p + 2] - prior_data[4 * p + 0];
            float prior_height = prior_data[4 * p + 3] - prior_data[4 * p + 1];

            if (variance_encoded_in_target) {
                bbox_data[4 * p + 0] = prior_data[4 * p + 0] + loc_data[4 * p + 0] * prior_width;
                bbox_data[4 * p + 1] = prior_data[4 * p + 1] + loc_data[4 * p + 1] * prior_height;
                bbox_data[4 * p + 2] = exp(prior_data[4 * p + 2]) + loc_data[4 * p + 2] * prior_width;
                bbox_data[4 * p + 3] = exp(prior_data[4 * p + 3]) + loc_data[4 * p + 3] * prior_height;
            }else {
                bbox_data[4 * p + 0] = prior_data[4 * p + 0] +
                                       prior_data[4 * num_priors + 4 * p + 0] * loc_data[4 * p + 0] * prior_width;
                bbox_data[4 * p + 1] = prior_data[4 * p + 1] +
                                       prior_data[4 * num_priors + 4 * p + 1] * loc_data[4 * p + 1] * prior_height;
                bbox_data[4 * p + 2] = prior_data[4 * p + 2] +
                                       prior_data[4 * num_priors + 4 * p + 2] * loc_data[4 * p + 2] * prior_width;
                bbox_data[4 * p + 3] = prior_data[4 * p + 3] +
                                       prior_data[4 * num_priors + 4 * p + 3] * loc_data[4 * p + 3] * prior_height;
            }
        }

    }else{
        std::cout << "Unknown LocLossType." << std::endl;
    }
}


template <typename Dtype>
Dtype BBoxSize(const Dtype* bbox, const bool normalized = true) {
    if (bbox[2] < bbox[0] || bbox[3] < bbox[1]) {
        // If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
        return Dtype(0.);
    } else {
        const Dtype width = bbox[2] - bbox[0];
        const Dtype height = bbox[3] - bbox[1];
        if (normalized) {
            return width * height;
        } else {
            // If bbox is not within range [0, 1].
            return (width + 1) * (height + 1);
        }
    }
}

template <typename Dtype>
Dtype JaccardOverlap(const Dtype* bbox1, const Dtype* bbox2) {
    if (bbox2[0] > bbox1[2] || bbox2[2] < bbox1[0] ||
        bbox2[1] > bbox1[3] || bbox2[3] < bbox1[1]) {
        return Dtype(0.);
    } else {
        const Dtype inter_xmin = std::max(bbox1[0], bbox2[0]);
        const Dtype inter_ymin = std::max(bbox1[1], bbox2[1]);
        const Dtype inter_xmax = std::min(bbox1[2], bbox2[2]);
        const Dtype inter_ymax = std::min(bbox1[3], bbox2[3]);

        const Dtype inter_width = inter_xmax - inter_xmin;
        const Dtype inter_height = inter_ymax - inter_ymin;
        const Dtype inter_size = inter_width * inter_height;

        const Dtype bbox1_size = BBoxSize(bbox1);
        const Dtype bbox2_size = BBoxSize(bbox2);

        return inter_size / (bbox1_size + bbox2_size - inter_size);
    }
}

template <typename T>
bool SortScorePairDescend(const pair<float, T>& pair1,
                          const pair<float, T>& pair2) {
    return pair1.first > pair2.first;
}

template <typename Dtype>
void GetMaxScoreIndex(const Dtype* scores, const int num, const float threshold,
                      const int top_k, vector<pair<Dtype, int> >* score_index_vec) {
    // Generate index score pairs.
    for (int i = 0; i < num; ++i) {
        if (scores[i] > threshold) {
            score_index_vec->push_back(std::make_pair(scores[i], i));
        }
    }

    // Sort the score pair according to the scores in descending order
    std::sort(score_index_vec->begin(), score_index_vec->end(),
              SortScorePairDescend<int>);

    // Keep top_k scores if needed.
    if (top_k > -1 && top_k < score_index_vec->size()) {
        score_index_vec->resize(top_k);
    }
}

template <typename Dtype>
void ApplyNMSFast(const Dtype* bboxes, const Dtype* scores, const int num,
                  const float score_threshold, const float nms_threshold,
                  const float eta, const int top_k, vector<int>* indices) {
    // Get top_k scores (with corresponding indices).
    vector<pair<Dtype, int> > score_index_vec;
    //float n1 = cv::getTickCount();
    GetMaxScoreIndex(scores, num, score_threshold, top_k, &score_index_vec);
    // n1 = (cv::getTickCount()-n1) / cv::getTickFrequency();
    //printf("======n==1 Forward_DetectionOutputLayer time is %f \n", n1);

    // Do nms.
    float adaptive_threshold = nms_threshold;
    indices->clear();
    //float n2 = cv::getTickCount();
    std::cout<<"======n==n" <<score_index_vec.size()<<std::endl;
    while (score_index_vec.size() != 0) {
        const int idx = score_index_vec.front().second;
        bool keep = true;
        for (int k = 0; k < indices->size(); ++k) {
            if (keep) {
                const int kept_idx = (*indices)[k];
                float overlap = JaccardOverlap(bboxes + idx * 4, bboxes + kept_idx * 4);
                keep = overlap <= adaptive_threshold;
            } else {
                break;
            }
        }
        if (keep) {
            indices->push_back(idx);
        }
        score_index_vec.erase(score_index_vec.begin());
        if (keep && eta < 1 && adaptive_threshold > 0.5) {
            adaptive_threshold *= eta;
        }
    }
    //n2 = (cv::getTickCount()-n2) / cv::getTickFrequency();
    //printf("======n==2 Forward_DetectionOutputLayer time is %f \n", n2);
}


void Forward_DetectionOutputLayer(float* loc_data, float* conf_data, float* prior_data, int num_priors_, int num_classes_, vector<vector<float> >* detecions) {
    // Retrieve all location predictions.
    /*vector<vector<float>> all_loc_preds;
    GetLocPredictions(loc_data, num_priors_, num_loc_classes_, &all_loc_preds);
    // Retrieve all confidences.
    vector <vector<float>> all_conf_scores;
    GetConfidenceScores(conf_data, num_priors_, num_classes_,
                        &all_conf_scores);
    // Retrieve all prior bboxes.
    vector<vector<float>> prior_bboxes;
    vector<vector<float>> prior_variances;
    GetPriorBBoxes(prior_data, num_priors_, &prior_bboxes, &prior_variances);
    // Decode all loc predictions to bboxes.
    vector<vector<float>> all_decode_bboxes;
    //const bool clip_bbox = false;
    DecodeBBoxes(prior_bboxes, prior_variances, code_type_,
                  variance_encoded_in_target_, clip_bbox, all_loc_preds,
                 &all_decode_bboxes);*/


    int num_kept = 0;
    vector<map<int, vector<int> > > all_indices;

    map<int , vector<int>> indices;
    int num_det = 0;
    const int conf_idx = num_classes_ * num_priors_;
    const bool share_location_ = true;
    const int num_loc_classes = 1;
    int background_label_id_ = 0;
    float confidence_threshold_ = 0.1;
    float nms_threshold_ = 0.45;
    float eta_ = 1.0;//默认1.0
    int top_k_ = 400;
    int keep_top_k_ = 200;

    const int code_type = 1;//center
    const bool variance_encoded_in_target = false;//default
    const bool clip_bbox = false;

    float* decode_bboxes = new float[4 * num_priors_];
    float t = cv::getTickCount();
    DecodeBBoxes_2<float>(loc_data, prior_data, code_type, variance_encoded_in_target, num_priors_, share_location_, num_loc_classes,background_label_id_, clip_bbox, decode_bboxes);
    t = (cv::getTickCount()-t) / cv::getTickFrequency();
    printf("======1 Forward_DetectionOutputLayer time is %f \n", t);
    float* new_conf_data = new float[num_priors_ * num_classes_];
    float t1 = cv::getTickCount();
    ConfData(conf_data, num_classes_, num_priors_, new_conf_data);
    t1 = (cv::getTickCount()-t1) / cv::getTickFrequency();
    printf("======2 Forward_DetectionOutputLayer time is %f \n", t1);

    float t2 = cv::getTickCount();
    for(int c = 0; c < num_classes_; c++){
        if(c == background_label_id_){
            continue;
        }
        float* cur_conf_data = new_conf_data + c * num_priors_;
        //float* cur_bbox_data = all_decode_bboxes
        float tt = cv::getTickCount();
        ApplyNMSFast<float>(decode_bboxes, cur_conf_data, num_priors_,
                            confidence_threshold_, nms_threshold_, eta_, top_k_, &(indices[c]));
         tt = (cv::getTickCount()-tt) / cv::getTickFrequency();
         std::cout<<"===nms==="<<c<<"==nms=="<<std::endl;
         printf("======nms Forward_DetectionOutputLayer time is %f \n", tt);
        num_det += indices[c].size();
    }
    t2 = (cv::getTickCount()-t2) / cv::getTickFrequency();
    printf("======3 Forward_DetectionOutputLayer time is %f \n", t2);

    float t3 = cv::getTickCount();
    if(keep_top_k_ > -1 && num_det > keep_top_k_){
        vector<pair<float, pair<int, int> > > score_index_pairs;
        for(map<int, vector<int> >::iterator it = indices.begin(); it != indices.end(); ++it){
            int label = it->first;
            const vector<int>& label_indices = it->second;
            for(int j = 0; j < label_indices.size(); ++j){
                int idx = label_indices[j];
                float score = new_conf_data[label * num_priors_ + idx];
                score_index_pairs.push_back(std::make_pair(score, std::make_pair(label, idx)));
            }
        }
        // Keep top k results per image.
        std::sort(score_index_pairs.begin(), score_index_pairs.end(), SortScorePairDescend<pair<int, int> >);
        score_index_pairs.resize(keep_top_k_);
        // Store the new indices.
        map<int, vector<int> > new_indices;
        for(int j = 0; j < score_index_pairs.size(); ++j){
            int label = score_index_pairs[j].second.first;
            int idx = score_index_pairs[j].second.second;
            new_indices[label].push_back(idx);
        }
        all_indices.push_back(new_indices);
        num_kept += keep_top_k_;
    }else{
        all_indices.push_back(indices);
        num_kept += num_det;
    }
    if(num_kept == 0){
        printf("Couldn't find any detections");
    }else{
        for(map<int, vector<int> >::iterator it = all_indices[0].begin(); it != all_indices[0].end(); ++it){
            int label = it->first;
            vector<int>& _indices = it->second;
            const float* _cur_conf_data = new_conf_data + label * num_priors_;
	    
            for(int j = 0; j < _indices.size(); ++j){
                int idx = _indices[j];
                vector<float> detect;
                for(int k = 0; k < 4; ++k){
                    detect.push_back(decode_bboxes[idx * 4 + k]);
                }
                detect.push_back(_cur_conf_data[idx]);
                detect.push_back(label);
                detecions->push_back(detect);
            }
        }
    }
    t3 = (cv::getTickCount()-t3) / cv::getTickFrequency();
    printf("======4 Forward_DetectionOutputLayer time is %f \n", t3);

    delete[] decode_bboxes;
    delete[] new_conf_data;
}
