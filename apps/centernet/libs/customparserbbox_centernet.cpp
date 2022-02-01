#include <cstring>
#include <iostream>
#include "nvdsinfer_custom_impl.h"
#include <cassert>
#include <cmath>
#include <tuple>
#include <memory>
#include <opencv2/opencv.hpp>

#define CLIP(a, min, max) (MAX(MIN(a, max), min))

/* C-linkage to prevent name-mangling */
extern "C" bool NvDsInferParseCustomTfSSD(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
										  NvDsInferNetworkInfo const &networkInfo,
										  NvDsInferParseDetectionParams const &detectionParams,
										  std::vector<NvDsInferObjectDetectionInfo> &objectList);

/* This is a smaple bbox parsing function for the centernet detection model*/
struct FrcnnParams
{
	int inputHeight;
	int inputWidth;
	int outputClassSize;
	float visualizeThreshold;
	int postNmsTopN;
	int outputBboxSize;
	std::vector<float> classifierRegressorStd;
};

struct PersonInfo
{
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
};

float Logist(float data){ return 1./(1. + exp(-data)); }

/* NMS for centernet */
static void nms(std::vector<PersonInfo> &input, std::vector<PersonInfo> &output, float nmsthreshold)
{
	std::sort(input.begin(), input.end(),
			  [](const PersonInfo &a, const PersonInfo &b) {
				  return a.score > b.score;
			  });

	int box_num = input.size();
	
	std::vector<int> merged(box_num, 0);

	for (int i = 0; i < box_num; i++)
	{
		if (merged[i])
			continue;
		
		output.push_back(input[i]);

		float h0 = input[i].y2 - input[i].y1 + 1;
		float w0 = input[i].x2 - input[i].x1 + 1;

		float area0 = h0 * w0;

		for (int j = i + 1; j < box_num; j++)
		{
			if (merged[j])
				continue;

			float inner_x0 = input[i].x1 > input[j].x1 ? input[i].x1 : input[j].x1; //std::max(input[i].x1, input[j].x1);
			float inner_y0 = input[i].y1 > input[j].y1 ? input[i].y1 : input[j].y1;

			float inner_x1 = input[i].x2 < input[j].x2 ? input[i].x2 : input[j].x2; //bug fixed ,sorry
			float inner_y1 = input[i].y2 < input[j].y2 ? input[i].y2 : input[j].y2;

			float inner_h = inner_y1 - inner_y0 + 1;
			float inner_w = inner_x1 - inner_x0 + 1;

			if (inner_h <= 0 || inner_w <= 0)
				continue;

			float inner_area = inner_h * inner_w;

			float h1 = input[j].y2 - input[j].y1 + 1;
			float w1 = input[j].x2 - input[j].x1 + 1;

			float area1 = h1 * w1;

			float score;

			score = inner_area / (area0 + area1 - inner_area);

			if (score > nmsthreshold)
				merged[j] = 1;
		}
	}
}

/* For CenterNetdetection */
//extern "C"
static std::vector<int> getIds(float *heatmap, int h, int w, float thresh)
{
	std::vector<int> ids;
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			float objProb = Logist(heatmap[i * w + j]);

			if (objProb > thresh)
			{
				float max = -1;
				int max_index = 0;
				int grid_x = (i * w + j) % w;
				int grid_y = (i * w + j) / w % h;

				for (int l = 0; l < 3; ++l)
					for (int m = 0; m < 3; ++m){
						int cur_x = -1 + l + grid_x;
						int cur_y = -1 + m + grid_y;
						int cur_index = cur_y * w + cur_x;
						int valid = (cur_x >= 0 && cur_x < w && cur_y >= 0 && cur_y < h);
						float val = (valid != 0) ? Logist(heatmap[cur_index]) : -1;
						max_index = (val > max) ? cur_index : max_index;
						max = (val > max) ? val : max;
					}
				
				if ((i * w + j) == max_index)
				{
					ids.push_back(i);
					ids.push_back(j);
				}

			}
		}
	}
	return ids;
}

/* customcenternet */
extern "C" bool NvDsInferParseCustomCenterNet(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
												  NvDsInferNetworkInfo const &networkInfo,
												  NvDsInferParseDetectionParams const &detectionParams,
												  std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
	auto layerFinder = [&outputLayersInfo](const std::string &name)
		-> const NvDsInferLayerInfo * {
		for (auto &layer : outputLayersInfo)
		{

			if (layer.dataType == FLOAT &&
				(layer.layerName && name == layer.layerName))
			{
				return &layer;
			}
		}
		return nullptr;
	};
	objectList.clear();
	const NvDsInferLayerInfo *heatmap = layerFinder("hm");
	const NvDsInferLayerInfo *scale = layerFinder("wh");
	const NvDsInferLayerInfo *offset = layerFinder("reg");

	if (!heatmap || !scale || !offset)
	{
		std::cerr << "ERROR: some layers missing or unsupported data types "
				  << "in output tensors" << std::endl;
		return false;
	}

	int fea_h = heatmap->inferDims.d[1]; //#heatmap.size[2];
	int fea_w = heatmap->inferDims.d[2]; //heatmap.size[3];

	int spacial_size = fea_w * fea_h;

	float *heatmap_ = (float *)(heatmap->buffer);

	float *scale0 = (float *)(scale->buffer);
	float *scale1 = scale0 + spacial_size;

	float *offset0 = (float *)(offset->buffer);
	float *offset1 = offset0 + spacial_size;
	
	
	float scoreThresh = 0.3;
	std::vector<int> ids = getIds(heatmap_, fea_h, fea_w, scoreThresh);
	
	int width = networkInfo.width;
	int height = networkInfo.height;
	int d_h = (int)(std::ceil(height / 32) * 32);
	int d_w = (int)(std::ceil(width / 32) * 32);
	
	std::vector<PersonInfo> people_tmp;
	std::vector<PersonInfo> people;

	for (int i = 0; i < ids.size() / 2; i++)
	{
		int id_h = ids[2 * i];
		int id_w = ids[2 * i + 1];
		int index = id_h * fea_w + id_w;

		float s0 = scale0[index];
		float s1 = scale1[index];

		float o0 = offset0[index];
		float o1 = offset1[index];
		float x1 = std::max(float(0.), (id_w + o0) - s0 / 2);
		float y1 = std::max(float(0.), (id_h + o1) - s1 / 2);
		float x2 = std::min((float)d_w, (id_w + o0) + s0 / 2);
		float y2 = std::min((float)d_h, (id_h + o1) + s1 / 2);

		PersonInfo personbox;
		personbox.x1 = x1;
		personbox.y1 = y1;
		personbox.x2 = x2;
		personbox.y2 = y2;
		personbox.score = Logist(heatmap_[index]);

		people_tmp.push_back(personbox);
	}

	const float threshold = 0.3;
	nms(people_tmp, people, threshold);
	for (int k = 0; k < people.size(); k++)
	{
		NvDsInferObjectDetectionInfo object;
		/* Clip object box co-ordinates to network resolution */

		object.left = CLIP(people[k].x1, 0, d_w - 1);
		object.top = CLIP(people[k].y1, 0, d_h - 1);
		object.width = CLIP((people[k].x2 - people[k].x1), 0, d_w - 1);
		object.height = CLIP((people[k].y2 - people[k].y1), 0, d_h - 1);

		if (object.width && object.height)
		{
			object.detectionConfidence = people[k].score;
			object.classId = 0;
			objectList.push_back(object);
		}
	}
	
	return true;
}
/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomCenterNet);