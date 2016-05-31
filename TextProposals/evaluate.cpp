#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include<cstdlib>
#include<string>
#include<vector>
#include<dirent.h>

#include  "opencv2/highgui.hpp"
#include  "opencv2/imgproc.hpp"
#include  "opencv2/features2d.hpp"

#include "compute_regions.h"

using namespace std;
using namespace cv;

#define SAVE_IMAGE_INFO    1 // Saves in a txt file each image filename and its score
#define SAVE_REGION_INFO       1 // Saves a txt filo for each image with its region BB and their score
#define REGION_SCORE_STRATEGY 1 
/* 0: Regions score by weak classifier
 * 1: Regions score is mean region probability by heat-map
 * */
#define IMAGE_SCORE_STRATEGY 0 
/* 0: Image score is max region score
 * 1: 
 * */


int main()
{  
  
  string directory("/home/imatge/caffe-master/data/coco-text/");
  DIR *dpdf;
  struct dirent *epdf;
  string filename;
  Mat img;
  Mat heatmap;

  //Build evaluation file  
  ofstream imevfile;
  string evaluation_filename(directory + "images_evaluation/images_scores.txt");
  imevfile.open (evaluation_filename.c_str());
  
  dpdf = opendir((directory + "miniTrain/").c_str());
  if (dpdf != NULL){
    while (epdf = readdir(dpdf)){
	filename = epdf->d_name;
	
	if (filename.length() < 4){
	  continue;}  
	 if (filename.compare(filename.length()-4,filename.length(),".jpg")){
	  continue;}  

	img = imread(directory + "miniTrain/" + filename);
	cout << "\nImage read: " << filename;
	ComputeRegions segmentator;
	vector<HCluster> regions_info;
	segmentator(img, regions_info);
	
	if(SAVE_REGION_INFO){
	  // Create file to save regions data  
	  ofstream myfile;
	  string evaluation_filename(directory + "regions_evaluation/" + filename.substr(0, filename.length() - 3) + "txt");	
	  myfile.open (evaluation_filename.c_str());
	  
	  switch(REGION_SCORE_STRATEGY)
	  {
	    case 0: //Region scores from weak classifier are already computed
	      break;
	      
	    case 1: //Compute region score from heatmap
	      heatmap = imread(directory + "heatmaps/" + filename.substr(0, filename.length() - 4) + ".png", CV_LOAD_IMAGE_GRAYSCALE);
	      for (int k=0; k<regions_info.size(); k++)
	      {
		Mat regionHeatMap;
		regionHeatMap = heatmap(regions_info[k].rect);		
		regions_info[k].probability = sum(regionHeatMap)[0] / (regionHeatMap.total() * 255);
	      }
	      break;
	  } 
	     //Write regions scores to file
	      for (int k=0; k<regions_info.size(); k++)
	      {	
		  int ml = 1;
		  /*
		  if (c>=num_channels) ml=2;// update sizes for smaller pyramid lvls
		  if (c>=2*num_channels) ml=4;// update sizes for smaller pyramid lvls		    
		  */
		  myfile << regions_info[k].rect.x*ml << " " << regions_info[k].rect.y*ml << " "
			<< regions_info[k].rect.width*ml << " " << regions_info[k].rect.height*ml << " "
			<< (float)regions_info[k].probability << endl;		
	      }
	      myfile.close();
	      cout << "\nRegion Segmentation info. saved";
	}
	
	if(SAVE_IMAGE_INFO){
	// Compute score per image
	  float image_score = 0;
	  
	  switch(IMAGE_SCORE_STRATEGY)
	  {
	    case 0:
	      for (int k=0; k<regions_info.size(); k++)
		{	
		  if ((float)regions_info[k].probability > image_score){
		      image_score = (float)regions_info[k].probability;
		  }
		}
	      break;
	     
	    case 1:	      
	      break;	      
	  }
	  
	// Save the score of every image to a file	
	imevfile << filename.substr(15, 12) << " " << image_score << endl;
	cout << "\nImage score saved";	
	}
     }
  }
  imevfile.close();
}






