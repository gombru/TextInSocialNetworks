#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include<cstdlib>
#include<string>
#include<vector>
#include<dirent.h>
#include <time.h>

#include  "opencv2/highgui.hpp"
#include  "opencv2/imgproc.hpp"
#include  "opencv2/features2d.hpp"

#include "compute_regions.h"

using namespace std;
using namespace cv;

#define SAVE_IMAGE_INFO    1 // Saves in a txt file each image filename and its score
#define SAVE_REGION_INFO       1 // Saves a txt filo for each image with its region BB and their score



int main()
{  
  
  cout << "\nStarting ...";
  string directory("/home/imatge/caffe-master/data/coco-text/");
  DIR *dpdf;
  struct dirent *epdf;
  string filename;
  Mat img;
  Mat heatmap;
  int count = 0;
  int msec = 0;
  
  
  //Build evaluation file  
  ofstream imevfile;
  string evaluation_filename(directory + "images_evaluation/images_scores.txt");
  imevfile.open (evaluation_filename.c_str());
  
  cout << "\nOpening dir...";
  dpdf = opendir((directory + "val-withoutIllegible/").c_str());
  clock_t start = clock(), diff;
  

  if (dpdf != NULL){
    while (epdf = readdir(dpdf)){
	filename = epdf->d_name;
	
	if (filename.length() < 4){
	  continue;}  
	 if (filename.compare(filename.length()-4,filename.length(),".jpg")){
	  continue;}  
	
	clock_t start_i = clock(), diff;
	
	count ++;
	img = imread(directory + "val-withoutIllegible/" + filename);
	cout << "\nImage read: " << filename;
	
	
	ComputeRegions segmentator;
	vector<HCluster> regions_info;
	segmentator(img, regions_info);
	/*
	  diff = clock() - start_i;
	  msec = diff * 1000 / CLOCKS_PER_SEC;
	  printf("\nSegmentation done in %d seconds %d milliseconds", msec/1000, msec%1000);*/
	
	if(SAVE_REGION_INFO){
	  // Create file to save regions data  
	  ofstream myfile;
	  string evaluation_filename(directory + "regions_evaluation/" + filename.substr(0, filename.length() - 3) + "txt");	
	  myfile.open (evaluation_filename.c_str());
	  

	     //Write regions scores to file
	      for (int k=0; k<regions_info.size(); k++)
	      {	
		  int ml = 1;
		  /*
		  if (c>=num_channels) ml=2;// update sizes for smaller pyramid lvls
		  if (c>=2*num_channels) ml=4;// update sizes for smaller pyramid lvls		    
		  */
		  if ((float)regions_info[k].probability < 0.001){
		      regions_info[k].probability = 0;}
		    
		  myfile << regions_info[k].rect.x*ml << " " << regions_info[k].rect.y*ml << " "
			<< regions_info[k].rect.width*ml << " " << regions_info[k].rect.height*ml << " "
			<< (float)regions_info[k].probability << endl;		
	      }
	      myfile.close();
	      cout << "\nRegion Segmentation info. saved";

	}
	
	/*
	diff = clock() - start_i;
	msec = diff * 1000 / CLOCKS_PER_SEC;
	printf("\nRegions scores done in %d seconds %d milliseconds", msec/1000, msec%1000);*/
	
	
	if(SAVE_IMAGE_INFO){
	// Compute score per image
	  float image_score = 0;
	  float image_score2 = 0;
	  float image_score3 = 0;
	  

	      for (int k=0; k<regions_info.size(); k++)
		{	
		  if ((float)regions_info[k].probability > image_score){
		      image_score = (float)regions_info[k].probability;
		  }
		}

	  
	// Save the score of every image to a file	
	imevfile << filename.substr(15, 12) << " " << image_score << " " << image_score2<< " " << image_score3 << endl;
	cout << "\nImage score saved";	
	}
	
	/*
	diff = clock() - start_i;
	msec = diff * 1000 / CLOCKS_PER_SEC;
	printf("\nImages scores done in %d seconds %d milliseconds", msec/1000, msec%1000);*/
	
     //if (count == 5) {break;}
     
     }
  }
  diff = clock() - start;
  printf("\nNum images: %d", count);
  msec = diff * 1000 / CLOCKS_PER_SEC;
  printf("\nTime taken %d seconds %d milliseconds", msec/1000, msec%1000);
  printf("\nTime taken per image has been %d seconds %d milliseconds", (msec/count)/1000, (msec/count)%1000); 
  cout << "\nDone";	
  imevfile.close();
}






