
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <map>
#include <limits>
#include <math.h>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // compute the average euclidean distances between keypoint matches
    double distance_aver = 0;
    int keyPointsNumber = 0;
    for(auto match : kptMatches)
    {
        int preKeyInd = match.queryIdx;
        int curKeyInd = match.trainIdx;
        // Determine whether the KeyPoint belongs to the boundingBox
        if(boundingBox.roi.contains(kptsCurr[curKeyInd].pt))
        {
            keyPointsNumber++;
            distance_aver += std::sqrt(std::pow((kptsCurr[curKeyInd].pt.x-kptsCurr[preKeyInd].pt.x),2)+std::pow((kptsCurr[curKeyInd].pt.y-kptsCurr[preKeyInd].pt.y),2));
        }
    }
    distance_aver = distance_aver / keyPointsNumber;

    // remove those keypoints that are too far away from the mean
    for(auto match : kptMatches)
    {
        int preKeyInd = match.queryIdx;
        int curKeyInd = match.trainIdx;
        // Determine whether the KeyPoint belongs to the boundingBox
        if(boundingBox.roi.contains(kptsCurr[curKeyInd].pt))
        {
            int distance = std::sqrt(std::pow((kptsCurr[curKeyInd].pt.x-kptsCurr[preKeyInd].pt.x),2)+std::pow((kptsCurr[curKeyInd].pt.y-kptsCurr[preKeyInd].pt.y),2));
            if(distance < distance_aver*1.2 && distance > distance_aver*0.8)
            {
                boundingBox.keypoints.push_back(kptsCurr[curKeyInd]);
                boundingBox.kptMatches.push_back(match);
            }
        }
    }

    // the end
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    // compute camera-based TTC from distance ratios
    //均值滤波器，可能存在错误匹配的情况，均值滤波器误差较大
    // double meanDistRatio = std::accumulate(distRatios.begin(), distRatios.end(), 0.0) / distRatios.size();
    //采用中值滤波器，考虑奇偶数
    std::sort(distRatios.begin(),distRatios.end());
    auto middle_index = std::floor(distRatios.size()/2.0);
    double meanDistRatio = distRatios.size()%2==0 ? (distRatios[middle_index-1]+distRatios[middle_index])/2.0 : distRatios[middle_index];
    double dT = 1 / frameRate;
    TTC = -dT / (1 - meanDistRatio);
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // compute the center of the pointcloud
    // -pre_pointcloud
    int lidar_pre_num = lidarPointsPrev.size();
    int pre_x_aver = 0;
    int pre_y_aver = 0;
    for(int i=0; i<lidar_pre_num; i++)
    {
        pre_x_aver += lidarPointsPrev[i].x;
        pre_y_aver += lidarPointsPrev[i].y;
    }
    pre_x_aver = pre_x_aver / lidar_pre_num;
    pre_y_aver = pre_y_aver / lidar_pre_num;
    // -curr_pointcloud
    int lidar_cur_num = lidarPointsCurr.size();
    int cur_x_aver = 0;
    int cur_y_aver = 0;
    for(int i=0; i<lidar_cur_num; i++)
    {
        cur_x_aver += lidarPointsCurr[i].x;
        cur_y_aver += lidarPointsCurr[i].y;
    }
    cur_x_aver = cur_x_aver / lidar_pre_num;
    cur_y_aver = cur_y_aver / lidar_pre_num;

    // compute the nearest distance of curr-pointcloud
    int cur_x_min = std::numeric_limits<int>::max();
    for(int i=0; i<lidar_cur_num; i++)
    {
        if(lidarPointsCurr[i].x < cur_x_min) cur_x_min = lidarPointsCurr[i].x;
    }

    // compute the ttc
    TTC = cur_x_min/((pre_x_aver-cur_x_aver)*frameRate);
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{

    std::multimap<int, int> map{};
    for(auto match : matches)
    {
        int preKeyInd = match.queryIdx;
        int curKeyInd = match.trainIdx;
        int pre_box_id = -1;
        int cur_box_id = -1;
        // judge which boundingbox the keypoint belongs-preframe
        for(auto bbox : prevFrame.boundingBoxes)
        {
            if(bbox.roi.contains(prevFrame.keypoints[preKeyInd].pt)) pre_box_id = bbox.boxID;
        }
        // judge which boundingbox the keypoint belongs-currentframe
        for(auto bbox : currFrame.boundingBoxes)
        {
            if(bbox.roi.contains(currFrame.keypoints[curKeyInd].pt)) cur_box_id = bbox.boxID;
        }
        //pushback the boxmap
        map.insert({cur_box_id, pre_box_id});
    }

    for(auto bbox : currFrame.boundingBoxes)
    {
        int bbox_id = bbox.boxID;
        int pre_box_num = prevFrame.boundingBoxes.size();
        vector<int> pre_box_count(pre_box_num, 0);
        auto map_pair = map.equal_range(bbox_id);
        for(auto it=map_pair.first; it!=map_pair.second; it++)
        {
            if(it->first != -1) pre_box_count[it->second]++;
        }
        int pre_box_id = std::distance(std::begin(pre_box_count), std::max_element(std::begin(pre_box_count), std::end(pre_box_count)));
        bbBestMatches.insert({bbox_id, pre_box_id});
    }
}
