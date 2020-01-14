# SFND_3D_Object_Tracking

## TFP.1 : Match 3D Objects

* Solution: Lines 234 ~ 270 at camFusion_Student.cpp
* first, i get the preFrame boundingbox number vector, then check which number is the biggest.
```
    int p = prevFrame.boundingBoxes.size();
    int c = currFrame.boundingBoxes.size();
    int pt_counts[p][c] = { };
    for (auto it = matches.begin(); it != matches.end() - 1; ++it)     
    {
        cv::KeyPoint query = prevFrame.keypoints[it->queryIdx];
        auto query_pt = cv::Point(query.pt.x, query.pt.y);
        bool query_found = false;

        cv::KeyPoint train = currFrame.keypoints[it->trainIdx];
        auto train_pt = cv::Point(train.pt.x, train.pt.y);
        bool train_found = false;

        std::vector<int> query_id, train_id;
        for (int i = 0; i < p; i++) 
        {
            if (prevFrame.boundingBoxes[i].roi.contains(query_pt))            
             {
                query_found = true;
                query_id.push_back(i);
             }
        }
        for (int i = 0; i < c; i++) 
        {
            if (currFrame.boundingBoxes[i].roi.contains(train_pt))            
            {
                train_found= true;
                train_id.push_back(i);
            }
        }

        if (query_found && train_found) 
        {
            for (auto id_prev: query_id)
                for (auto id_curr: train_id)
                     pt_counts[id_prev][id_curr] += 1;
        }
    }
   
    for (int i = 0; i < p; i++)
    {  
         int max_count = 0;
         int id_max = 0;
         for (int j = 0; j < c; j++)
             if (pt_counts[i][j] > max_count)
             {  
                  max_count = pt_counts[i][j];
                  id_max = j;
             }
          bbBestMatches[i] = id_max;
    } 
```


## FP.2 : Compute Lidar-based TTC

* Solution: Lines 204 ~ 230 at camFusion_Student.cpp
* The shape of the point cloud objects will be very different each frame,so we should use the middle value instead of the mean value.
```
    double pre_x = getMiddle_x_dimen(lidarPointsPrev);
    double curr_x = getMiddle_x_dimen(lidarPointsCurr);
    TTC = curr_x/((pre_x-curr_x)*frameRate);
```

```
double getMiddle_x_dimen(std::vector<LidarPoint> &lidarPoints)
{
    vector<double> x;
    int number = lidarPoints.size();
    for(int i=0; i<number; i++)
    {
        x.push_back(lidarPoints[i].x);
    }
    std::sort(x.begin(),x.end());
    auto middle_index = std::floor(x.size()/2.0);
    double middle_value = x.size()%2==0 ? (x[middle_index-1]+x[middle_index])/2.0 : x[middle_index];
    return middle_value;
}
```


## FP.3 : Associate Keypoint Correspondences with Bounding Boxes

* Solution: Lines 139 ~ 143 at camFusion_Student.cpp
```
for (cv::DMatch match : kptMatches) {
    if (boundingBox.roi.contains(kptsCurr[match.trainIdx].pt)) {
        boundingBox.kptMatches.push_back(match);
    }
}
```


## FP.4 : Compute Camera-based TTC

* Solution: Lines 149 ~ 201 at camFusion_Student.cpp
```
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
    std::sort(distRatios.begin(),distRatios.end());
    auto middle_index = std::floor(distRatios.size()/2.0);
    double meanDistRatio = distRatios.size()%2==0 ? (distRatios[middle_index-1]+distRatios[middle_index])/2.0 : distRatios[middle_index];
    double dT = 1 / frameRate;
    TTC = -dT / (1 - meanDistRatio);
```


## FP.5 : Performance Evaluation 1-the Lidar-based TTC estimation

* The Lidar-based TTC estimation are almost correct.
* TTC from Lidar sometimes is not correct because of lidar point clouds on vehicle ahead is unstable. Maybe it's because of the material's reflectivity.
* The abnormal pictures can be seen in 
## FP.6 : Performance Evaluation 2-the camera-based TTC estimation

* The top3 detector / descriptor combinations:
    * SHITOMASI+ORB
    * SHITOMASI+SIFT
    * SHITOMASI+FREAK
* The different TTCs can be seen in the file ttc_dif.csv.
