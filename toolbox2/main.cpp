#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

Mat src; Mat src_gray;
int thresh = 100;
int max_thresh = 255;
RNG rng(12345);

int morph_elem = 0;
int morph_size = 0;
int morph_operator = 0;

/// Function header
void thresh_callback(int, void* );

/** @function main */
int main( int argc, char** argv )
{
    /// Load source image
    src = imread("test.bmp", 1 );
    
    /// Convert image to gray
    cvtColor( src, src_gray, CV_BGR2GRAY );

    // ehhh, 'close' it with a morph size of 2?
    
    morph_size = 2;
    
    Mat element = getStructuringElement( 2, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
    
    morphologyEx( src_gray, src_gray, 0, element );

    // blur image 3x3 pixels
    
    blur( src_gray, src_gray, Size(3,3) );
    
    // make variables
    
    Mat canny_output;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    
    /// Detect edges using canny
    Canny( src_gray, canny_output, thresh, thresh*2, 3 );
    
    /// Find contours
    findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    
    // Make somewhere to put the output
    Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
    
    
    // Sort contours by size
    sort(contours.begin(), contours.end(), [](const vector<Point>& c1, const vector<Point>& c2){
        return contourArea(c1, false) < contourArea(c2, false);
    });
    
    // Loop over all the contours, ending with the largest
    int i=0;
    
    for (i = 0; i<contours.size(); i++)
    {

        // Pick a random colour
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        
        // Draw the contours filled
        drawContours( drawing, contours, i, color, -1, 8, hierarchy, 0, Point() );
        
    }
    
    
    // Make windows and show output
    
    char* source_window = "Source";
    namedWindow( source_window, CV_WINDOW_NORMAL );
    imshow( source_window, src_gray );
    namedWindow( "Contours", CV_WINDOW_NORMAL );
    imshow( "Contours", drawing );
    
    
    waitKey(0);
    return(0);
}


