// bresenham.h
#pragma once
#include <vector>
#include <utility>

// returns vector of (x,y) pixel coordinates along the integer Bresenham line
// from (x0,y0) to (x1,y1)
std::vector<std::pair<int,int>> bresenham(int x0,int y0,int x1,int y1);
