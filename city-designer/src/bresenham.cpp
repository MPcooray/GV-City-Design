// bresenham.cpp
#include "bresenham.h"
#include <cmath>

std::vector<std::pair<int,int>> bresenham(int x0,int y0,int x1,int y1){
    std::vector<std::pair<int,int>> pts;
    int dx = std::abs(x1 - x0);
    int dy = std::abs(y1 - y0);
    int sx = x0 < x1 ? 1 : -1;
    int sy = y0 < y1 ? 1 : -1;
    int err = (dx > dy ? dx : -dy) / 2;
    int x = x0, y = y0;

    while(true){
        pts.emplace_back(x,y);
        if(x == x1 && y == y1) break;
        int e2 = err;
        if(e2 > -dx){ err -= dy; x += sx; }
        if(e2 <  dy){ err += dx; y += sy; }
    }
    return pts;
}
