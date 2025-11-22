#include "circle.h"

static void add8(std::vector<std::pair<int,int>>& v,int cx,int cy,int x,int y){
    v.emplace_back(cx + x, cy + y);
    v.emplace_back(cx - x, cy + y);
    v.emplace_back(cx + x, cy - y);
    v.emplace_back(cx - x, cy - y);
    v.emplace_back(cx + y, cy + x);
    v.emplace_back(cx - y, cy + x);
    v.emplace_back(cx + y, cy - x);
    v.emplace_back(cx - y, cy - x);
}

std::vector<std::pair<int,int>> midpoint_circle(int cx,int cy,int r){
    std::vector<std::pair<int,int>> pts;
    int x = 0, y = r;
    int p = 1 - r;
    add8(pts,cx,cy,x,y);
    while(x < y){
        x++;
        if(p < 0) p += 2*x + 1;
        else { y--; p += 2*(x - y) + 1; }
        add8(pts,cx,cy,x,y);
    }
    return pts;
}
