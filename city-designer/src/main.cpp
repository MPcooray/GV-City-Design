// src/main.cpp
// Full updated main.cpp (with fixes for building placement & 2D markers)
// NOTE: removed 'glad' usage per your request — using system OpenGL headers.

#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <string>
#include <algorithm>

#ifdef __APPLE__
  #define GL_SILENCE_DEPRECATION
  #include <OpenGL/gl3.h>
  #include <OpenGL/gl3ext.h>
#else
  // If you're on Linux/Windows and not using GLAD, ensure your build provides a loader
  // (GLEW/OS-specific). This includes a generic header fallback; replace if you use GLEW.
  #include <GL/gl.h>
#endif

#include <GLFW/glfw3.h>

#include "bresenham.h"
#include "circle.h"
#include "shader.h"

// stb_image for texture loading
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// GLM (header-only)
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// Window / map sizes
const int WIN_W = 1000;
const int WIN_H = 700;
const int MAP_W = 256;
const int MAP_H = 256;

// Grid spacing in pixels on the generated map
int CELL = 32;

// GUI / CLI options (used by generator)
enum RoadPattern { ROAD_GRID = 0, ROAD_RADIAL = 1, ROAD_RANDOM = 2 };
enum SkylineMode { SKY_LOW = 0, SKY_MID = 1, SKY_MIX = 2, SKY_SKY = 3 };

struct Theme { glm::vec3 buildingColor; unsigned char roadR,roadG,roadB; unsigned char parkR,parkG,parkB; };
static std::vector<Theme> g_themes;
static int g_road_pattern = ROAD_GRID;
static int g_skyline_mode = SKY_MID;
static int g_theme_index = 0;
static int g_park_radius = 0;

static void initThemes(){
    if(!g_themes.empty()) return;
    g_themes.push_back({ glm::vec3(0.85f,0.6f,0.6f), 20,20,20, 34,139,34 }); // Default
    g_themes.push_back({ glm::vec3(0.7f,0.3f,0.3f), 40,20,20, 60,160,60 }); // Brick
    g_themes.push_back({ glm::vec3(0.6f,0.7f,0.85f), 25,25,30, 50,180,80 }); // Modern
    g_themes.push_back({ glm::vec3(0.9f,0.8f,0.6f), 30,30,30, 80,200,120 }); // Pastel
}

enum Tile { EMPTY=0, ROAD=1, PARK=2, BUILDING=3 };
static std::vector<int> tileMap; // MAP_W * MAP_H
// last-generation stats (set by generate_map_custom)
static int g_last_available_cells = 0;
static int g_last_intersections = 0;
// how many roundabouts were actually placed during last generation
static int g_last_placed_roundabouts = 0;

// Texture IDs for buildings and roads
static GLuint g_buildingTexture = 0;
static GLuint g_roadTexture = 0;

// Texture loading helper
GLuint loadTexture(const char* path) {
    int w, h, channels;
    unsigned char* data = stbi_load(path, &w, &h, &channels, 3); // force RGB
    
    // If failed, try parent directory (for when running from build/)
    if (!data) {
        std::string parentPath = std::string("../") + path;
        data = stbi_load(parentPath.c_str(), &w, &h, &channels, 3);
        if (data) {
            std::cout << "Loaded texture: " << parentPath << " (" << w << "x" << h << ")\n";
        }
    } else {
        std::cout << "Loaded texture: " << path << " (" << w << "x" << h << ")\n";
    }
    
    if (!data) {
        std::cerr << "Failed to load texture: " << path << " (tried ../" << path << " too)\n";
        return 0;
    }
    
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);
    
    stbi_image_free(data);
    return tex;
}

inline int tileIndex(int x,int y){ return y*MAP_W + x; }

// Simple image container (RGB)
struct Image {
    int w,h;
    std::vector<unsigned char> data; // RGB
    Image(int W,int H): w(W), h(H), data(W*H*3, 200) {}
    void setPixel(int x,int y, unsigned char r,unsigned char g,unsigned char b){
        if(x<0||x>=w||y<0||y>=h) return;
        int idx = (y*w + x)*3;
        data[idx]=r; data[idx+1]=g; data[idx+2]=b;
    }
};

// helper: set pixel and tile type
void setPixelAndTile(Image &img,int x,int y, unsigned char r,unsigned char g,unsigned char b, Tile t){
    if(x<0||x>=img.w||y<0||y>=img.h) return;
    img.setPixel(x,y,r,g,b);
    tileMap[tileIndex(x,y)] = (int)t;
}

// draw thick road around bresenham points
void draw_thick_road(Image &img,int x0,int y0,int x1,int y1,int halfWidth){
    auto pts = bresenham(x0,y0,x1,y1);
    for(auto &p: pts){
        int cx = p.first, cy = p.second;
        for(int dy=-halfWidth; dy<=halfWidth; ++dy){
            for(int dx=-halfWidth; dx<=halfWidth; ++dx){
                initThemes();
                Theme &th = g_themes[std::clamp(g_theme_index, 0, (int)g_themes.size()-1)];
                setPixelAndTile(img, cx+dx, cy+dy, th.roadR,th.roadG,th.roadB, ROAD);
            }
        }
    }
}

// fill circle
void fill_circle_tile(Image &img, int cx, int cy, int r, unsigned char rr,unsigned char gg,unsigned char bb, Tile t){
    for(int y=-r;y<=r;++y){
        int yy = cy + y;
        int xsq = r*r - y*y;
        if(xsq < 0) continue;
        int rx = (int)std::floor(std::sqrt((double)xsq));
        for(int x=-rx;x<=rx;++x){
            setPixelAndTile(img, cx+x, yy, rr,gg,bb, t);
        }
    }
}

// ---------- small & clear map generator (two roundabouts + ~6 buildings) ----------
// ---------- clean 2-roundabout + 6-building layout ----------
// Parameterized map generator. Produces roads, roundabouts and building markers
// according to requested counts. The markers are set as BUILDING tiles so the
// existing placement logic (`init_buildings_from_tiles`) will create 3D blocks.
void generate_map_custom(Image &img, int n_buildings, int n_roads, int n_roundabouts){
    tileMap.assign(MAP_W * MAP_H, (int)EMPTY);
    for(int y=0;y<img.h;++y)
        for(int x=0;x<img.w;++x)
            img.setPixel(x,y,230,230,230);

    int roadHalf = 6;
    int CX = img.w / 2;
    int CY = img.h / 2;

    // compute grid dimensions (cells available for buildings)
    int gridW = std::max(1, img.w / CELL);
    int gridH = std::max(1, img.h / CELL);

    // Clamp road counts so they don't exceed grid capacity (must leave at least one cell between edges)
    int vcount = n_roads/2 + (n_roads%2);
    int hcount = n_roads - vcount;
    int maxV = std::max(0, gridW - 1);
    int maxH = std::max(0, gridH - 1);
    if(vcount > maxV){
        std::cout << "Clamping vertical road count from " << vcount << " to " << maxV << " (gridW=" << gridW << ")\n";
        vcount = maxV;
    }
    if(hcount > maxH){
        std::cout << "Clamping horizontal road count from " << hcount << " to " << maxH << " (gridH=" << gridH << ")\n";
        hcount = maxH;
    }

    // Ensure non-negative counts after clamping
    if(vcount <= 0) vcount = 0;
    if(hcount < 0) hcount = 0;

    // Record road positions and intersections so we can place roundabouts
    std::vector<int> vxs;
    std::vector<int> vys;
    std::vector<std::pair<int,int>> intersections;
    if(g_road_pattern == ROAD_GRID){
        for(int i=0;i<vcount;++i){
            float fx = (float)(i+1) * (float)img.w / (float)(vcount+1);
            int ix = std::clamp((int)fx, 0, img.w-1);
            vxs.push_back(ix);
            draw_thick_road(img, ix, 0, ix, img.h-1, roadHalf);
        }
        for(int i=0;i<hcount;++i){
            float fy = (float)(i+1) * (float)img.h / (float)(hcount+1);
            int iy = std::clamp((int)fy, 0, img.h-1);
            vys.push_back(iy);
            draw_thick_road(img, 0, iy, img.w-1, iy, roadHalf);
        }
        // intersections will be detected after all roads are drawn (pixel-level)
    } else if(g_road_pattern == ROAD_RADIAL){
        int CXc = CX; int CYc = CY;
        int spokes = std::max(1, n_roads);
        float maxR = std::min(img.w, img.h) * 0.5f;
        for(int i=0;i<spokes;++i){
            float ang = (float)i / (float)spokes * 2.0f * 3.14159265f;
            int xend = CXc + (int)(cos(ang) * maxR);
            int yend = CYc + (int)(sin(ang) * maxR);
            draw_thick_road(img, CXc, CYc, xend, yend, roadHalf);
        }
        // For radial pattern, the center is the main intersection where all roads meet
        if(spokes >= 2){
            intersections.emplace_back(CXc, CYc);
        }
    } else { // ROAD_RANDOM
        for(int i=0;i<n_roads;++i){
            int side = rand() % 4;
            auto randEdgePoint = [&](int s){
                if(s==0) return std::make_pair(rand()%img.w, 0);
                if(s==1) return std::make_pair(rand()%img.w, img.h-1);
                if(s==2) return std::make_pair(0, rand()%img.h);
                return std::make_pair(img.w-1, rand()%img.h);
            };
            auto p1 = randEdgePoint(side);
            auto p2 = randEdgePoint((side+1 + (rand()%3))%4);
            draw_thick_road(img, p1.first, p1.second, p2.first, p2.second, roadHalf);
        }
        // intersections will be detected after all roads are drawn (pixel-level)
    }
    // For grid road patterns we can deterministically place intersections at the
    // cartesian product of vertical and horizontal road positions. This avoids
    // pixel-level detection ambiguities and ensures roundabouts sit exactly at
    // the grid crossings.
    if(g_road_pattern == ROAD_GRID && !vxs.empty() && !vys.empty()){
        intersections.clear();
        for(auto &vx : vxs){
            for(auto &vy : vys){
                intersections.emplace_back(vx, vy);
            }
        }
    }
    // Detect true pixel-level intersections by checking for road connectivity
    // in multiple directions (2+ directions = intersection candidate).
    // Skip pixel-level detection if we already generated intersections from the
    // grid vxs/vys positions above OR from radial center.
    if(!(g_road_pattern == ROAD_GRID && !vxs.empty() && !vys.empty()) && 
       g_road_pattern != ROAD_RADIAL){
        std::vector<char> seen(img.w * img.h, 0);
        const int clusterR = 5; // cluster nearby intersection pixels into a single intersection
        const int detectR = 8;  // search radius for detecting road continuation (larger for random roads)
        for(int y=0;y<img.h;++y){
            for(int x=0;x<img.w;++x){
                if(tileMap[tileIndex(x,y)] != (int)ROAD) continue;
                if(seen[tileIndex(x,y)]) continue;

                // For random roads, check 8 directions instead of just 4 cardinal
                bool left=false, right=false, up=false, down=false;
                bool upleft=false, upright=false, downleft=false, downright=false;
                
                for(int r=1; r<=detectR; ++r){
                    // Cardinal directions
                    int lx = x - r; if(lx>=0 && tileMap[tileIndex(lx,y)]==(int)ROAD) left = true;
                    int rx = x + r; if(rx<img.w && tileMap[tileIndex(rx,y)]==(int)ROAD) right = true;
                    int uy = y - r; if(uy>=0 && tileMap[tileIndex(x,uy)]==(int)ROAD) up = true;
                    int dy = y + r; if(dy<img.h && tileMap[tileIndex(x,dy)]==(int)ROAD) down = true;
                    
                    // Diagonal directions (important for random road crossings)
                    int ulx = x - r, uly = y - r; 
                    if(ulx>=0 && uly>=0 && tileMap[tileIndex(ulx,uly)]==(int)ROAD) upleft = true;
                    
                    int urx = x + r, ury = y - r; 
                    if(urx<img.w && ury>=0 && tileMap[tileIndex(urx,ury)]==(int)ROAD) upright = true;
                    
                    int dlx = x - r, dly = y + r; 
                    if(dlx>=0 && dly<img.h && tileMap[tileIndex(dlx,dly)]==(int)ROAD) downleft = true;
                    
                    int drx = x + r, dry = y + r; 
                    if(drx<img.w && dry<img.h && tileMap[tileIndex(drx,dry)]==(int)ROAD) downright = true;
                }

                // Count distinct direction groups (if we have roads extending in 2+ directions, it's an intersection)
                int directionGroups = 0;
                if(left || right) directionGroups++;
                if(up || down) directionGroups++;
                if(upleft || downright) directionGroups++;
                if(upright || downleft) directionGroups++;
                
                if(directionGroups >= 2){
                    intersections.emplace_back(x,y);
                    // mark cluster area as seen
                    for(int cy = std::max(0, y - clusterR); cy <= std::min(img.h-1, y + clusterR); ++cy){
                        for(int cx = std::max(0, x - clusterR); cx <= std::min(img.w-1, x + clusterR); ++cx){
                            seen[tileIndex(cx,cy)] = 1;
                        }
                    }
                }
            }
        }
    }

    // Roundabouts: place using the unified intersections vector
    // record requested value so we can report how many were actually placed
    int requested_roundabouts = n_roundabouts;
    int rr = (g_park_radius > 0) ? g_park_radius : (CELL/2 - 2);
    int intersectionsCount = (int)intersections.size();
    
    // For random roads with many detected intersections, filter to keep only the most significant ones
    // (those with the most road pixels nearby, indicating a true crossing rather than edge noise)
    if(g_road_pattern == ROAD_RANDOM && intersectionsCount > n_roundabouts * 3){
        std::cout << "Filtering " << intersectionsCount << " detected intersections to find best candidates...\n";
        std::vector<std::pair<int, std::pair<int,int>>> scored; // (score, (x,y))
        for(auto &pt : intersections){
            int score = 0;
            const int scoreR = 10;
            for(int dy = -scoreR; dy <= scoreR; ++dy){
                for(int dx = -scoreR; dx <= scoreR; ++dx){
                    int nx = pt.first + dx;
                    int ny = pt.second + dy;
                    if(nx >= 0 && nx < img.w && ny >= 0 && ny < img.h){
                        if(tileMap[tileIndex(nx, ny)] == (int)ROAD) score++;
                    }
                }
            }
            scored.emplace_back(score, pt);
        }
        // Sort by score descending
        std::sort(scored.begin(), scored.end(), [](auto &a, auto &b){ return a.first > b.first; });
        // Keep top candidates (enough for requested roundabouts plus some margin)
        int keepCount = std::min((int)scored.size(), std::max(n_roundabouts * 2, 20));
        intersections.clear();
        for(int i = 0; i < keepCount; ++i){
            intersections.push_back(scored[i].second);
        }
        intersectionsCount = (int)intersections.size();
        std::cout << "Kept top " << intersectionsCount << " intersection candidates.\n";
    }
    
    if(n_roundabouts > intersectionsCount){
        std::cout << "Requested " << n_roundabouts << " roundabouts but only " << intersectionsCount << " intersections available; clamping.\n";
        n_roundabouts = intersectionsCount;
    }
    if(n_roundabouts > 0 && !intersections.empty()){
        if((int)intersections.size() <= n_roundabouts){
            initThemes();
            Theme &th = g_themes[std::clamp(g_theme_index, 0, (int)g_themes.size()-1)];
            for(auto &p : intersections){
                fill_circle_tile(img, p.first, p.second, rr, th.parkR, th.parkG, th.parkB, PARK);
            }
        } else {
            std::vector<int> idx(intersections.size());
            for(size_t i=0;i<idx.size();++i) idx[i] = (int)i;
            for(int i=0;i<n_roundabouts;++i){
                int j = i + (rand() % (idx.size() - i));
                std::swap(idx[i], idx[j]);
                auto &p = intersections[idx[i]];
                initThemes();
                Theme &th = g_themes[std::clamp(g_theme_index, 0, (int)g_themes.size()-1)];
                fill_circle_tile(img, p.first, p.second, rr, th.parkR, th.parkG, th.parkB, PARK);
            }
        }
    }

    // how many roundabouts were actually placed (post-clamp)
    g_last_placed_roundabouts = n_roundabouts;

    // After roads and parks are placed, compute available cells for buildings
    std::vector<std::pair<int,int>> availableCells;
    for(int gy=0; gy<gridH; ++gy){
        for(int gx=0; gx<gridW; ++gx){
            int cx = gx * CELL;
            int cy = gy * CELL;
            bool ok = true;
            for(int yy = std::max(0, cy); yy <= std::min(img.h-1, cy + CELL - 1) && ok; ++yy){
                for(int xx = std::max(0, cx); xx <= std::min(img.w-1, cx + CELL - 1); ++xx){
                    int t = tileMap[tileIndex(xx,yy)];
                    if(t == (int)ROAD || t == (int)PARK){ ok = false; break; }
                }
            }
            if(ok){
                int px = gx * CELL + CELL/2;
                int py = gy * CELL + CELL/2;
                availableCells.emplace_back(px, py);
            }
        }
    }

    if((int)availableCells.size() < n_buildings){
        std::cout << "Only " << availableCells.size() << " available cells for buildings; clamping requested " << n_buildings << " -> " << availableCells.size() << "\n";
        n_buildings = (int)availableCells.size();
    }

    // record stats for the caller
    g_last_available_cells = (int)availableCells.size();
    g_last_intersections = intersectionsCount;

    // Helper: place building marker (big 8×8 blue square)
    auto mark_build = [&](int px, int py){
        initThemes();
        Theme &th = g_themes[std::clamp(g_theme_index, 0, (int)g_themes.size()-1)];
        unsigned char br = (unsigned char)std::min(255.0f, th.buildingColor.r * 255.0f);
        unsigned char bg = (unsigned char)std::min(255.0f, th.buildingColor.g * 255.0f);
        unsigned char bb = (unsigned char)std::min(255.0f, th.buildingColor.b * 255.0f);
        for(int dy=-4; dy<=4; ++dy)
            for(int dx=-4; dx<=4; ++dx)
                setPixelAndTile(img, px + dx, py + dy, br,bg,bb, BUILDING);
    };

    // Place buildings into available cells (randomized order)
    std::vector<int> order((int)availableCells.size());
    for(size_t i=0;i<order.size();++i) order[i] = (int)i;
    // shuffle deterministically using rand()
    for(int i=(int)order.size()-1;i>0;--i){ int j = rand() % (i+1); std::swap(order[i], order[j]); }
    int placed = 0;
    for(int oi=0; oi<(int)order.size() && placed < n_buildings; ++oi){
        int idx = order[oi];
        int px = availableCells[idx].first;
        int py = availableCells[idx].second;
        mark_build(px, py);
        placed++;
    }

    // Done: if still not placed all, leave as-is (user asked for many buildings)
}

// write image as simple PPM (ASCII P6) for easy inspection
static bool write_ppm(const Image &img, const std::string &path){
    FILE *f = fopen(path.c_str(), "wb");
    if(!f) return false;
    // P6 header
    fprintf(f, "P6\n%d %d\n255\n", img.w, img.h);
    fwrite(img.data.data(), 1, img.w * img.h * 3, f);
    fclose(f);
    return true;
}

// Fullscreen textured quad shaders (vertex + fragment)
const char* quadVS = R"glsl(
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUV;
out vec2 UV;
void main(){
    gl_Position = vec4(aPos, 0.0, 1.0);
    UV = aUV;
}
)glsl";

const char* quadFS = R"glsl(
#version 330 core
in vec2 UV;
out vec4 FragColor;
uniform sampler2D uTex;
void main(){
    vec3 t = texture(uTex, UV).rgb;
    FragColor = vec4(t,1.0);
}
)glsl";

// Cube vertex/fragment shaders with normals + world-space UV for ground + separate building color
const char* cubeVS = R"glsl(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUV;

uniform mat4 MVP;
uniform mat4 Model;
uniform mat3 normalMatrix;

out vec3 worldPos;
out vec3 vNormal;
out vec2 faceUV;

void main(){
    vec4 wp = Model * vec4(aPos, 1.0);
    worldPos = wp.xyz;
    vNormal = normalize(normalMatrix * aNormal);
    faceUV = aUV;
    gl_Position = MVP * vec4(aPos, 1.0);
}
)glsl";

const char* cubeFS = R"glsl(
#version 330 core
in vec3 worldPos;
in vec3 vNormal;
in vec2 faceUV;
out vec4 FragColor;
uniform sampler2D uTex;
uniform sampler2D uTileMap;
uniform sampler2D uBuildingTex;
uniform sampler2D uRoadTex;
uniform float groundWidth;
uniform float groundDepth;
uniform int isGround;
uniform vec3 lightDir;
uniform vec3 buildingColor;
uniform int useTextures;  // 0=color only, 1=use textures
void main(){
    vec3 color;
    if(isGround == 1){
        float u = (worldPos.x / groundWidth) + 0.5;
        float v = (worldPos.z / groundDepth) + 0.5;
        vec3 groundColor = texture(uTex, vec2(u, v)).rgb;
        
        // If using textures, use tile map to determine where to apply road texture
        if(useTextures == 1) {
            // Sample tile map to get tile type (0=EMPTY, 1=ROAD, 2=PARK, 3=BUILDING)
            float tileType = texture(uTileMap, vec2(u, v)).r * 255.0;
            
            // Sample road texture with tiling for detail
            vec3 roadColor = texture(uRoadTex, vec2(u * 10.0, v * 10.0)).rgb;
            
            // Apply road texture only on ROAD tiles (type 1)
            if(tileType > 0.5 && tileType < 1.5) {
                // This is a road tile
                color = roadColor * 0.85;  // Slightly darken for realism
            } else {
                // Not a road, use original ground color
                color = groundColor;
            }
        } else {
            color = groundColor;
        }
    } else {
        vec3 base;
        if(useTextures == 1){
            // Sample building texture with proper tiling
            base = texture(uBuildingTex, faceUV * 3.0).rgb;
            // Slight blend with building color for variation (70% texture, 30% tint)
            base = mix(base, base * buildingColor, 0.25);
        } else {
            base = buildingColor;
            // small window-dot pattern using faceUV
            float wx = fract(faceUV.x * 6.0);
            float wy = fract(faceUV.y * 6.0);
            float dotPattern = step(0.85, wx) * step(0.85, wy);
            base = mix(base, base * vec3(0.55,0.55,0.6), dotPattern);
        }
        color = base;
    }
    // lighting using actual normal
    vec3 n = normalize(vNormal);
    float diff = max(dot(n, normalize(-lightDir)), 0.0);
    color *= (0.35 + 0.65*diff);
    FragColor = vec4(color, 1.0);
}
)glsl";

// Cube mesh (indexed)
struct Mesh {
    GLuint VAO=0, VBO=0, EBO=0;
    GLsizei indexCount=0;
} cubeMesh;

void createCubeMesh(){
    if(cubeMesh.VAO) return;
    // pos(3), normal(3), uv(2)
    float verts[] = {
        // positions         normals            u,v
        -0.5f,-0.5f,-0.5f,   0.0f, 0.0f,-1.0f,  0.0f,0.0f, // 0
         0.5f,-0.5f,-0.5f,   0.0f, 0.0f,-1.0f,  1.0f,0.0f, // 1
         0.5f, 0.5f,-0.5f,   0.0f, 0.0f,-1.0f,  1.0f,1.0f, // 2
        -0.5f, 0.5f,-0.5f,   0.0f, 0.0f,-1.0f,  0.0f,1.0f, // 3

        -0.5f,-0.5f, 0.5f,   0.0f, 0.0f, 1.0f,  0.0f,0.0f, // 4
         0.5f,-0.5f, 0.5f,   0.0f, 0.0f, 1.0f,  1.0f,0.0f, // 5
         0.5f, 0.5f, 0.5f,   0.0f, 0.0f, 1.0f,  1.0f,1.0f, // 6
        -0.5f, 0.5f, 0.5f,   0.0f, 0.0f, 1.0f,  0.0f,1.0f  // 7
    };
    unsigned int idx[] = {
        4,5,6, 4,6,7,     // front
        1,0,3, 1,3,2,     // back
        0,4,7, 0,7,3,     // left
        5,1,2, 5,2,6,     // right
        3,7,6, 3,6,2,     // top
        0,1,5, 0,5,4      // bottom
    };
    glGenVertexArrays(1,&cubeMesh.VAO);
    glGenBuffers(1,&cubeMesh.VBO);
    glGenBuffers(1,&cubeMesh.EBO);

    glBindVertexArray(cubeMesh.VAO);
    glBindBuffer(GL_ARRAY_BUFFER, cubeMesh.VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cubeMesh.EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx), idx, GL_STATIC_DRAW);

    GLsizei stride = 8 * sizeof(float);
    // position
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,stride,(void*)0);
    // normal
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,stride,(void*)(3*sizeof(float)));
    // uv
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2,2,GL_FLOAT,GL_FALSE,stride,(void*)(6*sizeof(float)));

    cubeMesh.indexCount = sizeof(idx)/sizeof(idx[0]);
    glBindVertexArray(0);
}

// camera
glm::vec3 camPos(0.0f, 8.0f, 12.0f);
float camYaw = -90.0f, camPitch = -25.0f;
float camSpeed = 6.0f;
float camTurnSpeed = 60.0f;

glm::mat4 view, proj;
int windowW = WIN_W, windowH = WIN_H;
bool view3D = false;

void updateCameraMatrices(){
    glm::vec3 front;
    front.x = cos(glm::radians(camYaw)) * cos(glm::radians(camPitch));
    front.y = sin(glm::radians(camPitch));
    front.z = sin(glm::radians(camYaw)) * cos(glm::radians(camPitch));
    glm::vec3 center = camPos + glm::normalize(front);
    view = glm::lookAt(camPos, center, glm::vec3(0.0f,1.0f,0.0f));
    proj = glm::perspective(glm::radians(60.0f), (float)windowW / (float)windowH, 0.1f, 200.0f);
}

// building placement from tileMap
std::vector<glm::vec3> buildingPositions;
std::vector<float> buildingHeights;

// Map helpers: explicit pixel -> world conversions (consistent & robust)
inline float pixelToWorldX(int px, int imgW){
    // Map pixel X [0..imgW-1] -> world X centered at 0, scale = CELL pixels per world unit
    return ( (float)px - (float)imgW * 0.5f ) / (float)CELL;
}
inline float pixelToWorldZ(int py, int imgH){
    return ( (float)py - (float)imgH * 0.5f ) / (float)CELL;
}

float mapToWorldX(int gx){ return pixelToWorldX(gx, MAP_W); }
float mapToWorldZ(int gy){ return pixelToWorldZ(gy, MAP_H); }

// Toggle: set to true for strict 1-to-1 mapping (every blue marker => a building).
// Set to false to use the conservative "avoid roads / try nearby cell / nudge" logic.
const bool STRICT_BUILDING_PLACEMENT = true;

void init_buildings_from_tiles(const Image &ground){
    createCubeMesh();
    buildingPositions.clear();
    buildingHeights.clear();
    srand((unsigned)time(nullptr));

    const int cell = CELL;
    const int imgW = ground.w;
    const int imgH = ground.h;

    auto cellCenterWorld = [&](int gx, int gy){
        // gx/gy are cell origin in pixels; center is +cell/2
        int centerPxX = gx + cell/2;
        int centerPxY = gy + cell/2;
        float wx = pixelToWorldX(centerPxX, imgW);
        float wz = pixelToWorldZ(centerPxY, imgH);
        return glm::vec2(wx, wz);
    };

    auto cellIndexFromPixel = [&](int px, int py){
        // clamp
        int cx = std::clamp(px, 0, imgW-1);
        int cy = std::clamp(py, 0, imgH-1);
        // floor to cell origin
        int gx = (cx / cell) * cell;
        int gy = (cy / cell) * cell;
        return std::pair<int,int>(gx, gy);
    };

    // Helper: check if any pixel in the cell area is ROAD
    auto cellTouchesRoad = [&](int gx, int gy, int checkR)->bool{
        int minx = gx;
        int miny = gy;
        int maxx = std::min(imgW - 1, gx + cell - 1);
        int maxy = std::min(imgH - 1, gy + cell - 1);
        for(int y = std::max(0, miny - checkR); y <= std::min(imgH-1, maxy + checkR); ++y){
            for(int x = std::max(0, minx - checkR); x <= std::min(imgW-1, maxx + checkR); ++x){
                if(tileMap[tileIndex(x,y)] == (int)ROAD) return true;
            }
        }
        return false;
    };

    int placedStrict = 0, placedSafe = 0, skippedBecauseClose = 0;
    std::vector<std::pair<int,int>> seenCells; // unique set of chosen cells

    for(int y = 0; y < imgH; ++y){
        for(int x = 0; x < imgW; ++x){
            if(tileMap[tileIndex(x,y)] != (int)BUILDING) continue; // not a marker

            // map marker pixel to the canonical cell origin
            auto [gx, gy] = cellIndexFromPixel(x, y);

            // If we've already handled this cell (multiple pixels inside same marker),
            // skip duplicates
            bool already = false;
            for(auto &c : seenCells){
                if(c.first == gx && c.second == gy){ already = true; break; }
            }
            if(already) continue;
            seenCells.emplace_back(gx, gy);

            // compute world center for this cell
            glm::vec2 wc = cellCenterWorld(gx, gy);
            float wx = wc.x, wz = wc.y;

            if(STRICT_BUILDING_PLACEMENT){
                // Strict: place building at the cell center if not too close to an existing building
                bool tooClose = false;
                const float minDist = 0.5f; // smaller threshold for strict mode
                for(auto &bp : buildingPositions){
                    float dx = bp.x - wx;
                    float dz = bp.z - wz;
                    if((dx*dx + dz*dz) < (minDist * minDist)){
                        tooClose = true; break;
                    }
                }
                if(tooClose){
                    skippedBecauseClose++;
                    continue;
                }
                float jitterX = ((rand()%11)-5)/200.0f;
                float jitterZ = ((rand()%11)-5)/200.0f;
                buildingPositions.emplace_back(wx + jitterX, 0.0f, wz + jitterZ);
                buildingHeights.push_back(0.9f + (rand()%100)/140.0f);
                placedStrict++;
                continue;
            }

            // ----- Non-strict / safe behaviour (kept in case you flip the constant) -----
            const int checkR = 3;
            bool touchesRoad = cellTouchesRoad(gx, gy, checkR);

            int chosen_gx = gx;
            int chosen_gy = gy;
            bool chosen_by_search = false;

            if(touchesRoad){
                const int cellSearchRadius = 2;
                bool found = false;
                for(int r=1; r<=cellSearchRadius && !found; ++r){
                    for(int cy = -r; cy<=r && !found; ++cy){
                        for(int cx = -r; cx<=r && !found; ++cx){
                            int sgx = gx + cx * cell;
                            int sgy = gy + cy * cell;
                            if(sgx < 0 || sgy < 0 || sgx >= imgW || sgy >= imgH) continue;
                            if(cellTouchesRoad(sgx, sgy, checkR)) continue;
                            // check distance
                            glm::vec2 cw = cellCenterWorld(sgx, sgy);
                            bool tooClose = false;
                            for(auto &bp : buildingPositions){
                                float dx = bp.x - cw.x;
                                float dz = bp.z - cw.y;
                                if((dx*dx + dz*dz) < (0.9f * 0.9f)){ tooClose = true; break; }
                            }
                            if(tooClose) continue;
                            chosen_gx = sgx;
                            chosen_gy = sgy;
                            chosen_by_search = true;
                            found = true;
                        }
                    }
                }
            }

            glm::vec2 finalWC = cellCenterWorld(chosen_gx, chosen_gy);
            wx = finalWC.x; wz = finalWC.y;

            if(touchesRoad && !chosen_by_search){
                // nudge away from road (simple small push)
                int midPxX = gx + cell/2;
                int midPxY = gy + cell/2;
                glm::vec2 avgRoad(0.0f);
                int count = 0;
                for(int ry = std::max(0, gy - checkR); ry <= std::min(imgH-1, gy + cell - 1 + checkR); ++ry){
                    for(int rx = std::max(0, gx - checkR); rx <= std::min(imgW-1, gx + cell - 1 + checkR); ++rx){
                        if(tileMap[tileIndex(rx,ry)] == (int)ROAD){
                            avgRoad += glm::vec2((float)rx, (float)ry);
                            ++count;
                        }
                    }
                }
                if(count > 0){
                    avgRoad /= (float)count;
                    glm::vec2 cellPx(midPxX, midPxY);
                    glm::vec2 dir = cellPx - avgRoad;
                    if(glm::length(dir) < 1e-5f) dir = glm::vec2(1.0f,0.0f);
                    dir = glm::normalize(dir);
                    float pushPx = cell * 0.6f;
                    glm::vec2 newCenterPx = cellPx + dir * pushPx;
                    int newCellX = std::clamp(((int)newCenterPx.x / cell) * cell, 0, imgW-1);
                    int newCellY = std::clamp(((int)newCenterPx.y / cell) * cell, 0, imgH-1);
                    glm::vec2 newW = cellCenterWorld(newCellX, newCellY);
                    wx = newW.x; wz = newW.y;
                } else {
                    wx += ((rand()%11)-5)/200.0f;
                    wz += ((rand()%11)-5)/200.0f;
                }
            }

            // final distance check
            bool tooClose = false;
            const float minDist = 0.85f;
            for(auto &bp : buildingPositions){
                float dx = bp.x - wx;
                float dz = bp.z - wz;
                if((dx*dx + dz*dz) < (minDist * minDist)){
                    tooClose = true; break;
                }
            }
            if(tooClose){
                skippedBecauseClose++;
                continue;
            }
            // place building
            float jitterX = ((rand()%11)-5)/200.0f;
            float jitterZ = ((rand()%11)-5)/200.0f;
            buildingPositions.emplace_back(wx + jitterX, 0.0f, wz + jitterZ);
            buildingHeights.push_back(0.9f + (rand()%100)/140.0f);
            placedSafe++;
        }
    }

    if(STRICT_BUILDING_PLACEMENT){
        std::cout << "[init_buildings] STRICT mode -> placed " << placedStrict
                  << " buildings (skipped close: " << skippedBecauseClose << ")\n";
    } else {
        std::cout << "[init_buildings] SAFE mode -> placed " << placedSafe
                  << " buildings (skipped close: " << skippedBecauseClose << ")\n";
    }
}


void render_buildings(GLuint cubeProg, GLuint tex, GLuint tileMapTex){
    glUseProgram(cubeProg);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    glUniform1i(glGetUniformLocation(cubeProg,"uTex"), 0);
    
    // Bind tile map texture
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, tileMapTex);
    glUniform1i(glGetUniformLocation(cubeProg,"uTileMap"), 1);
    
    // Bind building texture
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, g_buildingTexture);
    glUniform1i(glGetUniformLocation(cubeProg,"uBuildingTex"), 2);
    
    // Bind road texture
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, g_roadTexture);
    glUniform1i(glGetUniformLocation(cubeProg,"uRoadTex"), 3);
    
    // Enable textures if they loaded successfully
    glUniform1i(glGetUniformLocation(cubeProg,"useTextures"), (g_buildingTexture && g_roadTexture) ? 1 : 0);

    glm::vec3 lightDir = glm::normalize(glm::vec3(-1.0f, -1.0f, -0.3f));
    glUniform3fv(glGetUniformLocation(cubeProg,"lightDir"), 1, glm::value_ptr(lightDir));

    // set building color uniform (nice visible color)
    glUniform3f(glGetUniformLocation(cubeProg,"buildingColor"), 0.85f, 0.6f, 0.6f);

    glBindVertexArray(cubeMesh.VAO);
    for(size_t i=0;i<buildingPositions.size();++i){
        glm::mat4 M = glm::translate(glm::mat4(1.0f), buildingPositions[i] + glm::vec3(0.0f, buildingHeights[i]/2.0f, 0.0f));
        M = glm::scale(M, glm::vec3(0.9f, buildingHeights[i], 0.9f));
        glm::mat4 MVP = proj * view * M;
        glUniformMatrix4fv(glGetUniformLocation(cubeProg,"MVP"),1,GL_FALSE, glm::value_ptr(MVP));
        glUniformMatrix4fv(glGetUniformLocation(cubeProg,"Model"),1,GL_FALSE, glm::value_ptr(M));
        // normal matrix = inverse transpose of model's upper-left 3x3
        glm::mat3 normalMat = glm::transpose(glm::inverse(glm::mat3(M)));
        glUniformMatrix3fv(glGetUniformLocation(cubeProg,"normalMatrix"), 1, GL_FALSE, glm::value_ptr(normalMat));
        glUniform1i(glGetUniformLocation(cubeProg,"isGround"), 0);
        glDrawElements(GL_TRIANGLES, cubeMesh.indexCount, GL_UNSIGNED_INT, 0);
    }
    glBindVertexArray(0);
}

int main(int argc, char** argv){
    // Default parameters
    int n_buildings = 6;
    int n_roads = 4;
    int n_roundabouts = 2;
    unsigned int seed = 0; // 0 => use time

    bool use_gui = false;
    // Simple flag parsing: --buildings N --roads N --roundabouts N --seed N --gui
    for(int i=1;i<argc;++i){
        std::string a = argv[i];
        if(a == "--buildings" && i+1 < argc){ n_buildings = std::max(0, std::stoi(argv[++i])); }
        else if(a == "--roads" && i+1 < argc){ n_roads = std::max(0, std::stoi(argv[++i])); }
        else if(a == "--roundabouts" && i+1 < argc){ n_roundabouts = std::max(0, std::stoi(argv[++i])); }
        else if(a == "--cell" && i+1 < argc){ CELL = std::max(4, std::stoi(argv[++i])); }
        else if(a == "--seed" && i+1 < argc){ seed = (unsigned)std::stoul(argv[++i]); }
        else if(a == "--road-pattern" && i+1 < argc){ g_road_pattern = std::clamp(std::stoi(argv[++i]), 0, 2); }
        else if(a == "--skyline" && i+1 < argc){ g_skyline_mode = std::clamp(std::stoi(argv[++i]), 0, 3); }
        else if(a == "--theme" && i+1 < argc){ g_theme_index = std::clamp(std::stoi(argv[++i]), 0, 3); }
        else if(a == "--park-radius" && i+1 < argc){ g_park_radius = std::max(0, std::stoi(argv[++i])); }
        else if(a == "--gui"){ use_gui = true; }
        else {
            // try positional fallback: buildings roads roundabouts
            if(i == 1) try{ n_buildings = std::max(0, std::stoi(a)); } catch(...){}
            else if(i == 2) try{ n_roads = std::max(0, std::stoi(a)); } catch(...){}
            else if(i == 3) try{ n_roundabouts = std::max(0, std::stoi(a)); } catch(...){}
        }
    }

    // if seed unset -> randomize
    if(seed == 0) seed = (unsigned)time(nullptr);
    srand(seed);

    // If user did not request GUI, run interactive CLI mode: prompt and generate -> write map.ppm
    if(!use_gui){
        initThemes();
        std::cout << "--- City Designer (CLI mode) ---\n";
        std::cout << "Enter number of buildings (int): ";
        if(!(std::cin >> n_buildings)) n_buildings = 6;

        std::cout << "Layout size (1=Small 2=Medium 3=Large 4=Custom cell px): ";
        int layoutSel = 2; if(!(std::cin >> layoutSel)) layoutSel = 2;
        if(layoutSel == 1) CELL = std::max(4, MAP_W / 8);
        else if(layoutSel == 2) CELL = std::max(4, MAP_W / 12);
        else if(layoutSel == 3) CELL = std::max(4, MAP_W / 16);
        else {
            std::cout << "Enter cell size in pixels (e.g. 16,32): ";
            if(!(std::cin >> CELL)) CELL = 32;
        }

        std::cout << "Road pattern (0=Grid 1=Radial 2=Random): "; if(!(std::cin >> g_road_pattern)) g_road_pattern = 0;
        std::cout << "Skyline (0=Low 1=Mid 2=Mix 3=Skyscraper): "; if(!(std::cin >> g_skyline_mode)) g_skyline_mode = 1;
        std::cout << "Theme (0=Default 1=Brick 2=Modern 3=Pastel): "; if(!(std::cin >> g_theme_index)) g_theme_index = 0;
        std::cout << "Park/Fountain radius in px (0=auto): "; if(!(std::cin >> g_park_radius)) g_park_radius = 0;
        std::cout << "Number of roads (int): "; if(!(std::cin >> n_roads)) n_roads = 4;
        std::cout << "Number of roundabouts (int): "; if(!(std::cin >> n_roundabouts)) n_roundabouts = 2;
        std::cout << "Optional seed (0=random): "; unsigned int inseed = 0; if(!(std::cin >> inseed)) inseed = seed; if(inseed != 0) seed = inseed;
        srand(seed);

        // generate and write PPM
        Image ground(MAP_W, MAP_H);
        generate_map_custom(ground, n_buildings, n_roads, n_roundabouts);
        const std::string out = "map.ppm";
            if(write_ppm(ground, out)){
            std::cout << "Wrote generated map to " << out << "\n";
            std::cout << "Grid cells available: " << std::max(1, ground.w / CELL) << " x " << std::max(1, ground.h / CELL) << " -> available building cells: " << g_last_available_cells << ", intersections: " << g_last_intersections << ", roundabouts placed: " << g_last_placed_roundabouts << " (requested: " << n_roundabouts << ")\n";
            // After CLI generation, automatically open the GUI to show the map
            use_gui = true;
        } else {
            std::cerr << "Failed to write " << out << "\n";
            return 1;
        }
    }

    // Initialize GLFW and request modern context
    if(!glfwInit()){ std::cerr << "glfwInit() failed\n"; return -1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* win = glfwCreateWindow(WIN_W, WIN_H, "City Designer - Part1", NULL, NULL);
    if(!win){ std::cerr<<"Window create failed\n"; glfwTerminate(); return -1; }
    glfwMakeContextCurrent(win);

    // NOTE: glad loader call was intentionally removed per your setup.
    // If you run on non-Apple OS and you need a loader, ensure you call your loader here.

    // Load textures from assets folder
    std::cout << "Loading textures...\n";
    g_buildingTexture = loadTexture("assets/building.jpg");
    g_roadTexture = loadTexture("assets/road.jpg");
    bool texturesLoaded = (g_buildingTexture != 0 && g_roadTexture != 0);
    if(!texturesLoaded) {
        std::cout << "Warning: Textures not loaded. Using procedural colors instead.\n";
    }

    // build the CPU map texture
    Image ground(MAP_W, MAP_H);
    generate_map_custom(ground, n_buildings, n_roads, n_roundabouts);
    // show effective constraints to the user
    int gridW = std::max(1, ground.w / CELL);
    int gridH = std::max(1, ground.h / CELL);
    std::cout << "Grid (cells): " << gridW << " x " << gridH << " -> available building cells: " << g_last_available_cells << ", intersections: " << g_last_intersections << ", roundabouts placed: " << g_last_placed_roundabouts << " (requested: " << n_roundabouts << ")\n";

    // upload texture
    GLuint tex;
    glGenTextures(1,&tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glPixelStorei(GL_UNPACK_ALIGNMENT,1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, ground.w, ground.h, 0, GL_RGB, GL_UNSIGNED_BYTE, ground.data.data());

    // Create tile map texture (single channel, stores tile type: 0=EMPTY, 1=ROAD, 2=PARK, 3=BUILDING)
    GLuint tileMapTex;
    std::vector<unsigned char> tileMapData(MAP_W * MAP_H);
    for(int i = 0; i < MAP_W * MAP_H; ++i) {
        tileMapData[i] = (unsigned char)tileMap[i];
    }
    glGenTextures(1, &tileMapTex);
    glBindTexture(GL_TEXTURE_2D, tileMapTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, MAP_W, MAP_H, 0, GL_RED, GL_UNSIGNED_BYTE, tileMapData.data());

    // fullscreen quad setup
    float quadVerts[] = {
        -1.0f,-1.0f, 0.0f,0.0f,
         1.0f,-1.0f, 1.0f,0.0f,
         1.0f, 1.0f, 1.0f,1.0f,
        -1.0f, 1.0f, 0.0f,1.0f
    };
    unsigned int quadIdx[] = {0,1,2, 0,2,3};
    GLuint qVAO, qVBO, qEBO;
    glGenVertexArrays(1,&qVAO);
    glGenBuffers(1,&qVBO);
    glGenBuffers(1,&qEBO);
    glBindVertexArray(qVAO);
    glBindBuffer(GL_ARRAY_BUFFER,qVBO);
    glBufferData(GL_ARRAY_BUFFER,sizeof(quadVerts),quadVerts,GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,qEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,sizeof(quadIdx),quadIdx,GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,4*sizeof(float),(void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,4*sizeof(float),(void*)(2*sizeof(float)));
    glBindVertexArray(0);

    // compile shaders (shader.h expected to implement compileShader/linkProgram)
    GLuint qVS = compileShader(GL_VERTEX_SHADER, quadVS);
    GLuint qFS = compileShader(GL_FRAGMENT_SHADER, quadFS);
    GLuint quadProg = linkProgram(qVS,qFS);
    glDeleteShader(qVS); glDeleteShader(qFS);

    GLuint cVS = compileShader(GL_VERTEX_SHADER, cubeVS);
    GLuint cFS = compileShader(GL_FRAGMENT_SHADER, cubeFS);
    GLuint cubeProg = linkProgram(cVS,cFS);
    glDeleteShader(cVS); glDeleteShader(cFS);

    // create cube mesh and init buildings from tileMap
    createCubeMesh();
    init_buildings_from_tiles(ground);
    updateCameraMatrices();

    // OpenGL state
    glEnable(GL_DEPTH_TEST);

    // Input state
    double lastTime = glfwGetTime();
    while(!glfwWindowShouldClose(win)){
        double now = glfwGetTime();
        float dt = float(now - lastTime);
        lastTime = now;
        glfwPollEvents();

        // Runtime controls: keyboard-driven parameter adjustments + regenerate
        static bool lastU=false, lastJ=false, lastI=false, lastK=false;
        static bool lastO=false, lastL=false, lastG=false;
        bool pressU = glfwGetKey(win, GLFW_KEY_U) == GLFW_PRESS; // buildings +1
        bool pressJ = glfwGetKey(win, GLFW_KEY_J) == GLFW_PRESS; // buildings -1
        bool pressI = glfwGetKey(win, GLFW_KEY_I) == GLFW_PRESS; // roads +1
        bool pressK = glfwGetKey(win, GLFW_KEY_K) == GLFW_PRESS; // roads -1
        bool pressO = glfwGetKey(win, GLFW_KEY_O) == GLFW_PRESS; // roundabouts +1
        bool pressL = glfwGetKey(win, GLFW_KEY_L) == GLFW_PRESS; // roundabouts -1
        bool pressG = glfwGetKey(win, GLFW_KEY_G) == GLFW_PRESS; // regenerate

        bool paramsChanged = false;
        if(pressU && !lastU){ n_buildings += 1; paramsChanged = true; }
        if(pressJ && !lastJ){ n_buildings = std::max(0, n_buildings - 1); paramsChanged = true; }
        if(pressI && !lastI){ n_roads += 1; paramsChanged = true; }
        if(pressK && !lastK){ n_roads = std::max(0, n_roads - 1); paramsChanged = true; }
        if(pressO && !lastO){ n_roundabouts += 1; paramsChanged = true; }
        if(pressL && !lastL){ n_roundabouts = std::max(0, n_roundabouts - 1); paramsChanged = true; }

        // regenerate if 'G' pressed or params changed
        bool doRegen = false;
        if(pressG && !lastG) doRegen = true;
        if(paramsChanged) doRegen = true;

        lastU = pressU; lastJ = pressJ; lastI = pressI; lastK = pressK;
        lastO = pressO; lastL = pressL; lastG = pressG;

        if(doRegen){
            std::cout << "Regenerating map -> buildings: " << n_buildings << ", roads: " << n_roads << ", roundabouts: " << n_roundabouts << "\n";
            // reseed for deterministic regeneration (same seed + params -> same layout)
            srand(seed + n_buildings * 73856093u + n_roads * 19349663u + n_roundabouts * 83492791u);
            generate_map_custom(ground, n_buildings, n_roads, n_roundabouts);
            // update GPU texture
            glBindTexture(GL_TEXTURE_2D, tex);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, ground.w, ground.h, 0, GL_RGB, GL_UNSIGNED_BYTE, ground.data.data());
            init_buildings_from_tiles(ground);
        }

        // handle toggle T (debounced)
        static bool lastT = false;
        bool tpress = glfwGetKey(win, GLFW_KEY_T) == GLFW_PRESS;
        if(tpress && !lastT){ view3D = !view3D; }
        lastT = tpress;

        // movement: WASD
        glm::vec3 forward;
        forward.x = cos(glm::radians(camYaw)) * cos(glm::radians(camPitch));
        forward.y = sin(glm::radians(camPitch));
        forward.z = sin(glm::radians(camYaw)) * cos(glm::radians(camPitch));
        glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0.0f,1.0f,0.0f)));

        if(glfwGetKey(win, GLFW_KEY_W) == GLFW_PRESS) camPos += forward * camSpeed * dt;
        if(glfwGetKey(win, GLFW_KEY_S) == GLFW_PRESS) camPos -= forward * camSpeed * dt;
        if(glfwGetKey(win, GLFW_KEY_A) == GLFW_PRESS) camPos -= right * camSpeed * dt;
        if(glfwGetKey(win, GLFW_KEY_D) == GLFW_PRESS) camPos += right * camSpeed * dt;

        // rotate camera: arrows
        if(glfwGetKey(win, GLFW_KEY_LEFT) == GLFW_PRESS) camYaw -= camTurnSpeed * dt;
        if(glfwGetKey(win, GLFW_KEY_RIGHT) == GLFW_PRESS) camYaw += camTurnSpeed * dt;
        if(glfwGetKey(win, GLFW_KEY_UP) == GLFW_PRESS) camPitch += camTurnSpeed * dt;
        if(glfwGetKey(win, GLFW_KEY_DOWN) == GLFW_PRESS) camPitch -= camTurnSpeed * dt;

        if(camPitch > 89.0f) camPitch = 89.0f;
        if(camPitch < -89.0f) camPitch = -89.0f;

        int ww, hh;
        glfwGetFramebufferSize(win, &ww, &hh);
        windowW = ww; windowH = hh;
        updateCameraMatrices();

        // Update the GLFW window title with live stats so the user can see
        // requested vs placed roundabouts (simple on-screen overlay via title).
        {
            char titleBuf[256];
            snprintf(titleBuf, sizeof(titleBuf), "City Designer - roundabouts placed: %d (requested: %d) | buildings: %d | roads: %d",
                     g_last_placed_roundabouts, n_roundabouts, n_buildings, n_roads);
            glfwSetWindowTitle(win, titleBuf);
        }

        // Render
        glViewport(0,0,ww,hh);
        glClearColor(0.65f,0.75f,0.95f,1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if(!view3D){
            // draw top-down quad (2D)
            glDisable(GL_DEPTH_TEST);
            glUseProgram(quadProg);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, tex);
            glUniform1i(glGetUniformLocation(quadProg,"uTex"), 0);
            glBindVertexArray(qVAO);
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
            glBindVertexArray(0);
            glEnable(GL_DEPTH_TEST);
        } else {
            // draw 3D scene
            glUseProgram(cubeProg);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, tex);
            glUniform1i(glGetUniformLocation(cubeProg,"uTex"), 0);
            
            // Bind tile map texture
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, tileMapTex);
            glUniform1i(glGetUniformLocation(cubeProg,"uTileMap"), 1);
            
            // Bind building texture
            glActiveTexture(GL_TEXTURE2);
            glBindTexture(GL_TEXTURE_2D, g_buildingTexture);
            glUniform1i(glGetUniformLocation(cubeProg,"uBuildingTex"), 2);
            
            // Bind road texture
            glActiveTexture(GL_TEXTURE3);
            glBindTexture(GL_TEXTURE_2D, g_roadTexture);
            glUniform1i(glGetUniformLocation(cubeProg,"uRoadTex"), 3);
            
            // Enable textures if they loaded successfully
            glUniform1i(glGetUniformLocation(cubeProg,"useTextures"), (g_buildingTexture && g_roadTexture) ? 1 : 0);

            // ground as a very flat scaled cube centered at origin
            glm::mat4 Mground = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -0.01f, 0.0f));
            float gw = MAP_W / (float)CELL;
            float gd = MAP_H / (float)CELL;
            Mground = glm::scale(Mground, glm::vec3(gw, 0.02f, gd));
            glm::mat4 MVPground = proj * view * Mground;
            glUniformMatrix4fv(glGetUniformLocation(cubeProg,"MVP"),1,GL_FALSE, glm::value_ptr(MVPground));
            glUniformMatrix4fv(glGetUniformLocation(cubeProg,"Model"),1,GL_FALSE, glm::value_ptr(Mground));
            // normal matrix for ground
            glm::mat3 normalMatGround = glm::transpose(glm::inverse(glm::mat3(Mground)));
            glUniformMatrix3fv(glGetUniformLocation(cubeProg,"normalMatrix"), 1, GL_FALSE, glm::value_ptr(normalMatGround));
            glUniform1f(glGetUniformLocation(cubeProg,"groundWidth"), gw);
            glUniform1f(glGetUniformLocation(cubeProg,"groundDepth"), gd);
            glUniform1i(glGetUniformLocation(cubeProg,"isGround"), 1);
            glm::vec3 lightDir = glm::normalize(glm::vec3(-1.0f,-1.0f,-0.3f));
            glUniform3fv(glGetUniformLocation(cubeProg,"lightDir"), 1, glm::value_ptr(lightDir));

            glBindVertexArray(cubeMesh.VAO);
            glDrawElements(GL_TRIANGLES, cubeMesh.indexCount, GL_UNSIGNED_INT, 0);

            // render buildings
            render_buildings(cubeProg, tex, tileMapTex);
            glBindVertexArray(0);
        }

        glfwSwapBuffers(win);
    }

    // Cleanup
    glDeleteProgram(quadProg);
    glDeleteProgram(cubeProg);
    glDeleteBuffers(1,&qVBO);
    glDeleteBuffers(1,&qEBO);
    glDeleteVertexArrays(1,&qVAO);
    glDeleteBuffers(1,&cubeMesh.VBO);
    glDeleteBuffers(1,&cubeMesh.EBO);
    glDeleteVertexArrays(1,&cubeMesh.VAO);
    glDeleteTextures(1,&tex);
    glDeleteTextures(1,&tileMapTex);
    if(g_buildingTexture) glDeleteTextures(1, &g_buildingTexture);
    if(g_roadTexture) glDeleteTextures(1, &g_roadTexture);

    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
