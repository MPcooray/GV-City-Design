// src/main.cpp
// Full updated main.cpp (with fixes for building placement & 2D markers)
// NOTE: removed 'glad' usage per your request — using system OpenGL headers.

#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <ctime>
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
const int CELL = 32;

enum Tile { EMPTY=0, ROAD=1, PARK=2, BUILDING=3 };
static std::vector<int> tileMap; // MAP_W * MAP_H

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
                setPixelAndTile(img, cx+dx, cy+dy, 20,20,20, ROAD);
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
void generate_map(Image &img){
    tileMap.assign(MAP_W * MAP_H, (int)EMPTY);
    for(int y=0;y<img.h;++y)
        for(int x=0;x<img.w;++x)
            img.setPixel(x,y,230,230,230);

    int roadHalf = 6;
    int CX = img.w / 2;
    int CY = img.h / 2;

    // MAIN ROADS
    draw_thick_road(img, CX - CELL,       0, CX - CELL,       img.h-1, roadHalf);   // left vertical
    draw_thick_road(img, CX + CELL * 2,   0, CX + CELL * 2,   img.h-1, roadHalf);   // right vertical
    draw_thick_road(img, 0, CY,           img.w-1, CY,        roadHalf);            // horizontal

    // ROUNDABOUTS
    int rr = CELL/2 - 2;
    int rb1x = CX - CELL;
    int rb2x = CX + CELL * 2;
    fill_circle_tile(img, rb1x, CY, rr, 34,139,34, PARK);
    fill_circle_tile(img, rb2x, CY, rr, 34,139,34, PARK);

    // Helper: place building marker (big 8×8 blue square)
    auto mark_build = [&](int px, int py){
        for(int dy=-4; dy<=4; ++dy)
            for(int dx=-4; dx<=4; ++dx)
                setPixelAndTile(img, px + dx, py + dy, 135,206,250, BUILDING);
    };

    // Helper: compute center of a CELL away from roundabout
    auto off = [&](int ox, int oy, int bx){
        int px = bx + ox * CELL * 2;   // 2 full cells away → safe from roads
        int py = CY + oy * CELL * 2;
        px = (px / CELL) * CELL + CELL/2;
        py = (py / CELL) * CELL + CELL/2;
        return std::make_pair(px, py);
    };

    // 3 buildings around roundabout 1
    auto a = off( 0, -1, rb1x); mark_build(a.first, a.second);
    auto b = off(-1,  0, rb1x); mark_build(b.first, b.second);
    auto c = off( 0,  1, rb1x); mark_build(c.first, c.second);

    // 3 buildings around roundabout 2
    auto d = off( 0, -1, rb2x); mark_build(d.first, d.second);
    auto e = off(+1,  0, rb2x); mark_build(e.first, e.second);
    auto f = off( 0,  1, rb2x); mark_build(f.first, f.second);
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
uniform float groundWidth;
uniform float groundDepth;
uniform int isGround;
uniform vec3 lightDir;
uniform vec3 buildingColor;
void main(){
    vec3 color;
    if(isGround == 1){
        float u = (worldPos.x / groundWidth) + 0.5;
        float v = (worldPos.z / groundDepth) + 0.5;
        color = texture(uTex, vec2(u, v)).rgb;
    } else {
        vec3 base = buildingColor;
        // small window-dot pattern using faceUV
        float wx = fract(faceUV.x * 6.0);
        float wy = fract(faceUV.y * 6.0);
        float dotPattern = step(0.85, wx) * step(0.85, wy);
        color = mix(base, base * vec3(0.55,0.55,0.6), dotPattern);
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


void render_buildings(GLuint cubeProg, GLuint tex){
    glUseProgram(cubeProg);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    glUniform1i(glGetUniformLocation(cubeProg,"uTex"), 0);

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

int main(){
    if(!glfwInit()){ std::cerr<<"GLFW init failed\n"; return -1; }

    // Request modern context
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

    // build the CPU map texture
    Image ground(MAP_W, MAP_H);
    generate_map(ground);

    // upload texture
    GLuint tex;
    glGenTextures(1,&tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glPixelStorei(GL_UNPACK_ALIGNMENT,1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, ground.w, ground.h, 0, GL_RGB, GL_UNSIGNED_BYTE, ground.data.data());

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
            render_buildings(cubeProg, tex);
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

    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
