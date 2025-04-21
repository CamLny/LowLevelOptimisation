#include <iostream>
#include <cstdio>
#include <cstddef>
#include <cmath>

#include <list>
#include <vector>
#include <new>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <tracy/Tracy.hpp>
#include <algorithm>
#include <emmintrin.h>

constexpr int max_num_of_shapes = 32768;
constexpr int num_of_shape_types = 4;
constexpr float max_search_range = 0.125f;
constexpr float world_max_x = 1.0f;
constexpr float world_min_x = -1.0f;
constexpr float world_max_y = 1.0f;
constexpr float world_min_y = -1.0f;
__m128 WorldMin = _mm_set_ps(1, 1, world_min_y, world_min_x);
__m128 WorldMax = _mm_set_ps(1, 1, world_max_y, world_max_x);
__m128 WolrdSize = _mm_sub_ps(WorldMax, WorldMin);
constexpr float world_size_x = world_max_x - world_min_x;
constexpr float world_size_y = world_max_y - world_min_y;
constexpr float max_shape_size = 0.01f;
constexpr float max_shape_speed = max_shape_size * 0.5f;
constexpr float target_blend = 0.2f;
__m128 target_blendSIMD = _mm_set1_ps(target_blend);
int QuadXMax = 50;
int QuadYMax = 100;
int NBTileInGrid = QuadXMax * QuadYMax;

int attractor_type[4] = { 0, 1, 2, 3 };

struct Vertex
{
    float x;
    float y;

    float r;
    float g;
    float b;
};

struct Tile;

struct tri_list {
    void set_color(int i, unsigned char r, unsigned char g, unsigned char b);
    void set_position(int i, float x, float y);

    Vertex vertices[3];
};

void tri_list::set_color(int i, unsigned char r, unsigned char g, unsigned char b) {
    vertices[i].r = r;
    vertices[i].g = g;
    vertices[i].b = b;
}

void tri_list::set_position(int i, float x, float y) {
    vertices[i].x = x;
    vertices[i].y = y;
}


/*struct tri_list {
    void set_color(int i, unsigned char r, unsigned char g, unsigned char b);
    void set_position(int i, float x, float y);

    unsigned char m_red[3];
    unsigned char m_green[3];
    unsigned char m_blue[3];
    float m_px[3];
    float m_py[3];
};



void tri_list::set_color(int i, unsigned char r, unsigned char g, unsigned char b) {
    m_red[i] = r;
    m_green[i] = g;
    m_blue[i] = b;
}

void tri_list::set_position(int i, float x, float y) {
    m_px[i] = x;
    m_py[i] = y;
}*/

struct app {
    app() : m_num_of_shapes(0) {}
    virtual ~app() {}

    virtual int update(float dt, tri_list* tri) = 0;

    virtual void destroy_shapes(int num = max_num_of_shapes) = 0;
    virtual void spawn_triangle(float x, float y, float size) = 0;
    virtual void spawn_rectangle(float x, float y, float size) = 0;
    virtual void spawn_hexagon(float x, float y, float radius) = 0;
    virtual void spawn_octagon(float x, float y, float radius) = 0;

    int m_num_of_shapes;
};

inline int clampi(int v, int min, int max) {
    return v < min ? min : (v > max ? max : v);
}

// void* operator new(size_t size);
// void operator delete(void* p);

// shape
struct point_2d {
    point_2d() : m_x(0), m_y(0) { m_pos = _mm_set_ps(1, 1, m_y, m_x); }
    point_2d(float x, float y) : m_x(x), m_y(y) { m_pos = _mm_set_ps(1, 1, m_y, m_x); }

    //float get_x() const { return m_x; }
    //float get_y() const { return m_y; }
    __m128 get_pos() const { return m_pos; }

    private:

    float m_x;
    float m_y;
    __m128 m_pos;
};

/*
union shapeU
{
    triangle m_triangle;
    rectangle m_rectangle;
    octagon m_octagon;
    hexagon m_hexagon;


};*/

struct shape {
    shape() {}
    shape(float x, float y);
    virtual ~shape();

    //virtual void update(float dt, std::vector<Tile> Grid);
    virtual void update(float dt);
    virtual int draw(tri_list* tri) = 0;
    virtual bool test(shape* shape) = 0;
    virtual bool is_within(float x, float y) = 0;
    virtual int get_type() const = 0;
    

    bool IsInRange(shape* lookedShape);
    void find_target(shape* shape);
    void check_collision(shape* shape);

    float get_x() const { return _mm_cvtss_f32(m_pos); }
    float get_y() const { return _mm_cvtss_f32(_mm_shuffle_ps(m_pos, m_pos, _MM_SHUFFLE(1, 1, 1, 1))); }

    __m128 getpos() { return m_pos; }

    //static std::list<shape*> s_shapes;
    static std::vector<shape*> s_shapes;

    int CoordX = 0;
    int CoordY = 0;

protected:
    __m128 m_pos;
    float m_pos_x, m_pos_y;
    __m128 m_dir;
    float m_dir_x, m_dir_y;
    __m128 m_target;
    float m_target_x, m_target_y;
    float m_min_distance;
};

struct triangle : public shape {
    triangle(float x, float y, float size);
    virtual ~triangle() {}

    //void update(float dt, std::vector<Tile> Grid) override;
    void update(float dt) override;
    int draw(tri_list* tri) override;
    bool test(shape* shape) override;
    bool is_within(float x, float y) override;
    int get_type() const override { return 0; }

private:
    float m_size;
};

struct rectangle : public shape {
    rectangle(float x, float y, float size);
    virtual ~rectangle() {}

    //void update(float dt, std::vector<Tile> Grid) override;
    void update(float dt) override;
    int draw(tri_list* tri) override;
    bool test(shape* shape) override;
    bool is_within(float x, float y);
    int get_type() const { return 1; }

private:
    float m_size;
};

struct hexagon : public shape {
    hexagon(float x, float y, float radius);
    virtual ~hexagon() {}

    //void update(float dt, std::vector<Tile> Grid) override;
    void update(float dt) override;
    int draw(tri_list* tri) override;
    bool test(shape* shape) override;
    bool is_within(float x, float y) override;
    int get_type() const { return 2; }

private:
    float m_radius;
    point_2d m_points[6];
};

struct octagon : public shape {
    octagon(float x, float y, float radius);
    virtual ~octagon() {}

    //void update(float dt, std::vector<Tile> Grid) override;
    void update(float dt) override;
    int draw(tri_list* tri) override;
    bool test(shape* shape) override;
    bool is_within(float x, float y) override;
    int get_type() const override { return 3; }

private:
    float m_radius;
    point_2d m_points[8];
};

struct Tile {

    std::vector<shape*> shapes;
    int TileCoordX;
    int TileCoordY;

    void ReserveSize(int newSize) { shapes.reserve(newSize); }
};
std::vector<Tile> Grid(NBTileInGrid);
std::vector<shape*> shape::s_shapes;
//std::list<shape*> shape::s_shapes;

shape::shape(float x, float y)
    : m_pos_x(x), m_pos_y(y), m_dir_x(1.0f), m_dir_y(0.1f)
    , m_target_x(0), m_target_y(0), m_min_distance(max_search_range) {

    m_pos = _mm_set_ps(1, 1, m_pos_y, m_pos_x);
    m_dir = _mm_set_ps(1, 1, m_dir_y, m_dir_x);
    m_target = _mm_set_ps(1, 1, m_target_y, m_target_x);
}

shape::~shape() {
}

void shape::find_target(shape* shape) {

    __m128 delta = _mm_sub_ps(shape->getpos(), m_pos);
    __m128 length = _mm_mul_ps(delta, delta);
    length = _mm_hadd_ps(length, length);
    //length = _mm_sqrt_ss(length);
    length = _mm_rsqrt_ss(length);
    length = _mm_shuffle_ps(length, length, _MM_SHUFFLE(0, 0, 0, 0));

    if (_mm_cvtss_f32(length) < m_min_distance && shape->get_type() == attractor_type[get_type()]) {
        m_min_distance = _mm_cvtss_f32(length);

        __m128 result = _mm_cmpunord_ps(delta, delta);
        int maskNaN = _mm_movemask_ps(result);
        if (maskNaN != 0)
            std::cout << "NaN Delta in the Find Target" << std::endl;

        result = _mm_cmpunord_ps(length, length);
        maskNaN = _mm_movemask_ps(result);
        if (maskNaN != 0)
            std::cout << "NaN Length in the Find Target" << std::endl;

        m_target = _mm_mul_ps(delta, length);
    }

    /*
    float delta_x = shape->m_pos_x - m_pos_x;
    float delta_y = shape->m_pos_y - m_pos_y;
    float distance = sqrtf(delta_x * delta_x + delta_y * delta_y);

    if (distance < m_min_distance && shape->get_type() == attractor_type[get_type()]) {
        m_min_distance = distance;
        m_target_x = delta_x / distance;
        m_target_y = delta_y / distance;
    }*/
}

void shape::check_collision(shape* shape) {
    if (test(shape) || shape->test(this)) {

        /*float delta_x = shape->get_x() - m_pos_x;
        float delta_y = shape->get_y() - m_pos_y;

        float length = sqrtf(delta_x * delta_x + delta_y * delta_y);
        m_dir_x = -delta_x / length;
        m_dir_y = -delta_y / length;*/


        __m128 delta = _mm_sub_ps(shape->getpos(), m_pos);
        __m128 length = _mm_mul_ps(delta, delta);
        length = _mm_hadd_ps(length, length);
        //length = _mm_sqrt_ss(length);
        length = _mm_rsqrt_ss(length);
        length = _mm_shuffle_ps(length, length, _MM_SHUFFLE(0, 0, 0, 0));
        const __m128 sign_mask = _mm_set1_ps(-0.0f); // -0.0f a juste le bit de signe mis à 1
        m_dir = _mm_mul_ps(_mm_xor_ps(delta, sign_mask), length);


        //m_dir_x = -_mm_cvtss_f32(delta) / _mm_cvtss_f32(length);
        //m_dir_y = -_mm_cvtss_f32(_mm_shuffle_ps(delta, delta, _MM_SHUFFLE(1, 1, 1, 1))) / _mm_cvtss_f32(length);

    }
}

bool shape::IsInRange(shape* lookedShape)
{
    int tempLeftCoordX = lookedShape->CoordX - 1;
    int tempRightCoordX = lookedShape->CoordX + 1;
    int tempUpCoordY = lookedShape->CoordY - 1;
    int tempDownCoordY = lookedShape->CoordY + 1;
    if (CoordX == 0)
        tempLeftCoordX = QuadXMax;
    if (CoordX == QuadXMax)
        tempRightCoordX = 0;
    if (CoordY == 0)
        tempUpCoordY = QuadYMax;
    if (CoordY == QuadYMax)
        tempDownCoordY = 0;

    if ((tempLeftCoordX == CoordX - 1
        || lookedShape->CoordX == CoordX
        || tempRightCoordX == CoordX + 1)
        && (tempDownCoordY == CoordY - 1
            || lookedShape->CoordY == CoordY
            || tempUpCoordY == CoordY + 1))
        return true;
    else
        return false;

    /*const int x_gap = abs(lookedShape->CoordX - CoordX);
    const int y_gap = abs(lookedShape->CoordY - CoordY);

    return  (x_gap <= 1 || x_gap >= QuadXMax - 1) &&
        (y_gap <= 1 || y_gap >= QuadYMax - 1);*/
}

/*
std::vector<shape*> GetShapesInRange(const shape* current, std::vector<Tile> Grid)
{
    std::vector<shape*> output(current->s_shapes.size());
    output.insert(output.begin(), Grid[(current->CoordX - 1) * QuadYMax + current->CoordY - 1].shapes.begin(), Grid[(current->CoordX - 1) * QuadYMax + current->CoordY - 1].shapes.end());
    output.insert(output.end(), Grid[current->CoordX * QuadYMax + current->CoordY - 1].shapes.begin(), Grid[current->CoordX * QuadYMax + current->CoordY - 1].shapes.end());
    output.insert(output.end(), Grid[(current->CoordX + 1) * QuadYMax + current->CoordY - 1].shapes. begin(), Grid[(current->CoordX + 1) * QuadYMax + current->CoordY - 1].shapes.end());
    output.insert(output.end(), Grid[(current->CoordX - 1) * QuadYMax + current->CoordY].shapes.begin(), Grid[(current->CoordX - 1) * QuadYMax + current->CoordY].shapes.end());
    output.insert(output.end(), Grid[current->CoordX * QuadYMax + current->CoordY].shapes.begin(), Grid[current->CoordX * QuadYMax + current->CoordY].shapes.end());
    output.insert(output.end(), Grid[(current->CoordX + 1) * QuadYMax + current->CoordY - 1].shapes.begin(), Grid[(current->CoordX + 1) * QuadYMax + current->CoordY - 1].shapes.end());
    output.insert(output.end(), Grid[(current->CoordX - 1) * QuadYMax + current->CoordY + 1].shapes.begin(), Grid[(current->CoordX - 1) * QuadYMax + current->CoordY + 1].shapes.end());
    output.insert(output.end(), Grid[current->CoordX * QuadYMax + current->CoordY + 1].shapes.begin(), Grid[current->CoordX * QuadYMax + current->CoordY + 1].shapes.end());
    output.insert(output.end(), Grid[(current->CoordX + 1) * QuadYMax + current->CoordY + 1].shapes.begin(), Grid[(current->CoordX + 1) * QuadYMax + current->CoordY + 1].shapes.end());
    return output;
}*/



//void shape::update(float dt, std::vector<Tile> Grid) {
void shape::update(float dt) {



    __m128 resultMult1 = _mm_mul_ps(m_dir, _mm_set1_ps(1.0f - _mm_cvtss_f32(target_blendSIMD)));
    __m128 resultMult2 = _mm_mul_ps(m_target, target_blendSIMD);
    m_dir = _mm_add_ps(resultMult1, resultMult2);


    m_min_distance = max_search_range;
    m_target = m_dir;

    __m128 length = _mm_mul_ps(m_dir, m_dir);
  

    length = _mm_hadd_ps(length, length);
  
    //length = _mm_sqrt_ss(length);
    length = _mm_rsqrt_ss(length);
   

    length = _mm_shuffle_ps(length, length, _MM_SHUFFLE(0, 0, 0, 0));
  

    if(_mm_cvtss_f32(length) == 0)
        std::cout << "Length null" << std::endl;


    m_dir = _mm_mul_ps(m_dir, length);


    __m128 delta = _mm_mul_ps(m_dir, _mm_set1_ps(dt / 10000));


    // Masque pour ne garder que x et y (masque = { -1, -1, 0, 0 })
    __m128 mask = _mm_castsi128_ps(_mm_set_epi32(0, 0, -1, -1));

    // Applique le masque sur delta
    __m128 delta_masked = _mm_and_ps(delta, mask);
 

    // Met à jour m_pos uniquement sur x et y
    m_pos = _mm_add_ps(m_pos, delta_masked);
 

    // Si m_pos > world_max, soustraire world_size
    __m128 mask_max = _mm_cmpgt_ps(m_pos, WorldMax);
    m_pos = _mm_sub_ps(m_pos, _mm_and_ps(mask_max, WolrdSize));

    // Si m_pos < world_min, ajouter world_size
    __m128 mask_min = _mm_cmplt_ps(m_pos, WorldMin);
    m_pos = _mm_add_ps(m_pos, _mm_and_ps(mask_min, WolrdSize));

    /*/
    // Blend in target shape position
    m_dir_x = m_dir_x * (1.0f - target_blend) + m_target_x * target_blend;
    m_dir_y = m_dir_y * (1.0f - target_blend) + m_target_y * target_blend;

    // Reset target
    m_min_distance = max_search_range;
    m_target_x = m_dir_x;
    m_target_y = m_dir_y;

    // Normalize direction
    float length = sqrtf(m_dir_x * m_dir_x + m_dir_y * m_dir_y);
    m_dir_x /= length;
    m_dir_y /= length;

    // Move
    m_pos_x += dt/10000 * m_dir_x;
    m_pos_y += dt/10000 * m_dir_y;

    // Wrap around window frame
    if (m_pos_x > world_max_x)
        m_pos_x -= (world_size_x);
    if (m_pos_x < world_min_x)
        m_pos_x += (world_size_x);
    if (m_pos_y > world_max_y)
        m_pos_y -= (world_size_y);
    if (m_pos_y < world_min_y)
        m_pos_y += (world_size_y);

    */

    // Check collision against other shapes
    //std::list<shape*>::iterator i = s_shapes.begin();

    //UnOptimizedWay
    {
        /*
        int tempLeftCoordX = CoordX - 1;
        int tempRightCoordX = CoordX + 1;
        int tempUpCoordY = CoordY - 1;
        int tempDownCoordY = CoordY + 1;
        if (CoordX == 0)
            tempLeftCoordX = QuadXMax - 1;
        if (CoordX == QuadXMax - 1)
            tempRightCoordX = 0;
        if (CoordY == 0)
            tempUpCoordY = QuadYMax - 1;
        if (CoordY == QuadYMax - 1)
            tempDownCoordY = 0;

        int leftTop = tempLeftCoordX * QuadYMax + tempUpCoordY;
        int midTop = CoordX * QuadYMax + tempUpCoordY;
        int rightTop = tempRightCoordX * QuadYMax + tempUpCoordY;
        int left = tempLeftCoordX * QuadYMax + CoordY;
        int me = CoordX * QuadYMax + CoordY;
        int right = tempRightCoordX * QuadYMax + CoordY;
        int leftBot = tempLeftCoordX * QuadYMax + tempDownCoordY;
        int midBot = CoordX * QuadYMax + tempDownCoordY;
        int rightBot = tempRightCoordX * QuadYMax + tempDownCoordY;

        std::vector<shape*> closeShapes;
        //closeShapes.reserve(s_shapes.size());

        int size = Grid[leftTop].shapes.size();
        for (int i = 0; i < size; i++)
            closeShapes.push_back(Grid[leftTop].shapes[i]);

        size = Grid[midTop].shapes.size();
        for (int i = 0; i < Grid[midTop].shapes.size(); i++)
            closeShapes.push_back(Grid[midTop].shapes[i]);

        size = Grid[rightTop].shapes.size();
        for (int i = 0; i < Grid[rightTop].shapes.size(); i++)
            closeShapes.push_back(Grid[rightTop].shapes[i]);

        size = Grid[left].shapes.size();
        for (int i = 0; i < Grid[left].shapes.size(); i++)
            closeShapes.push_back(Grid[left].shapes[i]);

        size = Grid[me].shapes.size();
        for (int i = 0; i < Grid[me].shapes.size(); i++)
            closeShapes.push_back(Grid[me].shapes[i]);

        size = Grid[right].shapes.size();
        for (int i = 0; i < Grid[right].shapes.size(); i++)
            closeShapes.push_back(Grid[right].shapes[i]);

        size = Grid[leftBot].shapes.size();
        for (int i = 0; i < Grid[leftBot].shapes.size(); i++)
            closeShapes.push_back(Grid[leftBot].shapes[i]);

        size = Grid[midBot].shapes.size();
        for (int i = 0; i < Grid[midBot].shapes.size(); i++)
            closeShapes.push_back(Grid[midBot].shapes[i]);

        size = Grid[rightBot].shapes.size();
        for (int i = 0; i < Grid[rightBot].shapes.size(); i++)
            closeShapes.push_back(Grid[rightBot].shapes[i]);


        closeShapes.insert(closeShapes.begin(), Grid[leftTop].shapes.begin(), Grid[leftTop].shapes.end());
        closeShapes.insert(closeShapes.end(), Grid[midTop].shapes.begin(), Grid[midTop].shapes.end());
        closeShapes.insert(closeShapes.end(), Grid[rightTop].shapes.begin(), Grid[rightTop].shapes.end());
        closeShapes.insert(closeShapes.end(), Grid[left].shapes.begin(), Grid[left].shapes.end());
        closeShapes.insert(closeShapes.end(), Grid[me].shapes.begin(), Grid[me].shapes.end());
        closeShapes.insert(closeShapes.end(), Grid[right].shapes.begin(), Grid[right].shapes.end());
        closeShapes.insert(closeShapes.end(), Grid[leftBot].shapes.begin(), Grid[leftBot].shapes.end());
        closeShapes.insert(closeShapes.end(), Grid[midBot].shapes.begin(), Grid[midBot].shapes.end());
        closeShapes.insert(closeShapes.end(), Grid[rightBot].shapes.begin(), Grid[rightBot].shapes.end());

        std::vector<shape*>::iterator i = closeShapes.begin();
        while (i != closeShapes.end()) {
            //if (*i != this && (CoordX == (*i)->CoordX) && (CoordY == (*i)->CoordY))
            if (*i != this)
            {
                find_target(*i);
                check_collision(*i);
            }
            i++;
        }
        */
    }


    std::vector<shape*>::iterator i = s_shapes.begin();

    while (i != s_shapes.end()) {
        //if (*i != this && (CoordX == (*i)->CoordX) && (CoordY == (*i)->CoordY))
        if (*i != this && IsInRange((*i)))
        {
            find_target(*i);
            check_collision(*i);
        }
        i++;
    }
}

triangle::triangle(float x, float y, float size)
    : shape(x, y), m_size(size) {
}

bool triangle::test(shape* shape) {

    return shape->is_within(_mm_cvtss_f32(m_pos), _mm_cvtss_f32(_mm_shuffle_ps(m_pos, m_pos, _MM_SHUFFLE(1, 1, 1, 1))) + m_size)
        || shape->is_within(_mm_cvtss_f32(m_pos) - m_size, _mm_cvtss_f32(_mm_shuffle_ps(m_pos, m_pos, _MM_SHUFFLE(1, 1, 1, 1))) - m_size)
        || shape->is_within(_mm_cvtss_f32(m_pos) + m_size, _mm_cvtss_f32(_mm_shuffle_ps(m_pos, m_pos, _MM_SHUFFLE(1, 1, 1, 1))) - m_size);
}

bool edge_test(float p0x, float p0y, float p1x, float p1y, float x, float y) {
    float nx = -(p1y - p0y);
    float ny = p1x - p0x;

    float dot = nx * (x - p0x) + ny * (y - p0y);

    return dot < 0;
}

bool triangle::is_within(float x, float y) {
    float p0x = _mm_cvtss_f32(m_pos);
    float p0y = _mm_cvtss_f32(_mm_shuffle_ps(m_pos, m_pos, _MM_SHUFFLE(1, 1, 1, 1))) + m_size;
    float p1x = _mm_cvtss_f32(m_pos) + m_size;
    float p1y = _mm_cvtss_f32(_mm_shuffle_ps(m_pos, m_pos, _MM_SHUFFLE(1, 1, 1, 1))) - m_size;
    float p2x = _mm_cvtss_f32(m_pos) - m_size;
    float p2y = _mm_cvtss_f32(_mm_shuffle_ps(m_pos, m_pos, _MM_SHUFFLE(1, 1, 1, 1))) - m_size;

    return edge_test(p0x, p0y, p1x, p1y, x, y)
        && edge_test(p1x, p1y, p2x, p2y, x, y)
        && edge_test(p2x, p2y, p0x, p0y, x, y);
}

void triangle::update(float dt) {
    shape::update(dt);
}

int triangle::draw(tri_list* tri) {
    tri->set_color(0, 0, 255, 255);
    tri->set_position(0, _mm_cvtss_f32(m_pos), _mm_cvtss_f32(_mm_shuffle_ps(m_pos, m_pos, _MM_SHUFFLE(1, 1, 1, 1))) + m_size);
    tri->set_color(1, 0, 255, 255);
    tri->set_position(1, _mm_cvtss_f32(m_pos) - m_size, _mm_cvtss_f32(_mm_shuffle_ps(m_pos, m_pos, _MM_SHUFFLE(1, 1, 1, 1))) - m_size);
    tri->set_color(2, 0, 255, 255);
    tri->set_position(2, _mm_cvtss_f32(m_pos) + m_size, _mm_cvtss_f32(_mm_shuffle_ps(m_pos, m_pos, _MM_SHUFFLE(1, 1, 1, 1))) - m_size);
    return 1;
}

rectangle::rectangle(float x, float y, float size)
    : shape(x, y), m_size(size) {
}

bool rectangle::test(shape* shape) {
    return shape->is_within(_mm_cvtss_f32(m_pos) - m_size, _mm_cvtss_f32(_mm_shuffle_ps(m_pos, m_pos, _MM_SHUFFLE(1, 1, 1, 1))) - m_size)
        || shape->is_within(_mm_cvtss_f32(m_pos) + m_size, _mm_cvtss_f32(_mm_shuffle_ps(m_pos, m_pos, _MM_SHUFFLE(1, 1, 1, 1))) - m_size)
        || shape->is_within(_mm_cvtss_f32(m_pos) - m_size, _mm_cvtss_f32(_mm_shuffle_ps(m_pos, m_pos, _MM_SHUFFLE(1, 1, 1, 1))) + m_size)
        || shape->is_within(_mm_cvtss_f32(m_pos) + m_size, _mm_cvtss_f32(_mm_shuffle_ps(m_pos, m_pos, _MM_SHUFFLE(1, 1, 1, 1))) + m_size);
}

bool rectangle::is_within(float x, float y) {
    return x >= _mm_cvtss_f32(m_pos) - m_size
        && x <= _mm_cvtss_f32(m_pos) + m_size
        && y >= _mm_cvtss_f32(_mm_shuffle_ps(m_pos, m_pos, _MM_SHUFFLE(1, 1, 1, 1))) - m_size
        && y <= _mm_cvtss_f32(_mm_shuffle_ps(m_pos, m_pos, _MM_SHUFFLE(1, 1, 1, 1))) + m_size;
}

void rectangle::update(float dt) {
    shape::update(dt);
}

int rectangle::draw(tri_list* tri) {
    tri->set_color(0, 255, 0, 0);
    tri->set_position(0, _mm_cvtss_f32(m_pos) - m_size, _mm_cvtss_f32(_mm_shuffle_ps(m_pos, m_pos, _MM_SHUFFLE(1, 1, 1, 1))) - m_size);
    tri->set_color(1, 255, 0, 0);
    tri->set_position(1, _mm_cvtss_f32(m_pos) + m_size, _mm_cvtss_f32(_mm_shuffle_ps(m_pos, m_pos, _MM_SHUFFLE(1, 1, 1, 1))) - m_size);
    tri->set_color(2, 255, 0, 0);
    tri->set_position(2, _mm_cvtss_f32(m_pos) + m_size, _mm_cvtss_f32(_mm_shuffle_ps(m_pos, m_pos, _MM_SHUFFLE(1, 1, 1, 1))) + m_size);

    tri++;

    tri->set_color(0, 255, 0, 0);
    tri->set_position(0, _mm_cvtss_f32(m_pos) - m_size, _mm_cvtss_f32(_mm_shuffle_ps(m_pos, m_pos, _MM_SHUFFLE(1, 1, 1, 1))) - m_size);
    tri->set_color(1, 255, 0, 0);
    tri->set_position(1, _mm_cvtss_f32(m_pos) + m_size, _mm_cvtss_f32(_mm_shuffle_ps(m_pos, m_pos, _MM_SHUFFLE(1, 1, 1, 1))) + m_size);
    tri->set_color(2, 255, 0, 0);
    tri->set_position(2, _mm_cvtss_f32(m_pos) - m_size, _mm_cvtss_f32(_mm_shuffle_ps(m_pos, m_pos, _MM_SHUFFLE(1, 1, 1, 1))) + m_size);

    return 2;
}

hexagon::hexagon(float x, float y, float radius)
    : shape(x, y), m_radius(radius) {
    float radians = 0.0f;
    for (int a = 0; a < 6; a++, radians += 3.141592f / 6.0f * 2.0f) {
        m_points[a] = point_2d(cosf(radians) * m_radius, -sinf(radians) * m_radius);
    }
}

bool hexagon::test(shape* shape) {
    for (int a = 0; a < 6; a++) {
        if (shape->is_within(_mm_cvtss_f32(m_pos) + _mm_cvtss_f32(m_points[a % 6].get_pos()),
            _mm_cvtss_f32(_mm_shuffle_ps(m_pos, m_pos, _MM_SHUFFLE(1, 1, 1, 1))) + _mm_cvtss_f32(_mm_shuffle_ps(m_points[a % 6].get_pos(), m_points[a % 6].get_pos(), _MM_SHUFFLE(1, 1, 1, 1)))))
            return true;
    }
    return false;
}

bool hexagon::is_within(float x, float y) {
    int sum = 0;

    for (int a = 0; a < 6; a++) {
        sum += edge_test(_mm_cvtss_f32(m_pos) + _mm_cvtss_f32(m_points[a % 6].get_pos()), _mm_cvtss_f32(_mm_shuffle_ps(m_pos, m_pos, _MM_SHUFFLE(1, 1, 1, 1))) + _mm_cvtss_f32(_mm_shuffle_ps(m_points[a % 6].get_pos(), m_points[a % 6].get_pos(), _MM_SHUFFLE(1, 1, 1, 1))) , _mm_cvtss_f32(m_pos) + _mm_cvtss_f32(m_points[(a + 1) % 6].get_pos()), _mm_cvtss_f32(_mm_shuffle_ps(m_pos, m_pos, _MM_SHUFFLE(1, 1, 1, 1))) + _mm_cvtss_f32(_mm_shuffle_ps(m_points[(a + 1) % 6].get_pos(), m_points[(a + 1) % 6].get_pos(), _MM_SHUFFLE(1, 1, 1, 1))) , x, y);
    }

    return sum == 6;
}

void hexagon::update(float dt) {
    shape::update(dt);
}

int hexagon::draw(tri_list* tri) {
    for (int a = 0; a < 6; a++) {
        tri->set_color(0, 255, 0, 255);
        tri->set_position(0, _mm_cvtss_f32(m_pos), _mm_cvtss_f32(_mm_shuffle_ps(m_pos, m_pos, _MM_SHUFFLE(1, 1, 1, 1))));
        tri->set_color(1, 255, 0, 255);
        tri->set_position(1, _mm_cvtss_f32(m_pos) + _mm_cvtss_f32(m_points[a].get_pos()), _mm_cvtss_f32(_mm_shuffle_ps(m_pos, m_pos, _MM_SHUFFLE(1, 1, 1, 1))) + _mm_cvtss_f32(_mm_shuffle_ps(m_points[a].get_pos(), m_points[a].get_pos(), _MM_SHUFFLE(1, 1, 1, 1))) );
        tri->set_color(2, 255, 0, 255);
        tri->set_position(2, _mm_cvtss_f32(m_pos) + _mm_cvtss_f32(m_points[(a + 1) % 6].get_pos()), _mm_cvtss_f32(_mm_shuffle_ps(m_pos, m_pos, _MM_SHUFFLE(1, 1, 1, 1))) + _mm_cvtss_f32(_mm_shuffle_ps(m_points[(a + 1) % 6].get_pos(), m_points[(a + 1) % 6].get_pos(), _MM_SHUFFLE(1, 1, 1, 1))));

        tri++;
    }

    return 6;
}

octagon::octagon(float x, float y, float radius)
    : shape(x, y), m_radius(radius) {
    float radians = 0.0f;
    for (int a = 0; a < 8; a++, radians += 3.141592f / 8.0f * 2.0f) {
        m_points[a] = point_2d(cosf(radians) * m_radius, -sinf(radians) * m_radius);
    }
}

bool octagon::test(shape* shape) {
    for (int a = 0; a < 8; a++) {
        if (shape->is_within(_mm_cvtss_f32(m_pos) + _mm_cvtss_f32(m_points[a % 8].get_pos()) , _mm_cvtss_f32(_mm_shuffle_ps(m_pos, m_pos, _MM_SHUFFLE(1, 1, 1, 1))) + _mm_cvtss_f32(_mm_shuffle_ps(m_points[a % 8].get_pos(), m_points[a % 8].get_pos(), _MM_SHUFFLE(1, 1, 1, 1))) ))
            return true;
    }

    return false;
}

bool octagon::is_within(float x, float y) {
    int sum = 0;

    for (int a = 0; a < 8; a++) {
        sum += edge_test(_mm_cvtss_f32(m_pos) + _mm_cvtss_f32(m_points[a].get_pos()) , _mm_cvtss_f32(_mm_shuffle_ps(m_pos, m_pos, _MM_SHUFFLE(1, 1, 1, 1))) + _mm_cvtss_f32(_mm_shuffle_ps(m_points[a].get_pos(), m_points[a].get_pos(), _MM_SHUFFLE(1, 1, 1, 1))) , _mm_cvtss_f32(m_pos) + _mm_cvtss_f32(m_points[(a + 1) % 8].get_pos()) , _mm_cvtss_f32(_mm_shuffle_ps(m_pos, m_pos, _MM_SHUFFLE(1, 1, 1, 1))) + _mm_cvtss_f32(_mm_shuffle_ps(m_points[(a + 1) % 8].get_pos(), m_points[(a + 1) % 8].get_pos(), _MM_SHUFFLE(1, 1, 1, 1))) , x, y);
    }

    return sum == 8;
}

void octagon::update(float dt) {
    shape::update(dt);
}

int octagon::draw(tri_list* tri) {
    for (int a = 0; a < 8; a++) {
        tri->set_color(0, 255, 255, 0);
        tri->set_position(0, _mm_cvtss_f32(m_pos), _mm_cvtss_f32(_mm_shuffle_ps(m_pos, m_pos, _MM_SHUFFLE(1, 1, 1, 1))));
        tri->set_color(1, 255, 255, 0);
        tri->set_position(1, _mm_cvtss_f32(m_pos) + _mm_cvtss_f32(m_points[a].get_pos()) , _mm_cvtss_f32(_mm_shuffle_ps(m_pos, m_pos, _MM_SHUFFLE(1, 1, 1, 1))) + _mm_cvtss_f32(_mm_shuffle_ps(m_points[a].get_pos(), m_points[a].get_pos(), _MM_SHUFFLE(1, 1, 1, 1))) );
        tri->set_color(2, 255, 255, 0);
        tri->set_position(2, _mm_cvtss_f32(m_pos) + _mm_cvtss_f32(m_points[(a + 1) % 8].get_pos()) , _mm_cvtss_f32(_mm_shuffle_ps(m_pos, m_pos, _MM_SHUFFLE(1, 1, 1, 1))) + _mm_cvtss_f32(_mm_shuffle_ps(m_points[(a + 1) % 8].get_pos(), m_points[(a + 1) % 8].get_pos(), _MM_SHUFFLE(1, 1, 1, 1))) );

        tri++;
    }

    return 8;
};





struct default_app : public app {
    default_app();
    virtual ~default_app();

    int update(float dt, tri_list* tri) override;

    void destroy_shapes(int num = max_num_of_shapes) override;
    void spawn_triangle(float x, float y, float size) override;
    void spawn_rectangle(float x, float y, float size) override;
    void spawn_hexagon(float x, float y, float radius) override;
    void spawn_octagon(float x, float y, float radius) override;
};


default_app::default_app() {
}

default_app::~default_app() {
    destroy_shapes();
}

void default_app::destroy_shapes(int num) {
    for (int i = 0; i < num; i++) {
        if (shape::s_shapes.empty()) return;

        shape::s_shapes.pop_back();
    }
}

void default_app::spawn_triangle(float x, float y, float size) {
    shape::s_shapes.push_back(new triangle(x, y, size*0.5f));
}

void default_app::spawn_rectangle(float x, float y, float size) {
    shape::s_shapes.push_back(new rectangle(x, y, size * 0.5f));
}

void default_app::spawn_hexagon(float x, float y, float radius) {
    shape::s_shapes.push_back(new hexagon(x, y, radius));
}

void default_app::spawn_octagon(float x, float y, float radius) {
    shape::s_shapes.push_back(new octagon(x, y, radius));
}

int default_app::update(float dt, tri_list* tri) {
    ZoneScoped;


    int tri_count = 0;
    /*
    std::sort(shape::s_shapes.begin(), shape::s_shapes.end(),
        [](const shape* a, const shape* b)
        {
            return a->get_x() < b->get_x();
        });
        */

    
    for (int i = 0; i < QuadXMax; i++)
    {
        for (int y = 0; y < QuadYMax; y++)
        {
            Grid[i * QuadYMax + y].TileCoordX = i;
            Grid[i * QuadYMax + y].TileCoordY = y;
            Grid[i * QuadYMax + y].ReserveSize(shape::s_shapes.size());
        }
    }

    float caseSizeX = world_size_x / QuadXMax;
    float caseSizeY = world_size_y / QuadYMax;

    for (int i = 0; i < shape::s_shapes.size(); i++)
    {
        shape::s_shapes[i]->CoordX = floor((((shape::s_shapes[i]->get_x() - world_min_x) / world_size_x) * QuadXMax));;
        shape::s_shapes[i]->CoordY = floor((((shape::s_shapes[i]->get_y() - world_min_y) / world_size_y) * QuadYMax));;
        Grid[shape::s_shapes[i]->CoordX * QuadYMax + shape::s_shapes[i]->CoordY].shapes.push_back(shape::s_shapes[i]);
    }

    std::sort(shape::s_shapes.begin(), shape::s_shapes.end(),
        [](const shape* a, const shape* b)
        {
            if (a->CoordX != b->CoordX)
                return a->CoordX < b->CoordX;
            else
                return a->CoordY < b->CoordY;
        });

    /*shape::shapeSave.resize(shape::s_shapes.size());
    for (int i = 0; i < shape::s_shapes.size(); i++)
    {
        shape::shapeSave[i] = *shape::s_shapes[i];
    }*/
    
    /*float limitMin = -1.f;
    float limitMax = -0.5f;

    std::sort(shape::s_shapes.begin(), shape::s_shapes.end(),
        [](const shape* a, const shape* b)
        {
            if((limitMin > a-> get_x() < limitMax))
        });*/

    //std::list<shape*>::iterator i = shape::s_shapes.begin();

    std::vector<shape*>::iterator i = shape::s_shapes.begin();
    while (i != shape::s_shapes.end()) {
        (*i)->update(dt);
        tri_count += (*i)->draw(&tri[tri_count]);
        i++;

        FrameMark;
    }
    return tri_count;
}

struct factory_builder {
    app* (*func)();
    const char* name;
} factories[] = {
    { []() -> app* { return new default_app(); }, "default" }
};

app* factory(int index) {
    if (index < 0 || index >= (sizeof(factories) / sizeof(factories[0])))
        return nullptr;

    return factories[index].func();
}

static void glfw_error_callback(int error, const char *description) {
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

const char* glsl_version = "#version 130";

int randomInt(int min, int max) {
    return rand() % (max - min + 1) + min;
}


float randomFloat(float min, float max) {
    return ((float)rand() / RAND_MAX) * (max - min) + min;
}

void* operator new (std::size_t count)
{
    auto ptr = malloc(count);
    TracyAlloc(ptr, count);
    return ptr;
}

void operator delete (void* ptr) noexcept
{
    TracyFree(ptr);
    free(ptr);
}


int main(int ac, char *av[]) {
    GLFWwindow *window;

    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) {
        fprintf(stderr, "Could not initialize GLFW.\n");
        return EXIT_FAILURE;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    window = glfwCreateWindow(640, 480, "ISART Workshop", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return EXIT_FAILURE;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable VSync
                         //
    gladLoadGL();

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    bool show_demo_window = true;
    bool show_another_window = false;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    io.IniFilename = nullptr;

    app* current_app = factory(0);
    std::vector<tri_list> triangle_list;
    int tri_count = 0;
    int quad_count = 0;
    int hexa_count = 0;
    int octa_count = 0;

    const char* vtx_shader_src = "#version 330\n"
        "layout (location = 0) in vec2 aPos; \n"
        "layout (location = 1) in vec3 aCol; \n"
        "out vec3 fragColor; \n"
        "void main() {\n"
        "gl_Position.xyz = vec3(aPos, 1.0);\n"
        "gl_Position.w = 1.0;\n"
        "fragColor = aCol;\n"
        "}\n";

    const char* pix_shader_src = ""
        "#version 330\n"
        "out vec3 color;\n"
        "in vec3 fragColor;\n"
        ""
        "void main() {\n"
        "   color = fragColor;\n"
        "}\n";

    GLuint vtx_shader_id = glCreateShader(GL_VERTEX_SHADER);
    GLuint pix_shader_id = glCreateShader(GL_FRAGMENT_SHADER);

    glShaderSource(vtx_shader_id, 1, &vtx_shader_src, nullptr);
    glShaderSource(pix_shader_id, 1, &pix_shader_src, nullptr);

    glCompileShader(vtx_shader_id);
    glCompileShader(pix_shader_id);

    GLuint prog_id = glCreateProgram();
    glAttachShader(prog_id, vtx_shader_id);
    glAttachShader(prog_id, pix_shader_id);
    glLinkProgram(prog_id);

    glDetachShader(prog_id, pix_shader_id);
    glDetachShader(prog_id, vtx_shader_id);

    glDeleteShader(vtx_shader_id);
    glDeleteShader(pix_shader_id);

    GLuint vao_id;
    glGenVertexArrays(1, &vao_id);
    glBindVertexArray(vao_id);

    GLuint vbo_id;
    glGenBuffers(1, &vbo_id);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_id);


    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        if (glfwGetWindowAttrib(window, GLFW_ICONIFIED) != 0) {
            ImGui_ImplGlfw_Sleep(10);
            continue;
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        /*if (show_demo_window) {
            ImGui::ShowDemoWindow(&show_demo_window);
        }*/

        {
            static float f = 0.0f;
            static int counter = 0;

            ImGui::Begin("Hello, world!");
            ImGui::Checkbox("Demo Window", &show_demo_window);
            ImGui::Checkbox("Another Window", &show_another_window);

            ImGui::ColorEdit3("clear color", (float*)&clear_color);

            if (ImGui::Button("Button")) {
                counter++;
            }
            ImGui::SameLine();
            ImGui::Text("counter = %d", counter);

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
            ImGui::End();
        }

        // fill update, draw and stats here
        if (current_app) {
            ImGui::Begin("Shapes");
            ImGui::Text("Triangles: %d", tri_count);
            ImGui::SameLine();
            if (ImGui::Button("triangleAdd##Add")) {
                tri_count++;
                current_app->spawn_triangle(0.0f, 0.0f, max_shape_size);
                triangle_list.resize(triangle_list.size() + 1);
                

            }
            ImGui::Text("Rectangles: %d", quad_count);
            ImGui::SameLine();
            if (ImGui::Button("quadAdd##Add")) {
                quad_count++;
                current_app->spawn_rectangle(0.0f, 0.0f, max_shape_size);
                triangle_list.resize(triangle_list.size() + 2);
            }
            ImGui::Text("Hexagons: %d", hexa_count);
            ImGui::SameLine();
            if(ImGui::Button("hexaAdd##Add"))
            {
                hexa_count++;
                current_app->spawn_hexagon(0.0f, 0.0f, max_shape_size);
                triangle_list.resize(triangle_list.size() + 6);
            }
            ImGui::Text("Octagons: %d", octa_count);
            ImGui::SameLine();
            if(ImGui::Button("octaAdd##Add"))
            {
                octa_count++;
                current_app->spawn_octagon(0.0f, 0.0f, max_shape_size);
                triangle_list.resize(triangle_list.size() + 8);
            }
            if (ImGui::Button("Max shape##Add")) {
                //for (int s = 0; s < max_num_of_shapes; s++)
                for (int s = 0; s < 1000; s++)
                {
                    int rand = randomInt(1, 4);
                    if (rand == 1)
                    {
                        tri_count++;
                        current_app->spawn_triangle(randomFloat(-world_max_x, world_max_x), randomFloat(-world_max_y, world_max_y), max_shape_size);
                        triangle_list.resize(triangle_list.size() + 1);

                    }
                    else if (rand == 2)
                    {
                        quad_count++;
                        current_app->spawn_rectangle(randomFloat(-world_max_x, world_max_x), randomFloat(-world_max_y, world_max_y), max_shape_size);
                        triangle_list.resize(triangle_list.size() + 2);

                    }
                    else if (rand == 3)
                    {
                        hexa_count++;
                        current_app->spawn_hexagon(randomFloat(-world_max_x, world_max_x), randomFloat(-world_max_y, world_max_y), max_shape_size);
                        triangle_list.resize(triangle_list.size() + 6);

                    }
                    else if (rand == 4)
                    {
                        octa_count++;
                        current_app->spawn_octagon(randomFloat(-world_max_x, world_max_x), randomFloat(-world_max_y, world_max_y), max_shape_size);
                        triangle_list.resize(triangle_list.size() + 8);

                    }
                    std::cout << s << std::endl;
                }
                std::cout << "finish" << std::endl;

            }
            if (ImGui::Button("Reset all")) {
                tri_count = 0;
                quad_count = 0;
                hexa_count = 0;
                octa_count = 0;
                current_app->destroy_shapes();
                triangle_list.resize(0);
            }
            ImGui::End();

            current_app->update(1000.0f / io.Framerate, triangle_list.data());
        }

        float vertices[] = {
        -0.5f, -0.5f, 0.0f, // left  
         0.5f, -0.5f, 0.0f, // right 
         0.0f,  0.5f, 0.0f  // top   
        };

        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        if (triangle_list.size() > 0) {
            // how do I render a triangle list with opengl3?
            glUseProgram(prog_id);

            glBindVertexArray(vao_id);
            glEnableVertexAttribArray(0);
            glEnableVertexAttribArray(1);

            glBufferData(GL_ARRAY_BUFFER, triangle_list.size() * sizeof(Vertex) * 3, triangle_list.data(), GL_STATIC_DRAW);

            glBindBuffer(GL_ARRAY_BUFFER, vbo_id);
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, r));

            glDrawArrays(GL_TRIANGLES, 0, triangle_list.size() * 3);
            glDisableVertexAttribArray(0);

            glBindTexture(GL_TEXTURE0, 0);
            glUseProgram(0);
            glBindVertexArray(0);
        }
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
    return EXIT_SUCCESS;
}
