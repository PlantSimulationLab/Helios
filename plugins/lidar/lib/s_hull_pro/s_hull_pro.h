#ifndef _structures_h
#define _structures_h




// for FILE

#include <stdlib.h>
#include <vector>
#include <set>
#include <cmath>



/* copyright 2016 Dr David Sinclair
   david@s-hull.org
 
   program to compute Delaunay triangulation of a set of points.

   this code is released under GPL3, 
   a copy ofthe license can be found at
   http://www.gnu.org/licenses/gpl-3.0.html

   you can purchase a un-restricted licnese from 
   http://www.s-hull.org 
   for the price of one beer!

   revised 12/feb/2016
 
 */


struct Triad
{
  int a,b, c;
  int ab, bc, ac;  // adjacent edges index to neighbouring triangle.
  float ro, R,C;
  //std::set<int> idx;
  Triad() {};
  Triad(int x, int y) : a(x), b(y),c(0), ab(-1), bc(-1), ac(-1), ro(-1), R(0), C(0) {};
  Triad(int x, int y, int z) : a(x), b(y), c(z),  ab(-1), bc(-1), ac(-1), ro(-1), R(0), C(0) {};
  Triad(const Triad &p) : a(p.a), b(p.b), c(p.c), ab(p.ab), bc(p.bc), ac(p.ac), ro(p.ro), R(p.R), C(p.C) {};

  Triad &operator=(const Triad &p)
  {
    a = p.a;
    b = p.b;
    c = p.c;

    ab = p.ab;
    bc = p.bc;
    ac = p.ac;

    ro = p.ro;
    R = p.R;
    C = p.C;

    return *this;
  };
};



/* point structure for s_hull only.
   has to keep track of triangle ids as hull evolves.


*/


struct Shx
{
  int id, trid;
  float r,c, tr,tc ;
  float ro;
  Shx() {};
  Shx(float a, float b) : r(a), c(b), ro(0.0), tr(0.0), tc(0.0), id(-1) {}; 
  Shx(float a, float b, float x) : r(a), c(b), ro(x), id(-1), tr(0), tc(0) {};
  Shx(const Shx &p) : id(p.id), trid(p.trid), r(p.r), c(p.c), tr(p.tr), tc(p.tc), ro(p.ro) {};

  Shx &operator=(const Shx &p)
  {
    id = p.id;
    trid = p.trid;
    r = p.r;
    c = p.c;
    tr = p.tr;
    tc = p.tc;
    ro = p.ro;
    return *this;
  };

};


// sort into descending order (for use in corner responce ranking).
inline bool operator<(const Shx &a, const Shx &b)
{
  // Use epsilon-based comparison for floating-point values to ensure
  // consistent sorting across different architectures (ARM64 vs x86_64)
  // and compiler optimizations. Direct equality (==) on floats causes
  // platform-dependent sort orders that cascade through triangulation.
  const float epsilon = 1e-9f;

  if( std::abs(a.ro - b.ro) < epsilon ){
    if( std::abs(a.r - b.r) < epsilon ){
      return a.c < b.c;
    }
    return a.r < b.r;
  }
  return a.ro <  b.ro;
};


struct Dupex
{
  int id;
  float r,c;  

  Dupex() {};
  Dupex(float a, float b) : r(a), c(b), id(-1) {}; 
  Dupex(float a, float b, int x) : r(a), c(b), id(x) {};
  Dupex(const Dupex &p) : id(p.id),  r(p.r), c(p.c) {};

  Dupex &operator=(const Dupex &p)
  {
    id = p.id;
    r = p.r;
    c = p.c;
    return *this;
  };
};



// sort into descending order (for use in corner responce ranking).
inline bool operator<(const Dupex &a, const Dupex &b)
{
  // Use epsilon-based comparison for cross-platform consistency
  const float epsilon = 1e-9f;
  if( std::abs(a.r - b.r) < epsilon )
    return a.c < b.c;
  return a.r <  b.r;
};





// from s_hull.C


int s_hull_pro( std::vector<Shx> &pts, std::vector<Triad> &triads);
void circle_cent2(float r1,float c1, float r2,float c2, float r3,float c3,float &r,float &c, float &ro2);
void circle_cent4(float r1,float c1, float r2,float c2, float r3,float c3,float &r,float &c, float &ro2);
void write_Shx(std::vector<Shx> &pts, char * fname);
void write_Triads(std::vector<Triad> &ts, char * fname);
int Cline_Renka_test(float &Ax, float &Ay, float &Bx, float &By, float &Cx, float &Cy, float &Dx, float &Dy);
int T_flip_pro( std::vector<Shx> &pts, std::vector<Triad> &triads, std::vector<int> &slump, int numt, 
		int start, std::vector<int> &ids);
int T_flip_pro_idx( std::vector<Shx> &pts, std::vector<Triad> &triads, std::vector<int> &slump, 
		    std::vector<int> &ids, std::vector<int> &ids2);

int read_Shx(std::vector<Shx> &pts, char * fname);
int de_duplicate( std::vector<Shx> &pts,  std::vector<int> &outx );
int de_duplicateX( std::vector<Shx> &pts, std::vector<int> &outx,std::vector<Shx> &pts2 );
int  test_center(Shx &pt0, Shx &pt1,Shx &pt2);

int T_flip_edge( std::vector<Shx> &pts, std::vector<Triad> &triads, std::vector<int> &slump, 
		 int numt, int start, std::vector<int> &ids);


#endif
