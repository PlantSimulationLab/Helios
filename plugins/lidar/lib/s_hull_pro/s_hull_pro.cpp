#include <iostream>
//#include <hash_set.h>
//#include <hash_set>
#include <set>
#include <vector>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <algorithm>


#include "s_hull_pro.h"

using namespace std;


/* copyright 2016 Dr David Sinclair
   david@s-hull.org
 
   program to compute Delaunay triangulation of a set of points.

   this code is released under GPL3, 
   a copy ofthe license can be found at
   http://www.gnu.org/licenses/gpl-3.0.html

   you can purchase a un-restricted licnese from 
   http://www.s-hull.org 
   for the price of one beer!

   revised 2/April/2016
 
 */






void circle_cent2(float r1,float c1, float r2,float c2, float r3,float c3,
float &r,float &c, float &ro2){
  /*
   *  function to return the center of a circle and its radius
   * degenerate case should never be passed to this routine!!!!!!!!!!!!!
   * but will return r0 = -1 if it is.
   */
   
 float a1 = (r1+r2)/2.0;
 float a2 = (c1+c2)/2.0;
  float b1 = (r3+r2)/2.0;
 float b2 = (c3+c2)/2.0;

 float e2 = r1-r2;
  float e1 = -c1+c2;

 float q2 = r3-r2;
  float q1 = -c3+c2;

  r=0; c=0; ro2=-1;
  if( e1*-q2 + e2*q1 == 0 ) return;

  float beta = (-e2*(b1-a1) + e1*(b2-a2))/( e2*q1-e1*q2);

  r = b1 + q1*beta;
  c = b2 + q2*beta;
   
  ro2 = (r1-r)*(r1-r) + (c1-c)*(c1-c);
   return;
}


/*  
    read an ascii file of (r,c) point pairs.

    the first line of the points file should contain
    "NUMP  2 points"

    if it does not have the word points in it the first line is 
    interpretted as a point pair.

 */

int read_Shx(std::vector<Shx> &pts, char * fname){
  char s0[513];
  int nump =0;
  float p1,p2;

  Shx pt;

  std::string line;
  std::string points_str("points");

  std::ifstream myfile;
  myfile.open(fname);

  if (myfile.is_open()){
    
    getline (myfile,line);
	//int numc = line.length();

    // check string for the string "points"
    int n = (int) line.find( points_str);
	if( n > 0){ 
      while ( myfile.good() ){
	getline (myfile,line);
	if( line.length() <= 512){
	  copy( line.begin(), line.end(), s0);
	  s0[line.length()] = 0;
	  int v = sscanf( s0, "%g %g", &p1,&p2);
	  if( v>0 ){
	    pt.id = nump; 
	    nump++;
	    pt.r = p1;
	    pt.c = p2;
	    pts.push_back(pt);
	  }
	}   
      }
    }
    else{   // assume all number pairs on a line are points
      if( line.length() <= 512){
	copy( line.begin(), line.end(), s0);
	s0[line.length()] = 0;
	int v = sscanf( s0, "%g %g", &p1,&p2);
	if( v>0 ){
	    pt.id = nump; 
	    nump++;
	  pt.r = p1;
	  pt.c = p2;
	  pts.push_back(pt);
	}
      }   

      while ( myfile.good() ){
	getline (myfile,line);
	if( line.length() <= 512){
	  copy( line.begin(), line.end(), s0);
	  s0[line.length()] = 0;
	  int v = sscanf( s0, "%g %g", &p1,&p2);
	  if( v>0 ){
	    pt.id = nump; 
	    nump++;
	    pt.r = p1;
	    pt.c = p2;
	    pts.push_back(pt);
	  }
	}   
      }
    }
    myfile.close();
  }

  nump = (int) pts.size();

  return(nump);
};

/*
	write out a set of points to disk


*/

void write_Shx(std::vector<Shx> &pts, char * fname){
   std::ofstream out(fname, ios::out);
   
   int nr = (int) pts.size();
   //out << nr << " 2 points" << endl;
   
   for (int r = 0; r < nr; r++){
     out << pts[r].r << ' ' << pts[r].c <<  endl;
   }
   out.close();
   
   return;
};



/*
 write out triangle ids to be compatible with matlab/octave array numbering.

 */
void write_Triads(std::vector<Triad> &ts, char * fname){
   std::ofstream out(fname, ios::out);
   
   int nr = (int) ts.size();
   out << nr << " 6   point-ids (1,2,3)  adjacent triangle-ids ( limbs ab  ac  bc )" << endl;
   
   for (int r = 0; r < nr; r++){
     out << ts[r].a+1 << ' ' << ts[r].b+1 <<' ' << ts[r].c+1 <<' ' 
	 << ts[r].ab+1 <<' ' << ts[r].ac+1 <<' ' << ts[r].bc+1 << endl; //" " << ts[r].ro <<  endl;
   }
   out.close();
   
   return;
};





/*  version in which the ids of the triangles associated with the sides of the hull are tracked.
   

 */

int s_hull_pro( std::vector<Shx> &pts, std::vector<Triad> &triads)
{

  //write_Shx(pts,"points.txt");
  
  int nump = (int) pts.size();


  if( nump < 3 ){
    cerr << "less than 3 points, aborting " << endl;
    return(-1);
  }


  float r = pts[0].r;
  float c = pts[0].c;
  for( int k=0; k<nump; k++){
    float dr = pts[k].r-r;
    float dc = pts[k].c-c;

    pts[k].ro = dr*dr + dc*dc;

  }

  sort( pts.begin(), pts.end() );

  
  float r1 = pts[0].r;
  float c1 = pts[0].c;

  float r2 = pts[1].r;
  float c2 = pts[1].c;
  int mid = -1;
  float romin2 =  9.0e20, ro2, R,C;

  int k=2; 
  while (k<nump){

    circle_cent2(r1,c1,r2,c2,  pts[k].r,  pts[k].c, r,c,ro2);
    if( ro2 < romin2 && ro2 > 0 ){
      mid = k;
      romin2 = ro2;
      R = r;
      C = c;

    }
    else if( romin2 *4 < pts[k].ro )
      k=nump;

    k++;
  }

  if( mid < 0 ){
    cerr << "linear structure, aborting " << endl;
    return(-2);
  }


  Shx pt0 = pts[0];
  Shx pt1 = pts[1];
  Shx pt2 = pts[mid];

  int ptest = test_center(pt0, pt1, pt2 );
  if( ptest < 0 ){
    //cerr << "warning: obtuce seed triangle selected " << endl;
  }


  pts.erase(pts.begin() + mid);  // necessary for round off reasons:((((((
  pts.erase(pts.begin() );
  pts.erase(pts.begin() );

   for( int k=0; k<nump-3; k++){
    float dr = pts[k].r-R;
    float dc = pts[k].c-C;

    pts[k].ro = dr*dr + dc*dc;

  }

  sort( pts.begin(), pts.end() ); 

  pts.insert(pts.begin(), pt2);
  pts.insert(pts.begin(), pt1);
  pts.insert(pts.begin(), pt0);

  std::vector<int> slump;
  slump.resize(nump);

  for( int k=0; k<nump; k++){
    if( pts[k].id < nump){
      slump[ pts[k].id] = k;
    }
    else{
      int mx = pts[k].id+1;
      while( (int) slump.size() <= mx){
	slump.push_back(0);
      }
      slump[pts[k].id] = k;
    }
  }

  std::vector<Shx> hull;

  r = (pts[0].r + pts[1].r + pts[2].r )/(float) 3.0;
  c = (pts[0].c + pts[1].c + pts[2].c )/(float) 3.0;
  
  float dr0 = pts[0].r - r,  dc0 = pts[0].c - c;
  float tr01 =  pts[1].r - pts[0].r, tc01 =  pts[1].c - pts[0].c;

  float df = -tr01* dc0 + tc01*dr0;
  if( df < 0 ){   // [ 0 1 2 ]
    pt0.tr = pt1.r-pt0.r;
    pt0.tc = pt1.c-pt0.c;    
    pt0.trid = 0;
    hull.push_back( pt0 );

    pt1.tr = pt2.r-pt1.r;
    pt1.tc = pt2.c-pt1.c;    
    pt1.trid = 0;
    hull.push_back( pt1 );

    pt2.tr = pt0.r-pt2.r;
    pt2.tc = pt0.c-pt2.c;
    pt2.trid = 0;
    hull.push_back( pt2 );

    
    Triad tri(pt0.id,pt1.id,pt2.id);
    tri.ro = romin2;
    tri.R = R;
    tri.C = C;

    triads.push_back(tri);

  }
  else{          // [ 0 2 1 ] as anti-clockwise turning is the work of the devil....
    pt0.tr = pt2.r-pt0.r;
    pt0.tc = pt2.c-pt0.c;  
    pt0.trid = 0;
    hull.push_back( pt0 );

    pt2.tr = pt1.r-pt2.r;
    pt2.tc = pt1.c-pt2.c;    
    pt2.trid = 0;
    hull.push_back( pt2 );

    pt1.tr = pt0.r-pt1.r;
    pt1.tc = pt0.c-pt1.c;
    pt1.trid = 0;
    hull.push_back( pt1 );

    Triad tri(pt0.id,pt2.id,pt1.id);
    tri.ro = romin2;
    tri.R = R;
    tri.C = C;
    triads.push_back(tri);
  }

  // add new points into hull (removing obscured ones from the chain)
  // and creating triangles....
  // that will need to be flipped.

  float dr, dc, rx,cx;
  Shx  ptx;
  int numt;

  //  write_Triads(triads, "rose_0.mat");

  for( int k=3; k<nump; k++){
    rx = pts[k].r;    cx = pts[k].c;
    ptx.r = rx;
    ptx.c = cx;
    ptx.id = pts[k].id;

    int numh = (int) hull.size(), numh_old = numh;
    dr = rx- hull.at(0).r;    dc = cx- hull.at(0).c;  // outwards pointing from hull.at(0] to pt.

    std::vector<int> pidx, tridx;
    int hidx;  // new hull point location within hull.....

    float df = -dc* hull.at(0).tr + dr*hull.at(0).tc;    // visibility test vector.
    if( df < 0 ){  // starting with a visible hull facet !!!
      int e1 = 1, e2 = numh;
      hidx = 0;

      // check to see if segment numh is also visible
      df = -dc* hull.at(numh-1).tr + dr*hull.at(numh-1).tc;
      //cerr << df << ' ' ;
      if( df < 0 ){    // visible.
	pidx.push_back(hull.at(numh-1).id);
	tridx.push_back(hull.at(numh-1).trid);
	

	for( int h=0; h<numh-1; h++){
	  // if segment h is visible delete h
	  dr = rx- hull.at(h).r;    dc = cx- hull.at(h).c;
	  df = -dc* hull.at(h).tr + dr*hull.at(h).tc;
	  pidx.push_back(hull.at(h).id);
	  tridx.push_back(hull.at(h).trid);
	  if( df < 0 ){
	    hull.erase(hull.begin() + h);
	    h--;
	    numh--;
	    
	  }
	  else{	  // quit on invisibility
	    ptx.tr = hull.at(h).r - ptx.r;
	    ptx.tc = hull.at(h).c - ptx.c;

	    hull.insert( hull.begin() , ptx);
	    numh++;
	    break;
	  }
	}
	// look backwards through the hull structure.
	
	for( int h=numh-2; h>0; h--){
	  // if segment h is visible delete h + 1
	  dr = rx- hull.at(h).r;    dc = cx- hull.at(h).c;
	  df = -dc* hull.at(h).tr + dr*hull.at(h).tc;

	  if( df < 0 ){  // h is visible 
	    pidx.insert(pidx.begin(), hull.at(h).id);
	    tridx.insert(tridx.begin(), hull.at(h).trid);
	    hull.erase(hull.begin() + h+1);  // erase end of chain
	    
	  }
	  else{

	    h = (int) hull.size()-1;
	    hull.at(h).tr = -hull.at(h).r + ptx.r;   // points at start of chain.
	    hull.at(h).tc = -hull.at(h).c + ptx.c;
	    break;
	  }
	}

	df = 9;

      }
      else{
	//	cerr << df << ' ' << endl;
	hidx = 1;  // keep pt hull.at(0]
	tridx.push_back(hull.at(0).trid);
	pidx.push_back(hull.at(0).id);

	for( int h=1; h<numh; h++){
	  // if segment h is visible delete h  
	  dr = rx- hull.at(h).r;    dc = cx- hull.at(h).c;
	  df = -dc* hull.at(h).tr + dr*hull.at(h).tc;
	  pidx.push_back(hull.at(h).id);
	  tridx.push_back(hull.at(h).trid);
	  if( df < 0 ){                     // visible
	    hull.erase(hull.begin() + h);
	    h--;
	    numh--;
	  }
	  else{	  // quit on invisibility
	    ptx.tr = hull.at(h).r - ptx.r;
	    ptx.tc = hull.at(h).c - ptx.c;

	    hull.at(h-1).tr = ptx.r - hull.at(h-1).r;
	    hull.at(h-1).tc = ptx.c - hull.at(h-1).c;

	    hull.insert( hull.begin()+h, ptx);
	    break;
	  }
	}
      }

      df = 8;

    }
    else{
      int e1 = -1,  e2 = numh;
      for( int h=1; h<numh; h++){	
	dr = rx- hull.at(h).r;    dc = cx- hull.at(h).c;
	df = -dc* hull.at(h).tr + dr*hull.at(h).tc;
	if( df < 0 ){
	  if( e1 < 0 ) e1 = h;  // fist visible
	}
	else{
	  if( e1 > 0 ){ // first invisible segment.
	    e2 = h;
	    break;
	  }
	}

      }

      if( e1<0 ){//failed for some reason
	std::cout << "triangle failed" << std::endl;
	continue;
      }

      // triangle pidx starts at e1 and ends at e2 (inclusive).	
      if( e2 < numh ){
	for( int e=e1; e<=e2; e++){
	  pidx.push_back(hull.at(e).id);
	  tridx.push_back(hull.at(e).trid);
	}
      }
      else{
	for( int e=e1; e<e2; e++){
	  pidx.push_back(hull.at(e).id);
	  tridx.push_back(hull.at(e).trid);   // there are only n-1 triangles from n hull pts.
	}
	pidx.push_back(hull.at(0).id);
      }

      // erase elements e1+1 : e2-1 inclusive.
      
      if( e1 < e2-1){
	hull.erase(hull.begin() + e1+1, hull.begin()+ e2); 
      }

      // insert ptx at location e1+1.
      if( e2 == numh){
	ptx.tr = hull.at(0).r - ptx.r;
	ptx.tc = hull.at(0).c - ptx.c;
      }
      else{
	ptx.tr = hull.at(e1+1).r - ptx.r;
	ptx.tc = hull.at(e1+1).c - ptx.c;
      }

      hull.at(e1).tr = ptx.r - hull.at(e1).r;
      hull.at(e1).tc = ptx.c - hull.at(e1).c;
      
      hull.insert( hull.begin()+e1+1, ptx);
      hidx = e1+1;
      
    }

    int a = ptx.id, T0;
    Triad trx( a, 0,0);
    r1 = pts[slump[a]].r;
    c1 = pts[slump[a]].c;

    int npx = (int) pidx.size()-1;
    numt = (int) triads.size();
    T0 = numt;

    if( npx == 1){
       trx.b = pidx[0];
       trx.c = pidx[1];
      
      trx.bc = tridx[0];
      trx.ab = -1;
      trx.ac = -1;

      // index back into the triads.
      Triad &txx = triads.at(tridx[0]);
      if( ( trx.b == txx.a && trx.c == txx.b) |( trx.b == txx.b && trx.c == txx.a)) {
	txx.ab = numt;
      }
      else if( ( trx.b == txx.a && trx.c == txx.c) |( trx.b == txx.c && trx.c == txx.a)) {
	txx.ac = numt;
      }
      else if( ( trx.b == txx.b && trx.c == txx.c) |( trx.b == txx.c && trx.c == txx.b)) {
	txx.bc = numt;
      }
      

      hull.at(hidx).trid = numt;
      if( hidx > 0 )
	hull.at(hidx-1).trid = numt;
      else{
	numh = (int) hull.size();
	hull.at(numh-1).trid = numt;
      }
      triads.push_back( trx );
      numt++;
    }
    
    else{
      trx.ab = -1;
      for(int p=0; p<npx; p++){
	trx.b = pidx[p];
	trx.c = pidx[p+1];
  
    
	trx.bc = tridx[p];
	if( p > 0 )
	  trx.ab = numt-1;
	trx.ac = numt+1;

	if( tridx.at(p)<0 || tridx.at(p)>=triads.size() ){
	  std::cout << "WARNING: Skipping triangle." << std::endl;
	  continue;
	}
	
	// index back into the triads.
	Triad &txx = triads.at(tridx.at(p));
	if( ( trx.b == txx.a && trx.c == txx.b) |( trx.b == txx.b && trx.c == txx.a)) {
	  txx.ab = numt;
	}
	else if( ( trx.b == txx.a && trx.c == txx.c) |( trx.b == txx.c && trx.c == txx.a)) {
	  txx.ac = numt;
	}
	else if( ( trx.b == txx.b && trx.c == txx.c) |( trx.b == txx.c && trx.c == txx.b)) {
	  txx.bc = numt;
	}
      
	triads.push_back( trx );
	numt++;
      }
      triads[numt-1].ac=-1;

      hull.at(hidx).trid = numt-1;
      if( hidx > 0 )
	hull.at(hidx-1).trid = T0;
      else{
	numh = (int) hull.size();
	hull.at(numh-1).trid = T0;
      }


    }

    /*   
	 char tname[128];
	 sprintf(tname,"rose_%d.mat",k);
	 write_Triads(triads, tname);
	 int dbgb = 0;
    */
    
  }

  //cerr << "of triangles " << triads.size() << " to be flipped. "<< endl;
 
  //  write_Triads(triads, "tris0.mat");

  std::vector<int> ids, ids2;

  int tf = T_flip_pro( pts, triads, slump, numt, 0, ids);
  if( tf < 0 ){
    cerr << "cannot triangulate this set " << endl;
     return(-1);
    //exit(EXIT_FAILURE);
  }

  //  write_Triads(triads, "tris1.mat");

  // cerr << "n-ids " << ids.size() << endl;


  int nits = (int) ids.size(), nit=1;
  int nits_old;
  while(  nits > 0 && nit < 50 ){

    nits_old = nits;
    tf = T_flip_pro_idx( pts, triads, slump, ids, ids2);
    nits = (int) ids2.size(); 
    ids.swap(ids2);

    //cerr << "first flipping cycle  " << nit << "   active triangles " << nits << endl;

    if( nit>14 && nits>nits_old ){
      break;
    }
    
    nit ++;
    if( tf < 0 ){
      cerr << "cannot triangualte this set " << endl;
      return(-1);
      //exit(EXIT_FAILURE);
    }
  }

  ids.clear();
  nits = T_flip_edge( pts, triads, slump, numt, 0, ids);
  nit=0;
  

  while(  nits > 0 && nit < 50 ){

    nits_old = nits;
    tf = T_flip_pro_idx( pts, triads, slump, ids, ids2);
    ids.swap(ids2);
    nits = (int) ids.size();
    //cerr << "second flipping cycle  " << nit << "   active triangles " << nits << endl;

    if( nit>14 && nits>nits_old ){
      break;
    }
    
    nit ++;
    if( tf < 0 ){
      cerr << "cannot triangualte this set " << endl;
      return(-1);
      //exit(EXIT_FAILURE);
    } 
  }
  return(1);
}


void circle_cent4(float r1,float c1, float r2,float c2, float r3,float c3,
		  float &r,float &c, float &ro2){
  /*
   *  function to return the center of a circle and its radius
   * degenerate case should never be passed to this routine!!!!!!!!!!!!!
   * but will return r0 = -1 if it is.
   */
  
  double rd, cd;
  double v1 = 2*(r2-r1), v2 = 2*(c2-c1), v3 = r2*r2 - r1*r1 + c2*c2 - c1*c1;
  double v4 = 2*(r3-r1),
    v5 = 2*(c3-c1),
    v6 = r3*r3 - r1*r1 + c3*c3 - c1*c1,
    
    v7 =  v2*v4 - v1*v5;
  if( v7 == 0 ){   
    r=0;
    c=0;
    ro2 = -1;
    return;
  }
  
  cd = (v4*v3 - v1*v6)/v7;
  if( v1 != 0 )
    rd = (v3 - c*v2)/v1;
  else
    rd = (v6 - c*v5)/v4;
  
  ro2 = (float)  ( (rd-r1)*(rd-r1) + (cd-c1)*(cd-c1) );
  r = (float) rd;
  c = (float) cd;

  return;
}


/* test a set of points for duplicates.

   erase duplicate points, do not change point ids.
 
*/

int de_duplicate( std::vector<Shx> &pts, std::vector<int> &outx ){

  int nump = (int) pts.size();
  std::vector<Dupex> dpx;
  Dupex d;
  for( int k=0; k<nump; k++){
    d.r = pts[k].r;
    d.c = pts[k].c;
    d.id = k;
    dpx.push_back(d);
  }

  sort(dpx.begin(), dpx.end());
  
  for( int k=0; k<nump-1; k++){
    if( dpx[k].r == dpx[k+1].r && dpx[k].c == dpx[k+1].c ){
      //cerr << "duplicate-point ids " << dpx[k].id << "  " << dpx[k+1].id << "   at  ("  << pts[dpx[k+1].id].r << "," << pts[dpx[k+1].id].c << ")" << endl;
      outx.push_back( dpx[k+1].id);
    }
  }

  if( outx.size() == 0 )
    return(0);

  sort(outx.begin(), outx.end());

  int nx = (int) outx.size();
  for( int k=nx-1; k>=0; k--){
    pts.erase(pts.begin()+outx[k]);
  }

  return(nx);
}




/* 
   flip pairs of triangles that are not valid delaunay triangles
   the Cline-Renka test is used rather than the less stable circum 
   circle center computation test of s-hull.

   or the more expensive determinant test.

 */


int T_flip_pro( std::vector<Shx> &pts, std::vector<Triad> &triads, std::vector<int> &slump, int numt, int start, std::vector<int> &ids){

	float r3,c3;
  int pa,pb,pc, pd, D, L1, L2, L3, L4, T2;

  Triad tx, tx2;


  for( int t=start; t<numt; t++){

    Triad &tri = triads[t];
    // test all 3 neighbours of tri 

    int flipped = 0;

    if( tri.bc >= 0 ){

      pa = slump[tri.a];
      pb = slump[tri.b];
      pc = slump[tri.c];

      T2 = tri.bc;
      Triad &t2 = triads[T2];
      // find relative orientation (shared limb).
      if( t2.ab == t ){
	D = t2.c;
	pd = slump[t2.c];

	if( tri.b == t2.a){
	  L3 = t2.ac;
	  L4 = t2.bc;
	}
	else{
	  L3 = t2.bc;
	  L4 = t2.ac;
	}
      }
      else if(  t2.ac == t ){
	D = t2.b;
	pd = slump[t2.b];
	
	if( tri.b == t2.a){
	  L3 = t2.ab;
	  L4 = t2.bc;
	}
	else{
	  L3 = t2.bc;
	  L4 = t2.ab;
	}
      }
      else if(  t2.bc == t ){
	D = t2.a;
	pd = slump[t2.a];
	
	if( tri.b == t2.b){
	  L3 = t2.ab;
	  L4 = t2.ac;
	}
	else{
	  L3 = t2.ac;
	  L4 = t2.ab;
	}
      }
      else{
	cerr << "triangle flipping error. " << t << endl;
	return(-5);
      }


      if( pd < 0 || pd > 100)
	int dfx = 9;

      r3 = pts[pd].r;
      c3 = pts[pd].c;
      
      int XX = Cline_Renka_test( pts[pa].r, pts[pa].c, pts[pb].r, pts[pb].c,  
				  pts[pc].r, pts[pc].c, r3, c3 );
				 
      if( XX < 0 ){ 

	L1 = tri.ab;
	L2 = tri.ac;	
      	if( L1 != L3 && L2 != L4 ){  // need this check for stability.

	tx.a = tri.a;
	tx.b = tri.b;
	tx.c = D;
	
	tx.ab = L1;
	tx.ac = T2;
	tx.bc = L3;
	
	
	// triangle 2;
	tx2.a = tri.a;
	tx2.b = tri.c;
	tx2.c = D;
	
	tx2.ab = L2;
	tx2.ac = t;
	tx2.bc = L4;
	

	ids.push_back(t);
	ids.push_back(T2);
	
	t2 = tx2;
	tri = tx;
	flipped = 1;

	// change knock on triangle labels.
	if( L3 >= 0 ){
	  Triad &t3 = triads[L3];
	  if( t3.ab == T2 ) t3.ab = t;
	  else if( t3.bc == T2 ) t3.bc = t;
	  else if( t3.ac == T2 ) t3.ac = t;
	}

	if(L2 >= 0 ){
	  Triad &t4 = triads[L2];
	  if( t4.ab == t ) t4.ab = T2;
	  else if( t4.bc == t ) t4.bc = T2;
	  else if( t4.ac == t ) t4.ac = T2;
	}
	}
      }
    }


    if(  flipped == 0 && tri.ab >= 0 ){

      pc = slump[tri.c];
      pb = slump[tri.b];
      pa = slump[tri.a];

      T2 = tri.ab;
      Triad &t2 = triads[T2];
      // find relative orientation (shared limb).
      if( t2.ab == t ){
	D = t2.c;
	pd = slump[t2.c];

	if( tri.a == t2.a){
	  L3 = t2.ac;
	  L4 = t2.bc;
	}
	else{
	  L3 = t2.bc;
	  L4 = t2.ac;
	}
      }
      else if(  t2.ac == t ){
	D = t2.b;
	pd = slump[t2.b];
	
	if( tri.a == t2.a){
	  L3 = t2.ab;
	  L4 = t2.bc;
	}
	else{
	  L3 = t2.bc;
	  L4 = t2.ab;
	}
      }
      else if(  t2.bc == t ){
	D = t2.a;
	pd = slump[t2.a];
	
	if( tri.a == t2.b){
	  L3 = t2.ab;
	  L4 = t2.ac;
	}
	else{
	  L3 = t2.ac;
	  L4 = t2.ab;
	}
      }
      else{
	cerr << "triangle flipping error. " << t << endl;
	return(-5);
      }

      r3 = pts[pd].r;
      c3 = pts[pd].c;
      
      int XX = Cline_Renka_test( pts[pc].r, pts[pc].c, pts[pb].r, pts[pb].c,  
				  pts[pa].r, pts[pa].c,r3, c3);

      if( XX < 0){ 
	

	L1 = tri.ac;
	L2 = tri.bc;	
      	if( L1 != L3 && L2 != L4 ){  // need this check for stability.

	tx.a = tri.c;
	tx.b = tri.a;
	tx.c = D;
	
	tx.ab = L1;
	tx.ac = T2;
	tx.bc = L3;
	
	
	// triangle 2;
	tx2.a = tri.c;
	tx2.b = tri.b;
	tx2.c = D;
	
	tx2.ab = L2;
	tx2.ac = t;
	tx2.bc = L4;
	

	ids.push_back(t);
	ids.push_back(T2);
	
	t2 = tx2;
	tri = tx;
	flipped = 1;

	// change knock on triangle labels.
	if( L3 >= 0 ){
	  Triad &t3 = triads[L3];
	  if( t3.ab == T2 ) t3.ab = t;
	  else if( t3.bc == T2 ) t3.bc = t;
	  else if( t3.ac == T2 ) t3.ac = t;
	}

	if(L2 >= 0 ){
	  Triad &t4 = triads[L2];
	  if( t4.ab == t ) t4.ab = T2;
	  else if( t4.bc == t ) t4.bc = T2;
	  else if( t4.ac == t ) t4.ac = T2;
	}

	}

      }
    }


    if( flipped == 0 && tri.ac >= 0 ){

      pc = slump[tri.c];
      pb = slump[tri.b];
      pa = slump[tri.a];

      T2 = tri.ac;
      Triad &t2 = triads[T2];
      // find relative orientation (shared limb).
      if( t2.ab == t ){
	D = t2.c;
	pd = slump[t2.c];

	if( tri.a == t2.a){
	  L3 = t2.ac;
	  L4 = t2.bc;
	}
	else{
	  L3 = t2.bc;
	  L4 = t2.ac;
	}
      }
      else if(  t2.ac == t ){
	D = t2.b;
	pd = slump[t2.b];
	
	if( tri.a == t2.a){
	  L3 = t2.ab;
	  L4 = t2.bc;
	}
	else{
	  L3 = t2.bc;
	  L4 = t2.ab;
	}
      }
      else if(  t2.bc == t ){
	D = t2.a;
	pd = slump[t2.a];
	
	if( tri.a == t2.b){
	  L3 = t2.ab;
	  L4 = t2.ac;
	}
	else{
	  L3 = t2.ac;
	  L4 = t2.ab;
	}
      }
      else{
	cerr << "triangle flipping error. " << t << endl;
	return(-5);
      }

      r3 = pts[pd].r;
      c3 = pts[pd].c;
      
      int XX = Cline_Renka_test( pts[pb].r, pts[pb].c, pts[pa].r, pts[pa].c,  
				  pts[pc].r, pts[pc].c,r3, c3);

      if( XX < 0 ){ 

	L1 = tri.ab;   // .ac shared limb
	L2 = tri.bc;	
      	if( L1 != L3 && L2 != L4 ){  // need this check for stability.

	tx.a = tri.b;
	tx.b = tri.a;
	tx.c = D;
	
	tx.ab = L1;
	tx.ac = T2;
	tx.bc = L3;
	
	
	// triangle 2;
	tx2.a = tri.b;
	tx2.b = tri.c;
	tx2.c = D;
	
	tx2.ab = L2;
	tx2.ac = t;
	tx2.bc = L4;
       
	ids.push_back(t);
	ids.push_back(T2);
	
	t2 = tx2;
	tri = tx;

	// change knock on triangle labels.
	if( L3 >= 0 ){
	  Triad &t3 = triads[L3];
	  if( t3.ab == T2 ) t3.ab = t;
	  else if( t3.bc == T2 ) t3.bc = t;
	  else if( t3.ac == T2 ) t3.ac = t;
	}

	if(L2 >= 0 ){
	  Triad &t4 = triads[L2];
	  if( t4.ab == t ) t4.ab = T2;
	  else if( t4.bc == t ) t4.bc = T2;
	  else if( t4.ac == t ) t4.ac = T2;
	}

	}
      }
    }

   
  }


  return(1);
}
 
/* minimum angle cnatraint for circum circle test.
   due to Cline & Renka

   A   --    B
 
   |    /    |

   C   --    D

   
 */

int Cline_Renka_test(float &Ax, float &Ay,
		      float &Bx, float &By, 
		      float &Cx, float &Cy,
		      float &Dx, float &Dy)
{

  float v1x = Bx-Ax, v1y = By-Ay,    v2x = Cx-Ax, v2y = Cy-Ay,
    v3x = Bx-Dx, v3y = By-Dy,    v4x = Cx-Dx, v4y = Cy-Dy; 
  float cosA = v1x*v2x + v1y*v2y;
  float cosD = v3x*v4x + v3y*v4y;

  if( cosA < 0 && cosD < 0 ) // two obtuse angles 
    return(-1);

  float ADX = Ax-Dx, ADy = Ay-Dy;


  if( cosA > 0 && cosD > 0 )  // two acute angles
    return(1);


  float sinA = fabs(v1x*v2y - v1y*v2x);
  float sinD = fabs(v3x*v4y - v3y*v4x);

  if( cosA*sinD + sinA*cosD < 0 )
    return(-1);

  return(1);

}
  



// same again but with set of triangle ids to be iterated over.


int T_flip_pro_idx( std::vector<Shx> &pts, std::vector<Triad> &triads, std::vector<int> &slump, 
		    std::vector<int> &ids,  std::vector<int> &ids2){

  float  r3,c3;
  int pa,pb,pc, pd,  D, L1, L2, L3, L4, T2;

  Triad tx, tx2;
  ids2.clear();
  //std::vector<int> ids2;

  int numi = ids.size();

  for( int x=0; x<numi; x++){
    int t = ids[x];

    Triad &tri = triads[t];
    // test all 3 neighbours of tri 
    int flipped = 0;

   

    if( tri.bc >= 0 ){

      pa = slump[tri.a];
      pb = slump[tri.b];
      pc = slump[tri.c];

      T2 = tri.bc;
      Triad &t2 = triads[T2];
      // find relative orientation (shared limb).
      if( t2.ab == t ){
	D = t2.c;
	pd = slump[t2.c];

	if( tri.b == t2.a){
	  L3 = t2.ac;
	  L4 = t2.bc;
	}
	else{
	  L3 = t2.bc;
	  L4 = t2.ac;
	}
      }
      else if(  t2.ac == t ){
	D = t2.b;
	pd = slump[t2.b];
	
	if( tri.b == t2.a){
	  L3 = t2.ab;
	  L4 = t2.bc;
	}
	else{
	  L3 = t2.bc;
	  L4 = t2.ab;
	}
      }
      else if(  t2.bc == t ){
	D = t2.a;
	pd = slump[t2.a];
	
	if( tri.b == t2.b){
	  L3 = t2.ab;
	  L4 = t2.ac;
	}
	else{
	  L3 = t2.ac;
	  L4 = t2.ab;
	}
      }
      else{
	cerr << "triangle flipping error. " << t << "  T2: " <<  T2<<  endl;
	return(-6);
      }

      r3 = pts[pd].r;
      c3 = pts[pd].c;    

      int XX = Cline_Renka_test( pts[pa].r, pts[pa].c, pts[pb].r, pts[pb].c,  
				  pts[pc].r, pts[pc].c,r3, c3);

      if( XX < 0 ){ 
	L1 = tri.ab;
	L2 = tri.ac;	

	if( L1 != L3 && L2 != L4 ){  // need this check for stability.
      

	tx.a = tri.a;
	tx.b = tri.b;
	tx.c = D;
	
	tx.ab = L1;
	tx.ac = T2;
	tx.bc = L3;
	
	
	// triangle 2;
	tx2.a = tri.a;
	tx2.b = tri.c;
	tx2.c = D;
	
	tx2.ab = L2;
	tx2.ac = t;
	tx2.bc = L4;
		
	ids2.push_back(t);
	ids2.push_back(T2);
	
	t2 = tx2;
	tri = tx;
	flipped = 1;

	// change knock on triangle labels.
	if( L3 >= 0 ){
	  Triad &t3 = triads[L3];
	  if( t3.ab == T2 ) t3.ab = t;
	  else if( t3.bc == T2 ) t3.bc = t;
	  else if( t3.ac == T2 ) t3.ac = t;
	}

	if(L2 >= 0 ){
	  Triad &t4 = triads[L2];
	  if( t4.ab == t ) t4.ab = T2;
	  else if( t4.bc == t ) t4.bc = T2;
	  else if( t4.ac == t ) t4.ac = T2;
	}

	}
      }
    }


    if( flipped == 0 && tri.ab >= 0 ){

      pc = slump[tri.c];
      pb = slump[tri.b];
      pa = slump[tri.a];

      T2 = tri.ab;
      Triad &t2 = triads[T2];
      // find relative orientation (shared limb).
      if( t2.ab == t ){
	D = t2.c;
	pd = slump[t2.c];

	if( tri.a == t2.a){
	  L3 = t2.ac;
	  L4 = t2.bc;
	}
	else{
	  L3 = t2.bc;
	  L4 = t2.ac;
	}
      }
      else if(  t2.ac == t ){
	D = t2.b;
	pd = slump[t2.b];
	
	if( tri.a == t2.a){
	  L3 = t2.ab;
	  L4 = t2.bc;
	}
	else{
	  L3 = t2.bc;
	  L4 = t2.ab;
	}
      }
      else if(  t2.bc == t ){
	D = t2.a;
	pd = slump[t2.a];
	
	if( tri.a == t2.b){
	  L3 = t2.ab;
	  L4 = t2.ac;
	}
	else{
	  L3 = t2.ac;
	  L4 = t2.ab;
	}
      }
      else{
	cerr << "triangle flipping error. " << t <<  endl;
	return(-6);
      }

      r3 = pts[pd].r;
      c3 = pts[pd].c;   

      int XX = Cline_Renka_test( pts[pc].r, pts[pc].c, pts[pb].r, pts[pb].c,  
				  pts[pa].r, pts[pa].c,r3, c3);

      if( XX < 0 ){ 
	L1 = tri.ac;
	L2 = tri.bc;	
      	if( L1 != L3 && L2 != L4 ){  // need this check for stability.
	
	tx.a = tri.c;
	tx.b = tri.a;
	tx.c = D;
	
	tx.ab = L1;
	tx.ac = T2;
	tx.bc = L3;
	
	
	// triangle 2;
	tx2.a = tri.c;
	tx2.b = tri.b;
	tx2.c = D;
	
	tx2.ab = L2;
	tx2.ac = t;
	tx2.bc = L4;
	
	
	ids2.push_back(t);
	ids2.push_back(T2);
	
	t2 = tx2;
	tri = tx;
	flipped = 1;

	// change knock on triangle labels.
	if( L3 >= 0 ){
	  Triad &t3 = triads[L3];
	  if( t3.ab == T2 ) t3.ab = t;
	  else if( t3.bc == T2 ) t3.bc = t;
	  else if( t3.ac == T2 ) t3.ac = t;
	}

	if(L2 >= 0 ){
	  Triad &t4 = triads[L2];
	  if( t4.ab == t ) t4.ab = T2;
	  else if( t4.bc == t ) t4.bc = T2;
	  else if( t4.ac == t ) t4.ac = T2;
	}

	}
      }
    }


    if( flipped == 0 && tri.ac >= 0 ){

      pc = slump[tri.c];
      pb = slump[tri.b];
      pa = slump[tri.a];

      T2 = tri.ac;
      Triad &t2 = triads[T2];
      // find relative orientation (shared limb).
      if( t2.ab == t ){
	D = t2.c;
	pd = slump[t2.c];

	if( tri.a == t2.a){
	  L3 = t2.ac;
	  L4 = t2.bc;
	}
	else{
	  L3 = t2.bc;
	  L4 = t2.ac;
	}
      }
      else if(  t2.ac == t ){
	D = t2.b;
	pd = slump[t2.b];
	
	if( tri.a == t2.a){
	  L3 = t2.ab;
	  L4 = t2.bc;
	}
	else{
	  L3 = t2.bc;
	  L4 = t2.ab;
	}
      }
      else if(  t2.bc == t ){
	D = t2.a;
	pd = slump[t2.a];
	
	if( tri.a == t2.b){
	  L3 = t2.ab;
	  L4 = t2.ac;
	}
	else{
	  L3 = t2.ac;
	  L4 = t2.ab;
	}
      }
      else{
	cerr << "triangle flipping error. " << t << endl;
	return(-6);
      }

      r3 = pts[pd].r;
      c3 = pts[pd].c;   

      int XX = Cline_Renka_test( pts[pb].r, pts[pb].c, pts[pc].r, pts[pc].c,  
				 pts[pa].r, pts[pa].c,r3, c3);

      if( XX < 0 ){ 
	L1 = tri.ab;   // .ac shared limb
	L2 = tri.bc;	
      	if( L1 != L3 && L2 != L4 ){  // need this check for stability.


	tx.a = tri.b;
	tx.b = tri.a;
	tx.c = D;
	
	tx.ab = L1;
	tx.ac = T2;
	tx.bc = L3;
	
	
	// triangle 2;
	tx2.a = tri.b;
	tx2.b = tri.c;
	tx2.c = D;


	
	tx2.ab = L2;
	tx2.ac = t;
	tx2.bc = L4;


	ids2.push_back(t);
	ids2.push_back(T2);
	
	t2 = tx2;
	tri = tx;
	
	// change knock on triangle labels.
	if( L3 >= 0 ){
	  Triad &t3 = triads[L3];
	  if( t3.ab == T2 ) t3.ab = t;
	  else if( t3.bc == T2 ) t3.bc = t;
	  else if( t3.ac == T2 ) t3.ac = t;
	}
	
	if(L2 >= 0 ){
	  Triad &t4 = triads[L2];
	  if( t4.ab == t ) t4.ab = T2;
	  else if( t4.bc == t ) t4.bc = T2;
	  else if( t4.ac == t ) t4.ac = T2;
	}


	}
      }
    }
  }

  /*
  if( ids2.size() > 5){
    sort(ids2.begin(), ids2.end());
    int nums = ids2.size();
    int last = ids2[0], n=0;
    ids3.push_back(last);
    for(int g=1; g<nums; g++){
      n = ids2[g];
      if( n != last ){
	ids3.push_back(n);
	last = n;
      }
    }
  }
  else{
    int nums = ids2.size();
    for(int g=1; g<nums; g++){
      ids3.push_back(ids2[g]);
    }
    } */


  return(1);
}

/* test the seed configuration to see if the center 
   of the circum circle lies inside the seed triangle.
 
   if not issue a warning.
*/

 
int  test_center(Shx &pt0, Shx &pt1,Shx &pt2){

  float r01 = pt1.r - pt0.r;
  float c01 = pt1.c - pt0.c;

  float r02 = pt2.r - pt0.r;
  float c02 = pt2.c - pt0.c;

  float r21 = pt1.r - pt2.r;
  float c21 = pt1.c - pt2.c;

  float v = r01*r02 + c01*c02;
  if( v < 0 ) return(-1);
  
  v = r21*r02 + c21*c02;
  if( v > 0 ) return(-1);
 
  v = r01*r21 + c01*c21;
  if( v < 0 ) return(-1);
 
  return(1);
}

int de_duplicateX( std::vector<Shx> &pts, std::vector<int> &outx,std::vector<Shx> &pts2 ){

  int nump = (int) pts.size();
  std::vector<Dupex> dpx;
  Dupex d;
  for( int k=0; k<nump; k++){
    d.r = pts[k].r;
    d.c = pts[k].c;
    d.id = k;
    dpx.push_back(d);
  }

  sort(dpx.begin(), dpx.end());
  
  //cerr << "de-duplicating ";
  pts2.clear();
  pts2.push_back(pts[dpx[0].id]);
  pts2[0].id = 0;
  int cnt = 1;
  
  for( int k=0; k<nump-1; k++){
    if( dpx[k].r == dpx[k+1].r && dpx[k].c == dpx[k+1].c ){
      //cerr << "duplicate-point ids " << dpx[k].id << "  " << dpx[k+1].id << "   at  ("  << pts[dpx[k+1].id].r << "," << pts[dpx[k+1].id].c << ")" << endl;
      //cerr << dpx[k+1].id << " ";

      outx.push_back( dpx[k+1].id);
    }
    else{
      pts[dpx[k+1].id].id = cnt;
      pts2.push_back(pts[dpx[k+1].id]);
      cnt++;
    }
  }

  //cerr << "removed  " << outx.size() << endl;

  return(outx.size());
}



int T_flip_edge( std::vector<Shx> &pts, std::vector<Triad> &triads, std::vector<int> &slump, int numt, int start, std::vector<int> &ids){

	float r3,c3;
  int pa,pb,pc, pd, D, L1, L2, L3, L4, T2;

  Triad tx, tx2;


  for( int t=start; t<numt; t++){

    Triad &tri = triads[t];
    // test all 3 neighbours of tri 

    int flipped = 0;

    if( tri.bc >= 0  && (tri.ac < 0 || tri.ab < 0) ){

      pa = slump[tri.a];
      pb = slump[tri.b];
      pc = slump[tri.c];

      T2 = tri.bc;
      Triad &t2 = triads[T2];
      // find relative orientation (shared limb).
      if( t2.ab == t ){
	D = t2.c;
	pd = slump[t2.c];

	if( tri.b == t2.a){
	  L3 = t2.ac;
	  L4 = t2.bc;
	}
	else{
	  L3 = t2.bc;
	  L4 = t2.ac;
	}
      }
      else if(  t2.ac == t ){
	D = t2.b;
	pd = slump[t2.b];
	
	if( tri.b == t2.a){
	  L3 = t2.ab;
	  L4 = t2.bc;
	}
	else{
	  L3 = t2.bc;
	  L4 = t2.ab;
	}
      }
      else if(  t2.bc == t ){
	D = t2.a;
	pd = slump[t2.a];
	
	if( tri.b == t2.b){
	  L3 = t2.ab;
	  L4 = t2.ac;
	}
	else{
	  L3 = t2.ac;
	  L4 = t2.ab;
	}
      }
      else{
	cerr << "triangle flipping error. " << t << endl;
	return(-5);
      }


      if( pd < 0 || pd > 100)
	int dfx = 9;

      r3 = pts[pd].r;
      c3 = pts[pd].c;
      
      int XX = Cline_Renka_test( pts[pa].r, pts[pa].c, pts[pb].r, pts[pb].c,  
				  pts[pc].r, pts[pc].c, r3, c3 );
				 
      if( XX < 0 ){ 

	L1 = tri.ab;
	L2 = tri.ac;	
	//	if( L1 != L3 && L2 != L4 ){  // need this check for stability.

	tx.a = tri.a;
	tx.b = tri.b;
	tx.c = D;
	
	tx.ab = L1;
	tx.ac = T2;
	tx.bc = L3;
	
	
	// triangle 2;
	tx2.a = tri.a;
	tx2.b = tri.c;
	tx2.c = D;
	
	tx2.ab = L2;
	tx2.ac = t;
	tx2.bc = L4;
	

	ids.push_back(t);
	ids.push_back(T2);
	
	t2 = tx2;
	tri = tx;
	flipped = 1;

	// change knock on triangle labels.
	if( L3 >= 0 ){
	  Triad &t3 = triads[L3];
	  if( t3.ab == T2 ) t3.ab = t;
	  else if( t3.bc == T2 ) t3.bc = t;
	  else if( t3.ac == T2 ) t3.ac = t;
	}

	if(L2 >= 0 ){
	  Triad &t4 = triads[L2];
	  if( t4.ab == t ) t4.ab = T2;
	  else if( t4.bc == t ) t4.bc = T2;
	  else if( t4.ac == t ) t4.ac = T2;
	}
	//	}
      }
    }


    if(  flipped == 0 && tri.ab >= 0  && (tri.ac < 0 || tri.bc < 0)){

      pc = slump[tri.c];
      pb = slump[tri.b];
      pa = slump[tri.a];

      T2 = tri.ab;
      Triad &t2 = triads[T2];
      // find relative orientation (shared limb).
      if( t2.ab == t ){
	D = t2.c;
	pd = slump[t2.c];

	if( tri.a == t2.a){
	  L3 = t2.ac;
	  L4 = t2.bc;
	}
	else{
	  L3 = t2.bc;
	  L4 = t2.ac;
	}
      }
      else if(  t2.ac == t ){
	D = t2.b;
	pd = slump[t2.b];
	
	if( tri.a == t2.a){
	  L3 = t2.ab;
	  L4 = t2.bc;
	}
	else{
	  L3 = t2.bc;
	  L4 = t2.ab;
	}
      }
      else if(  t2.bc == t ){
	D = t2.a;
	pd = slump[t2.a];
	
	if( tri.a == t2.b){
	  L3 = t2.ab;
	  L4 = t2.ac;
	}
	else{
	  L3 = t2.ac;
	  L4 = t2.ab;
	}
      }
      else{
	cerr << "triangle flipping error. " << t << endl;
	return(-5);
      }

      r3 = pts[pd].r;
      c3 = pts[pd].c;
      
      int XX = Cline_Renka_test( pts[pc].r, pts[pc].c, pts[pb].r, pts[pb].c,  
				  pts[pa].r, pts[pa].c,r3, c3);

      if( XX < 0){ 
	

	L1 = tri.ac;
	L2 = tri.bc;	
	//	if( L1 != L3 && L2 != L4 ){  // need this check for stability.

	tx.a = tri.c;
	tx.b = tri.a;
	tx.c = D;
	
	tx.ab = L1;
	tx.ac = T2;
	tx.bc = L3;
	
	
	// triangle 2;
	tx2.a = tri.c;
	tx2.b = tri.b;
	tx2.c = D;
	
	tx2.ab = L2;
	tx2.ac = t;
	tx2.bc = L4;
	

	ids.push_back(t);
	ids.push_back(T2);
	
	t2 = tx2;
	tri = tx;
	flipped = 1;

	// change knock on triangle labels.
	if( L3 >= 0 ){
	  Triad &t3 = triads[L3];
	  if( t3.ab == T2 ) t3.ab = t;
	  else if( t3.bc == T2 ) t3.bc = t;
	  else if( t3.ac == T2 ) t3.ac = t;
	}

	if(L2 >= 0 ){
	  Triad &t4 = triads[L2];
	  if( t4.ab == t ) t4.ab = T2;
	  else if( t4.bc == t ) t4.bc = T2;
	  else if( t4.ac == t ) t4.ac = T2;
	}

	//	}

      }
    }


    if( flipped == 0 && tri.ac >= 0  && (tri.bc < 0 || tri.ab < 0) ){

      pc = slump[tri.c];
      pb = slump[tri.b];
      pa = slump[tri.a];

      T2 = tri.ac;
      Triad &t2 = triads[T2];
      // find relative orientation (shared limb).
      if( t2.ab == t ){
	D = t2.c;
	pd = slump[t2.c];

	if( tri.a == t2.a){
	  L3 = t2.ac;
	  L4 = t2.bc;
	}
	else{
	  L3 = t2.bc;
	  L4 = t2.ac;
	}
      }
      else if(  t2.ac == t ){
	D = t2.b;
	pd = slump[t2.b];
	
	if( tri.a == t2.a){
	  L3 = t2.ab;
	  L4 = t2.bc;
	}
	else{
	  L3 = t2.bc;
	  L4 = t2.ab;
	}
      }
      else if(  t2.bc == t ){
	D = t2.a;
	pd = slump[t2.a];
	
	if( tri.a == t2.b){
	  L3 = t2.ab;
	  L4 = t2.ac;
	}
	else{
	  L3 = t2.ac;
	  L4 = t2.ab;
	}
      }
      else{
	cerr << "triangle flipping error. " << t << endl;
	return(-5);
      }

      r3 = pts[pd].r;
      c3 = pts[pd].c;
      
      int XX = Cline_Renka_test( pts[pb].r, pts[pb].c, pts[pa].r, pts[pa].c,  
				  pts[pc].r, pts[pc].c,r3, c3);

      if( XX < 0 ){ 

	L1 = tri.ab;   // .ac shared limb
	L2 = tri.bc;	
	//	if( L1 != L3 && L2 != L4 ){  // need this check for stability.

	tx.a = tri.b;
	tx.b = tri.a;
	tx.c = D;
	
	tx.ab = L1;
	tx.ac = T2;
	tx.bc = L3;
	
	
	// triangle 2;
	tx2.a = tri.b;
	tx2.b = tri.c;
	tx2.c = D;
	
	tx2.ab = L2;
	tx2.ac = t;
	tx2.bc = L4;
       
	ids.push_back(t);
	ids.push_back(T2);
	
	t2 = tx2;
	tri = tx;

	// change knock on triangle labels.
	if( L3 >= 0 ){
	  Triad &t3 = triads[L3];
	  if( t3.ab == T2 ) t3.ab = t;
	  else if( t3.bc == T2 ) t3.bc = t;
	  else if( t3.ac == T2 ) t3.ac = t;
	}

	if(L2 >= 0 ){
	  Triad &t4 = triads[L2];
	  if( t4.ab == t ) t4.ab = T2;
	  else if( t4.bc == t ) t4.bc = T2;
	  else if( t4.ac == t ) t4.ac = T2;
	}

	//}
      }
    }

   
  }


  return(1);
}
 
