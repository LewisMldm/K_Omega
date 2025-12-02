D = 0.1;
x=1;
density_inlet_h = 101*x;
density_inlet_v = 81*x;
density_outlet_h = 151*x;
density_outlet_v = 41*x;

Point(1) = {    0,   0, 0};
Point(2) = {-10*D,   0, 0};
Point(3) = {-10*D, 5*D, 0};
Point(4) = {    0, 5*D, 0};
Point(5) = { 30*D, 5*D, 0};
Point(6) = { 30*D,   0, 0};
Point(7) = {    0,  -D, 0};
Point(8) = { 30*D,  -D, 0};

Line(1) = {2, 3};
Line(2) = {3, 4};
Line(3) = {4, 1};
Line(4) = {1, 2};
Line(5) = {4, 5};
Line(6) = {5, 6};
Line(7) = {6, 1};
Line(8) = {1, 7};
Line(9) = {7, 8};
Line(10) = {8, 6};

Line(11) = {5, 8};

Line Loop(12) = {1, 2, 3, 4};
Plane Surface(13) = {12};
Line Loop(14) = {5, 6, 7, -3};
Plane Surface(15) = {14};
Line Loop(16) = {9, 10, 7, 8};
Plane Surface(17) = {16};

Transfinite Line {-2, 4} = density_inlet_h Using Progression 1.02;
Transfinite Line {1, 3, -6} = density_inlet_v Using Bump 0.038;
Transfinite Line {5, -7, 9} = density_outlet_h Using Progression 1.018;
Transfinite Line {8, 10} = density_outlet_v Using Bump 0.038;
Transfinite Surface "*";
Recombine Surface "*";


Physical Line("inlet", 17) = {1};
Physical Line("outlet", 18) = {11};
Physical Line("noslip", 19) = {2, 5, 4, 8, 9};
Physical Surface("domain", 18) = {13, 15, 17};
