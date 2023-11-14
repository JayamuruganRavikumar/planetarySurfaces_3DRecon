function Rot = GetrotMatrix(w, phi,k)
w = deg2rad(w);
phi = deg2rad(phi);
k  = deg2rad(k);
R = zeros(3,3);

R(1, 1) = cos(phi)*cos(k);
R(1, 2) = (sin(w) * sin(phi)*cos(k)) + cos(w)*sin(k);
R(1, 3) = (-cos(w)*sin(phi)*cos(k)) + (sin(w) *sin(k));
R(2, 1) = - cos(phi) *sin(k);
R(2, 2) = (- sin(w) * sin(phi) * sin(k) )+ cos(w) * cos(k);
R(2, 3) = (cos(w)*sin(phi) * sin(k)) + sin(w) * cos(k);
R(3,1) = sin(phi);
R(3, 2) = - sin(w) * cos(phi);
R(3,3) = cos(w)*cos(phi);
Rot = R;
