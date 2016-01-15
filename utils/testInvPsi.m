

xx=-10:.01:10; 
y=invpsi(xx);
plot(xx,y);
xcheck=psi(y); 
hold on
plot(xcheck,y,'r');