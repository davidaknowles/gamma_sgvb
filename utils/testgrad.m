function testgrad(logL,tests)
    D=length(tests);
    [l0,ag]=logL(tests); 
    ndeps=1e-6; 
    for d=1:D
       temp=tests;
       temp(d)=temp(d)+ndeps; 
       nd= ( logL(temp) - l0 ) / ndeps; 
%        fprintf(1,'%f=%f\n',nd,ag(d));
       assert( abs(nd - ag(d) )/max(abs(nd),.001) < .01 ); 
    end