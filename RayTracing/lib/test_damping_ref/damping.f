c     PROGRAM DAMPING.f
c     CALCULATE NUMERICALLY THE RELATIVISTIC ELECTRON CYCLOTRON DAMPING
C     WITH AN ARBITRARY DISTRIBUTION FUNCTION AT AN ARBITRARY HARMONIC
C     AND THE RELATIVISTIC LANDAU DAMPING
      IMPLICIT REAL*8 (A-H,O-Z)
      PARAMETER(ITEST=0,NH=2,NB=4)
      DOUBLE PRECISION DD(653,653),DE(1303),DF(1303)           
      DOUBLE PRECISION XCC(653),YCC(653),BD(1303,653),BX(1303,653),   
     $D1(1303,653),DIND(1303),E(1303),F(1303),F5(1303,653)
      DOUBLE PRECISION DFDX(1303,653),DFDY(1303,653)
      REAL*4 VYPERP(653),VXPARA(1303)
      INTEGER NZ,IFAIL
      COMPLEX*16 Z,CY(NB)
      CHARACTER*1 SCALE,SCALE1
      EXTERNAL S17DEF,S18DCF,S18DEF
      SCALE='U'
      SCALE1='S'
C     WE SET NB=4. FOUR BESSEL FUNCTIONS ARE USED IN THE SUMMATION
C     HN=1. FOR HARMONIC NH=1 OR HN=2. FOR HARMONIC NH=2 etc...
      HN=FLOAT(NH)
C     ITEST=1 FOR A RELATIVISTIC MAXWELLIAN
C     ITEST=0 FOR A DISTRIBUTION FUNCTION CALCULATED FROM FOKKER-PLANCK CODE
      IF(ITEST.EQ.1) WRITE(6,*) 'RELATIVISTIC MAXWELLIAN USED'
      IF(ITEST.EQ.0) WRITE(6,*) 'DISTRIBUTION FUNCTION FROM F-P CODE'
      NZ=4
      TE=4.0                                                          
C     OMP2=OMEGAP**2/OMEGA**2
      OMP2=1.
      WRITE(6,*) 'ALL EPS NORMALIZED TO OMEGAP**2/OMEGA**2 =' ,OMP2
      WRITE(6,*) 'HARMONIC N=', HN
      WRITE(6,*) 'TEMPERATURE IN keV=', TE
C     OMC=N*OMEGAC/OMEGA
      OMC=0.95
      WRITE(6,*) 'N*OMEGAC/OMEGA = ', OMC
C     Npar=PARN, either positive or negative
      PARN=0.6
      WRITE(6,*) 'Npar= ', PARN
C     Nper=PERPN
      PERPN=0.4
      WRITE(6,*) 'Nperp= ', PERPN
C     OMCLH=OMEGAC/OMEGA FOR LH WAVE
      OMCLH=20.0
      WRITE(6,*) 'LOWER HYBRID OMEGAC/OMEGA = ', OMCLH
C     Npar FOR LH WAVE, either positive or negative, in magnitude > 1
      PARNLH=2.50
      WRITE(6,*) 'LOWER HYBRID Npar= ', PARNLH
C     Nperp for LH WAVE =PERPLH
      PERPLH=1.4
      WRITE(6,*) 'LOWER HYBRID Nperp= ', PERPLH
      BTH=DSQRT(TE)*0.044237436D0
      BTH2=BTH*BTH
      PARN2=PARN*PARN
      OMC2=OMC*OMC 
      CNST=PARN2+OMC2-1.0D0
      IF(CNST.LE.0.0) WRITE(6,*) 'DAMPING OF EC WAVE IMPOSSIBLE'
      IF(CNST.LE.0.0) STOP
      PI=3.141592653589D0
      PINIT=0.D0 
C     FOR ITEST=0, ADJUST PMAX, N1 AND N2  ACCORDING TO THE VALUE OF
C     TE AND THE INPUT DD MATRIX BELOW.  FOR ITEST=1, PMAX CAN BE
C     REDUCED, BUT CUTTING IT TOO SHORT CAN ELIMINATE CONTRIBUTIONS
C     FROM STEEP GRADIENTS. IN THIS CASE, N1 MAY HAVE TO BE INCREASED.
      PMAX=30.0D0                                                        
      N1=400
      IF (ITEST.EQ.1) PMAX=15.0D0
      IF (ITEST.EQ.1) N1=500
C     DISTRIBUTION FUNCTION IN SPHERICAL COORDINATES DD(I,J) IS CALCULATED
C     FROM A FOKKER-PLANCK CODE ON A GRID 400x200.  TAKE N2=200-2
      N2=198
      NX=N1                                                            
      NY=N2                                                             
      DMU=2.D0/FLOAT(NY-1)
      DMIN=-1.0D0
      DV=PMAX/FLOAT(NX-1)
      IF(ITEST.EQ.1) GO TO 21
C     Read the distribution function calculated from the Fokker-Planck code
      OPEN(UNIT=11,FILE='distr.dat',FORM='FORMATTED')
      READ(11,*) ((DD(I,J),I=1,NX),J=1,NY)
      CLOSE(11)
      DO 2 J=1,NY                                                       
      DO 2 I=1,NX                                                       
       IF(DD(I,J).LE.0.D0) DD(I,J)=1.D-30
    2  CONTINUE                   
   21 CONTINUE
      NXM1=NX-1                                                         
C     Grid for the spherical coordinate system (p, cos(theta))
      DO 1000 J=1,NY                                                    
 1000  YCC(J)=DMIN+(J-1)*DMU                                             
      DO 1001 I=1,NX                                                    
 1001  XCC(I)=(I-1)*DV+PINIT                                             
      XARG=1.0D0/BTH2
      Z=DCMPLX(XARG,0.0D0)
      FNU=1.0D0
C     CALCULATE BESSEL FUNCTION FROM THE NAG SCIENTIFIC LIBRARY
      CALL S18DCF(FNU,Z,NB,SCALE1,CY,NZ,IFAIL)
      KN=2
      CYKE=DREAL(CY(KN))
      IF(ITEST.EQ.0) GO TO 24
      DO 33 I=1,NX
      DO 33 J=1,NY
       GAMA=DSQRT(1.0D0+XCC(I)*XCC(I)*BTH2)
C      Relativistic Maxwellian distribution
       DD(I,J)=BTH2*BTH*DEXP((1.0D0-GAMA)/BTH2)/(4.D0*PI*CYKE*BTH2)
   33  CONTINUE
   24 CONTINUE
      E(1)=0.D0                                                           
      F(1)=0.D0                                                           
      DO 505 I=2,NX    
       E(I)=-1.0D0/(4.D0+E(I-1)) 
  505  CONTINUE       
C     Calculate derivative of the distribution function using cubic spline
      DO 500 J=1,NY     
       DO 501 I=1,NX  
  501   DIND(I)=DD(I,J) 
       DO 502 I=2,NXM1
  502   F(I)=(F(I-1)-3.D0*(DIND(I+1)-DIND(I-1))/DV)*E(I)
       D1(NX,J)=0.D0 
       DO 503 I=1,NXM1 
        K=NX-I  
  503   D1(K,J)=E(K)*D1(K+1,J)+F(K)                                       
  500  CONTINUE                                                          
C     Calculate density and current
      DINT=0.D0                                                           
      DINTT=0.D0                                                          
      DO 10 I=1,NX                                                      
       PL=(I-1)*DV+PINIT                                                 
       PL2=PL*PL
       PL3=PL2*PL
       GAMA=DSQRT(1.0D0+PL2*BTH2)                                       
       DO 10 J=1,NY                                                      
       YC= DMIN+(J-1)*DMU                                             
       DINT=DINT+DD(I,J)*DMU*DV*PL2                                  
       BD(I,J)=DD(I,J)*PL2                                           
       BX(I,J)=DD(I,J)*YC*PL3/GAMA                                   
       DINTT=DINTT+DD(I,J)*DMU*DV*YC*PL3/GAMA                        
   10  CONTINUE                                                          
      DINT=DINT*2.D0*PI
      DINTT=DINTT*2.D0*PI
      WRITE(6,*) ' '
      PRINT 23                                                          
   23 FORMAT(4X,17HDENSITY   CURRENT)                                     
      PRINT 22,DINT,DINTT 
   22 FORMAT(2X,8G14.7)
      V1=PINIT                                                          
      XMA=PMAX                                                          
      NNY=NX                                                            
      N=NX                                                              
      NNX=2*NX-1
      XMI=-XMA                                                          
      DO 1340 I=1,NX                                                    
      DO 1340 J=1,NY                                                    
 1340  F5(I,NY-J+1)=DD(I,J)                                              
      DO 1344 I=1,NX                                                    
      DO 1344 J=1,NY                                                    
 1344  DD(I,J)=F5(I,J)                                                   
      DO 1339 IX=1,NNX                                                  
      DO 1339 IY=1,NNY                                                  
 1339  F5(IX,IY)=0.D0 
      YMA=XMA  
      YMI=0.D0                                                            
      DX=(XMA-XMI)/(NNX-1)                                              
      DY=(YMA-YMI)/(NNY-1)                                              
      XX=XMI-DX                                                         
C     Calculate the distribution function F5(I,J) in a
C     cylindrical coordinate (Ppar=VXPARA,Pperp=VYPERP)
      DO 5 JX=1,NNX                                                     
       XX=XX+DX                                                          
       YY=YMI-DY                                                         
       DO 5 JY=1,NNY                                                     
       YY=YY+DY                                                          
       R=DSQRT(XX*XX+YY*YY)                                               
       IF(R.EQ.0.) GO TO 5                                               
       TH=ACOS(XX/R)
       W=1.0D0
       THP=-DACOS(W)                                                     
       DO 6 JMU=2,NY      
        W=1.0D0-(JMU-1)*DMU
        THA=DACOS(W)     
        IF(THP.GT.TH.OR.THA.LT.TH) GO TO 6
        RP=V1                                                             
        DO 3 JV=2,N                                                       
         RA=(JV-1.0D0)*DV +V1                                             
         IF(RP.LE.R.AND.RA.GE.R) GO TO 100
    3    RP=RA
        GO TO 5
    6   THP=THA                                                           
       JMU=NY                                                            
       THA=PI+DACOS(DMU/2.D0)                                  
       IF(THP.GT.TH.OR.THA.LT.TH) GO TO 101                               
       RP=V1                                                             
       DO 4 JV=2,N                                                       
        RA=(JV-1.0D0)*DV +V1                                     
        IF(RP.LE.R.AND.RA.GE.R) GO TO 100
    4   RP=RA                                                             
  101  CONTINUE                                                      
       GO TO 5                                     
  100  UV1=1.0D0-(JMU-1.0D0-1.0D0)*DMU
       UV2=1.0D0-(JMU-1.0D0)*DMU 
       FAC=(DCOS(TH)-UV1)/(UV2-UV1)
       B1=DD(JV-1,JMU-1)+(DD(JV-1,JMU)-DD(JV-1,JMU-1))*FAC               
       B2=DD(JV,JMU-1)+(DD(JV,JMU)-DD(JV,JMU-1))*FAC                     
       F5(JX,JY)=B1+(B2-B1)*(R-RP)/(RA-RP)                               
    5  CONTINUE                                                          
      DO 1342 IY=1,NNY
       VPERP=(FLOAT(IY)-1.0D0)*DY
       VYPERP(IY)=VPERP
 1342  CONTINUE
      DINT=0.D0
      DO I=1,NNX
      DO J=1,NNY
       DINT=DINT+F5(I,J)*VYPERP(J)*DX*DY*2.D0*PI
       ENDDO
       ENDDO
      N1=NNY                                                            
      DO I=1,NNX
       VXPARA(I)=(2.D0*FLOAT(I-1)/FLOAT(NNX-1)-1.0D0)*PMAX
       ENDDO
      OPEN(UNIT=3,FILE='contour.dat')
      WRITE(3,*) 1
      WRITE(3,*) NNX,NNY
      WRITE(3,*) (SNGL(VXPARA(I)),I=1,NNX)
      WRITE(3,*) (SNGL(VYPERP(I)),I=1,NNY)
      WRITE(3,*) ((SNGL(F5(I,J)),I=1,NNX),J=1,NNY)
      CLOSE(3)
      NNX1=NNX-1                                                        
      NNY1=NNY-1
C     CALCULATE DERIVATIVE OF DISTRIBUTION FUNCTION USING CUBIC SPLINES
      DO 506 J=1,NNY
       DE(1)=0.D0
       DF(1)=0.D0
       DO 507 I=2,NNX1
        DFI=3.D0*(F5(I+1,J)-F5(I-1,J))/DX
        DE(I)=-1.0D0/(DE(I-1)+4.D0)
  507   DF(I)=(DF(I-1)-DFI)*DE(I)
       DFDX(NNX,J)=0.0D0
       DO 508 I=1,NNX1
        K=NNX-I
  508   DFDX(K,J)=DE(K)*DFDX(K+1,J)+DF(K)
       IF(F5(I,J).LE.0.) DFDX(I,J)=0.D0
  506  CONTINUE
      DO 526 I=1,NNX
       DE(1)=0.D0
       DF(1)=0.D0
       DO 527 J=2,NNY1
        DFI=3.D0*(F5(I,J+1)-F5(I,J-1))/DY
        DE(J)=-1.0D0/(DE(J-1)+4.D0)
  527   DF(J)=(DF(J-1)-DFI)*DE(J)
       DFDY(I,NNY)=0.0D0
       DO 528 J=1,NNY1
        K=NNY-J
  528   DFDY(I,K)=DE(K)*DFDY(I,K+1)+DF(K)
       IF(F5(I,J).LE.0.) DFDY(I,J)=0.0D0
  526  CONTINUE
C     CALCULATION OF DERIVATIVES FOR A MAXWELLIAN DISTRIBUTION FUNCTION

C     OPTION FOR ANALYTICAL CALCULATION OF DERIVATIVES FOR A MAXWELLIAN 
C     DISTRIBUTION FUNCTION
C     DO I=1,NNX
C     DO J=1,NNY
C      GAMA=DSQRT(1.0D0+VXPARA(I)*VXPARA(I)*BTH2+
C    1 VYPERP(J)*VYPERP(J)*BTH2)
CC     GAMA=DSQRT(1.0D0+XCC(I)*XCC(I)*BTH2)
C     F5(I,J)=BTH2*BTH*DEXP((1.0D0-GAMA)/BTH2)/(4.D0*PI*CYKE*BTH2)
C      CONS1=-F5(I,J)*VYPERP(J)/GAMA
C      DFDY(I,J)=CONS1
C      CONS2=-F5(I,J)*VXPARA(I)/GAMA
C      DFDX(I,J)=CONS2
C      WRITE(6,*) I,J,CONS1,DFDY(I,J),CONS2,DFDX(I,J)
C      WRITE(6,*) F5(I,J)
C      ENDDO
C      ENDDO
C     IF LH LANDAU DAMPING ONLY IS NEEDED GO TO 300
C     GO TO 300
      IF(PARN2.NE.1.0) GO TO 301
      ZM=(OMC-1.0D0/OMC)/2.0D0
      ZP=-ZM
      GO TO 302
C     From Eq.(3)
  301 ZM=PARN*OMC/(1.0D0-PARN2)-DSQRT(CNST)/DABS(1.0D0-PARN2) 
      ZP=PARN*OMC/(1.0D0-PARN2)+DSQRT(CNST)/DABS(1.0D0-PARN2)
  302 IM=1
      IP=NNX
      DO I=1,NNX1
       PMBTH=VXPARA(I)*BTH
       PMBTH1=VXPARA(I+1)*BTH
       IF(PMBTH.LT.ZM.AND.PMBTH1.GT.ZM) IM=I
       IF((PMBTH.EQ.ZM).AND.(PARN2.GE.1.0)) IM=I
       IF((PMBTH1.EQ.ZM).AND.(PARN2.LT.1.0)) IM=I
       PPBTH=VXPARA(I)*BTH
       PPBTH1=VXPARA(I+1)*BTH
       IF(PPBTH.LT.ZP.AND.PPBTH1.GT.ZP) IP=I
       IF((PPBTH1.EQ.ZP).AND.(PARN2.GE.1.0)) IP=I
       IF((PPBTH.EQ.ZP).AND.(PARN2.LT.1.0)) IP=I
       ENDDO
      DINT11=0.D0
      DINT12=0.D0
      DINT13=0.D0
      DINT21=0.D0
      DINT22=0.D0
      DINT23=0.D0
      DINT31=0.D0
      DINT32=0.D0
      DINT33=0.D0
      DINT111=0.D0
      FACT=1
      DO I=1,NH
       FACT=FACT*I
       ENDDO
      FNU=HN-1.0D0
      IF (PARN2.GE.1.) GO TO 805
      II=IM+1
      IF=IP
      I11=II
      I22=IF
      DZ1=(VXPARA(IM+1)-ZM/BTH)/DX
      DZ2=(ZP/BTH-VXPARA(IP))/DX
      GO TO 806
  805 CONTINUE
      IF (PARN.LE.-1) GO TO 807
      II=IP+1
      IF=NNX1
      I11=II
      I22=0
      DZ1=(VXPARA(IP+1)-ZP/BTH)/DX
      DZ2=0.D0
      GO TO 806
  807 CONTINUE
      II=1
      IF=IM
      I22=IF
      I11=0
      DZ2=(ZM/BTH-VXPARA(IM))/DX
      DZ1=0.D0
  806 CONTINUE
      ZMAX=PMAX*BTH
      DO 600 I=II,IF
       ZPAR=VXPARA(I)*BTH
       ZARG=(PARN2-1.0D0)*ZPAR*ZPAR+2.0D0*PARN*OMC*ZPAR+OMC2-1.0D0
       IF(ZARG.LE.0.0) GO TO 600
       ZPERP=DSQRT(ZARG)
       IF(ZPERP.GE.ZMAX) GO TO 600
       XARG1=PERPN*ZPERP*HN/OMC
       Z=DCMPLX(XARG1,0.0D0)
C     CALCULATE BESSEL FUNCTION FROM THE NAG SCIENTIFIC LIBRARY
       CALL S17DEF(FNU,Z,NB,SCALE,CY,NZ,IFAIL)
       CY1=DREAL(CY(1))
       CY2=DREAL(CY(2))
CC     SMALL ARGUMENT EXPANSION OF THE BESSEL FUNCTIONS
       CY22=(XARG1*0.5D0)**HN/FACT 
       CY3=DREAL(CY(3))
       CY13=CY1-CY3
       PPERP=ZPERP/BTH
       IPERP=PPERP/DY+1
       V11=(HN*CY2/XARG1)**2
       V111=(HN*CY22/XARG1)**2
       V12=HN*CY2*CY13/(2.D0*XARG1)
       V13=(ZPAR/ZPERP)*HN*CY2*CY2/XARG1
       V22=CY13*CY13/4.D0
       V32=-(ZPAR/ZPERP)*CY2*(CY1-CY3)/2.D0
       V33=((ZPAR/ZPERP)*CY2)**2
       DDYY=(PPERP-VYPERP(IPERP))*DFDY(I,IPERP+1)/DY
     1 +(VYPERP(IPERP+1)-PPERP)*DFDY(I,IPERP)/DY
       DDXX=(PPERP-VYPERP(IPERP))*DFDX(I,IPERP+1)/DY
     1 +(VYPERP(IPERP+1)-PPERP)*DFDX(I,IPERP)/DY
       DERIVF=DDYY*OMC+DDXX*ZPERP*PARN
       IF(I.NE.I11) GO TO 800
       V11=V11*0.5D0*(1.0D0+DZ1)
       V111=V111*0.5D0*(1.0D0+DZ1)
       V12=V12*0.5D0*(1.0D0+DZ1)
       V13=V13*0.5D0*(1.0D0+DZ1)
       V22=V22*0.5D0*(1.0D0+DZ1)
       V32=V32*0.5D0*(1.0D0+DZ1)
       V33=V33*0.5D0*(1.0D0+DZ1)
  800  IF(I.NE.I22) GO TO 801
       V11=V11*0.5D0*(1.0D0+DZ2)
       V111=V111*0.5D0*(1.0D0+DZ2)
       V12=V12*0.5D0*(1.0D0+DZ2)
       V13=V13*0.5D0*(1.0D0+DZ2)
       V22=V22*0.5D0*(1.0D0+DZ2)
       V32=V32*0.5D0*(1.0D0+DZ2)
       V33=V33*0.5D0*(1.0D0+DZ2)
  801  CONTINUE
       DINT11=DINT11+ZPERP*V11*DERIVF
       DINT111=DINT111+ZPERP*V111*DERIVF
       DINT12=DINT12-ZPERP*V12*DERIVF
       DINT13=DINT13+ZPERP*V13*DERIVF
       DINT22=DINT22-ZPERP*V22*DERIVF
       DINT32=DINT32+ZPERP*V32*DERIVF
       DINT33=DINT33+ZPERP*V33*DERIVF
  600  CONTINUE
      CONST= OMP2*DX*2.D0*PI*PI/BTH**3
      DINT11=-DINT11*CONST
      DINT111=-OMP2*DINT111*DX*2.D0*PI*PI/BTH**3
C     EXACT AND SMALL ARGUMENT EXPANSION
      WRITE(6,*) ' '
      WRITE(6,*) '  RESULTS FROM THE NUMERICAL CALCULATION'
      WRITE(6,*) 'IMAGINARY EPS11',DINT11
      WRITE(6,*) 'SMALL ARGUMENT EXPANSION: IMAGINARY EPS11',DINT111
      DINT12=DINT12* CONST
      WRITE(6,*) 'EPS12',DINT12
      WRITE(6,*) 'EPS21=-EPS12'
      DINT22=DINT22* CONST
      WRITE(6,*) 'IMAGINARY EPS22',DINT22
      DINT13=-DINT13* CONST
      WRITE(6,*) 'IMAGINARY EPS13',DINT13
      WRITE(6,*) 'EPS31=EPS13'
      DINT32=DINT32* CONST
      WRITE(6,*) 'EPS32',DINT32
      WRITE(6,*) 'EPS23=-EPS32'
      DINT33=-DINT33* CONST
      WRITE(6,*) 'IMAGINARY EPS33',DINT33
C     START OF CALCULATIONS FOR A RELATIVISTIC MAXWELLIAN
      AN1=1.0D0/(FACT*(2.D0**HN))
      CONS5=(PERPN*PERPN*HN*HN*BTH2/OMC2)**(HN-1.0D0)
      IF(PARN2.EQ.1.0) GO TO 607
      CONS1=DSQRT(PI*BTH2/2.D0)
      CONS3=CNST**(HN/2.D0+0.25D0)
      IF(PARN.NE.0.) CONS3=CONS3/(PARN2**(HN/2.D0+0.25D0))
      CONS4=DEXP((PARN2-1.0D0+OMC-DABS(PARN)*DSQRT(CNST))
     1  /((PARN2-1.0D0)*BTH2))
      EPSQ=AN1*(CONS1/CYKE)*CONS3*CONS4*CONS5/BTH2
      IF(PARN.EQ.0.0) GO TO 609
      FNU=HN+0.5D0
      IF (PARN2.LT.1.) GO TO 602
      XARG2=DABS(PARN)*DSQRT(CNST)/(BTH2*(PARN2-1.0D0))
      Z=DCMPLX(XARG2,0.0D0)
C     CALCULATE BESSEL FUNCTION FROM THE NAG SCIENTIFIC LIBRARY
      CALL S18DCF(FNU,Z,NB,SCALE1,CY,NZ,IFAIL)
      EPSQ=OMP2*EPSQ/DSQRT(PARN2-1.0D0)
      IF (PARN2.GT.1.) GO TO 603
  602 CONTINUE
      XARG2=DABS(PARN)*DSQRT(CNST)/(BTH2*(1.0D0-PARN2))
      Z=DCMPLX(XARG2,0.0D0)
C     CALCULATE BESSEL FUNCTION FROM THE NAG SCIENTIFIC LIBRARY
      CALL S18DEF(FNU,Z,NB,SCALE1,CY,NZ,IFAIL)
      EPSQ=PI*OMP2*EPSQ/DSQRT(1.0D0-PARN2)
  603 CONTINUE
      CY3E=DREAL(CY(1))
      CY4E=DREAL(CY(2))
      EPS11=EPSQ*CY3E*HN*HN
      EPS13=EPSQ*PERPN*HN*PARN*HN/(OMC*(1.0D0-PARN2))
      EPS13=EPS13*(OMC*CY3E-DSQRT(CNST/PARN2)*CY4E)
      EPS33=EPSQ*PERPN*PERPN*HN*HN/OMC2
      EPS33=EPS33*((PARN2-1.0D0+(PARN2+1.0D0)*OMC2)*CY3E
     1 -2.D0*DSQRT(CNST)*(DABS(PARN)*OMC+BTH2*(HN+1.0D0)*
     2 (1.0D0-PARN2)/DABS(PARN))*CY4E)
      EPS33=EPS33/((1.0D0-PARN2)*(1.0D0-PARN2))
      GO TO 610
  607 CONS6=OMP2*PI/(2.D0*CYKE)
      CONS7=DEXP((2.0D0-OMC-1.0D0/OMC)/(2.0D0*BTH2))
      CONS8=OMC**HN
      EPSQ=AN1*CONS5*CONS6*CONS7*CONS8
      EPS11=EPSQ*HN*HN
      EPS13=EPSQ*PARN*HN*PERPN*HN/OMC
      EPS13=EPS13*(ZP+(HN+1.0D0)*BTH2)
      EPS33=EPSQ*PERPN*PERPN*HN*HN/OMC2
      EPS33=EPS33*(ZP*ZP+2.0D0*ZP*(HN+1.0D0)*BTH2
     1 +(HN+1.0D0)*(HN+2.0D0)*BTH2*BTH2)
      GO TO 610
  609 EPSQ=EPSQ*OMP2*PI
      XARG0=DSQRT(CNST)/BTH2
      FACT0=SQRT(PI)/2.0D0
      DO I=1,NH
       FACT0=FACT0*(I+0.5D0)
       ENDDO
      CY0=(0.5D0*XARG0)**(HN+0.5D0)/FACT0
      EPS11=EPSQ*CY0*HN*HN
      EPS13=0.0D0
      EPS33=EPS11*PERPN*PERPN*CNST/(OMC2*(2.0D0*HN+3.0D0))
  610 CONTINUE
      WRITE(6,*) ' '
      WRITE(6,*)'  ANALYTICAL RESULTS FROM A RELATIVISTIC MAXWELLIAN'
      WRITE(6,*) 'IMAGINARY EPS11',EPS11
      WRITE(6,*) 'EPS12',EPS11
      WRITE(6,*) 'EPS21=-EPS12'
      WRITE(6,*) 'IMAGINARY EPS22',EPS11
      WRITE(6,*) 'IMAGINARY EPS13',EPS13
      WRITE(6,*) 'EPS31=EPS13'
      WRITE(6,*) 'EPS32',EPS13
      WRITE(6,*) 'EPS23=-EPS32'
      WRITE(6,*) 'IMAGINARY EPS33',EPS33
  300 CONTINUE
      WRITE(6,*) ' '
      WRITE(6,*) '  RESULTS FROM NUMERICAL CALCULATION FOR A LH WAVE'
      PARNH2=PARNLH*PARNLH
      IF(PARNH2.LE.1.0) WRITE(6,*)  'DAMPING OF LH WAVE IMPOSSIBLE'
      IF(PARNH2.LE.1.0) STOP
      ZP=1.0D0/DSQRT(PARNH2-1.0D0)
      ZM=-ZP
      IM=1
      IP=NNX
      DO I=1,NNX1
       PMBTH=VXPARA(I)*BTH
       PMBTH1=VXPARA(I+1)*BTH
       IF(PMBTH.LT.ZM.AND.PMBTH1.GT.ZM) IM=I
       IF(PMBTH.EQ.ZM) IM=I
       PPBTH=VXPARA(I)*BTH
       PPBTH1=VXPARA(I+1)*BTH
       IF(PPBTH.LT.ZP.AND.PPBTH1.GT.ZP) IP=I
       IF(PPBTH1.EQ.ZP) IP=I
       ENDDO
      DINT22=0.0D0
      DINT32=0.0D0   
      DDXX2=-VYPERP(1)*DFDX(IP+1,2)/DY+VYPERP(2)*DFDX(IP+1,1)/DY
      DDXX1=-VYPERP(1)*DFDX(IM,2)/DY+VYPERP(2)*DFDX(IM,1)/DY
      IF(PARNLH.LT.-1.) GO TO 604
      DZ=(VXPARA(IP+1)-ZP/BTH)/DX
      DINT33=0.5D0*DDXX2*PARNLH*DZ*DX*(ZP**2)
      GO TO 605
  604 DZ=(ZM/BTH-VXPARA(IM))/DX
      DINT33=0.5D0*DDXX1*PARNLH*DZ*DX*(ZM**2)
  605 CONTINUE
      IF(PARNLH.GT.1.) II=IP+1
      IF(PARNLH.GT.1.) IF=NNX1
      IF(PARNLH.LT.-1.) II=+1
      IF(PARNLH.LT.-1.) IF=IM
      ZMAX=PMAX*BTH
      DO 601 I=II,IF
       ZPAR=VXPARA(I)*BTH
       ZARG=(PARNH2-1.0D0)*ZPAR*ZPAR-1.0D0
       IF(ZARG.LE.0.0) GO TO 601
       ZPERP=DSQRT(ZARG)
       IF(ZPERP.GE.ZMAX) GO TO 601
       PPERP=ZPERP/BTH
       IPERP=PPERP/DY+1
       DDXX=(PPERP-VYPERP(IPERP))*DFDX(I,IPERP+1)/DY
     1 +(VYPERP(IPERP+1)-PPERP)*DFDX(I,IPERP)/DY
       DER= DDXX*PARNLH*DX
C      SMALL ARGUMENT EXPANSION
       DIN22=(ZPERP**4)*DER/4.D0
       DIN32=(ZPERP**2)*ZPAR*DER/2.D0
       DIN33=(ZPAR**2)*DER
       IF((I.NE.(IP+1)).AND.(I.NE.IM)) GO TO 606
       DIN22=DIN22*0.5D0*(1.0D0+DZ)
       DIN32=DIN32*0.5D0*(1.0D0+DZ)
       DIN33=DIN33*0.5D0*(1.0D0+DZ)
  606  DINT22=DINT22+DIN22
       DINT32=DINT32+DIN32
       DINT33=DINT33+DIN33
  601  CONTINUE
C     LANDAU DAMPING
      XARGR=PERPLH/OMCLH
      CONST= OMP2*2.D0*PI*PI/BTH**3
      DINT32=DINT32*CONST*XARGR
      DINT22=-DINT22*CONST*(XARGR**2)
      DINT33=-DINT33*CONST  
      WRITE(6,*) 'LANDAU DAMPING IMAGINARY EPS22',DINT22
      DINT22=DINT22*(1.0D0/XARGR)**2/(2.D0*PARNH2*BTH2*BTH2)
      WRITE(6,*) 'NORMALIZED LANDAU DAMPING IMAGINARY EPS22',DINT22
      WRITE(6,*) 'LANDAU DAMPING EPS32',DINT32
      DINT32=DINT32/(XARGR*PARNLH*BTH2)
      WRITE(6,*) 'NORMALIZED LANDAU DAMPING EPS32',DINT32
      WRITE(6,*) 'LANDAU DAMPING IMAGINARY EPS33',DINT33 
      WRITE(6,*) ' '
      WRITE(6,*)
     1 '  ANALYTICAL RESULTS FROM RELATIVISTIC MAXWELLIAN FOR A LH WAVE' 
      DAPAR=DABS(PARNLH)
      XARG4=DAPAR/(BTH2*DSQRT(PARNH2-1.0D0))
      DCYK=DEXP(1.0D0/BTH2-XARG4)/CYKE
      EPSX=PI*0.5D0*OMP2*DCYK
      EPS33=EPSX*(1.0D0+2.D0/XARG4+2.D0/(XARG4*XARG4))
     1 /(BTH2*DAPAR*(PARNH2-1.0D0))
      EPS32=-EPSX*(1.0D0+3.D0/XARG4+3.D0/(XARG4*XARG4))/PARNH2
      EPS22=-2.D0*EPS32*(PARNH2-1.0D0)*BTH2*XARGR**2/DAPAR
      EPS32=EPS32*PARNLH*XARGR/DAPAR
      WRITE(6,*) 'LANDAU DAMPING IMAGINARY EPS22 ' , EPS22
      EPS22=EPS22*(1.0D0/XARGR)**2/(2.D0*PARNH2*BTH2*BTH2)
      WRITE(6,*) 'NORMALIZED LANDAU DAMPING IMAGINARY EPS22 ' , EPS22
      WRITE(6,*) 'LANDAU DAMPING EPS32 ' , EPS32
      EPS32=EPS32/(XARGR*PARNLH*BTH2)
      WRITE(6,*) 'NORMALIZED LANDAU DAMPING EPS32 ' , EPS32
      WRITE(6,*) 'LANDAU DAMPING IMAGINARY EPS33',EPS33      
      STOP                                                              
      END 
