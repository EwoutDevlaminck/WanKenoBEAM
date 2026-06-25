"""In this module an analytical calculation of 
derivatives for 2D Splines is implemented on the
basis of the RectBivariantSpline function of scipy."""

# Import Statements
import scipy.interpolate as spl

# Class that profides the possibility of interpolation 
# in 2D with also derivatives.
# ONLY FOR UNIFORMLY DISTRIBUTED GRID KNOTS #
class BiSpline(object):

    """This class provides the evaluation of a interpolation
    object as well as its derivatives.
    THE GRID KNOTS MUST BE DISTRIBUTED UNIFORMLY !

    USAGE:
      bispline = BiSpline(x,y,data,S=0)
    where x and y are 1D arrays giving the grid values of x and y 
    respectively and data is a 2D array giving the value to be 
    interpolated. S is a smoothing factor. The default value is 0
    which means no smoothing.

    LIST OF ATTRIBUTES:

     <> __interpol__: 
        RectBivariateSpline object with kx=ky=3, 
        used for evaluation of the interpolation

     <> __interpol_x__:
        RectBivariateSpline object with kx=2, ky=3,
        used for the evaluation of the derivative with
        respect to x
     
     <> __interpol_y__:
        RectBivariateSpline object with kx=3, ky=2,
        used for the evaluation of the derivative with
        respect to y

     <> __dX__:
        difference between two grid knots in X directtion

     <> __dY__:
        difference between two grid knots in Y direction     

    LIST OF METHODS:

      - eval(x,y):
        evaluates the interpolation object at position x,y.
        x and y must be floats.

      - eval_vec(x_arr, y_arr):
        evaluates the interpolation object at N scattered (x, y) points.
        x_arr and y_arr must be 1D arrays of length N.

      - derx(x,y):
        evaluates the derivative with respect to x of the
        interpolation. Again, x and y indicate the position
        and must be floats
      
      - dery(x,y):
        evaluates the derivative with respect to y of the
        interpolation. Again, x and y indicate the position
        and must be floats
    """
	
    ##########################################################################
    # INITIALIZATION OF THE CLASS
    # DEFINITION OF BASIC PROPERTIES AND FUNCTIONS.
    ##########################################################################
    def __init__(self, X, Y, data, S=0.):
        
        """Inizialization procedure. Given the input data as the instance
        idata of the class InputData, initialize the interpolation objects
        and estimate the position of the magetid axis in the R-z plana.
		Definition of some properties of the dispersion matrix.
        s is a smoothing factor.
        """
        
        # define the interpolation objects that must be used later
        self.__interpol__ = spl.RectBivariateSpline(X,Y,data,kx=3,ky=3,s=S)
        self.__interpol_x__ = spl.RectBivariateSpline(X,Y,data,kx=2,ky=3,s=S)
        self.__interpol_y__ = spl.RectBivariateSpline(X,Y,data,kx=3,ky=2,s=S)

        # for later, remember the grid density:
        self.__dX__ = X[1] - X[0]
        self.__dY__ = Y[1] - Y[0]

        # return from constructor
        return
 

    ##########################################################################
    # FUNCTION FOR THE EVALUATION OF INTERPOLATION AT A GIVEN POINT (x,y)
    ##########################################################################
    def eval(self,x,y):

        """This function returns the result of the interpolation
        at the given point (x,y). x and y must be floats."""

        return self.__interpol__(x,y).item(0)

    def eval_vec(self, x_arr, y_arr):
        """Evaluate the interpolant at N scattered (x, y) points.

        Parameters
        ----------
        x_arr, y_arr : 1-D ndarrays of length N

        Returns
        -------
        1-D ndarray of length N
        """
        return self.__interpol__.ev(x_arr, y_arr)

    ##########################################################################
    # FUNCTION FOR THE EVALUATION OF DERIVATIVE WITH RESPECT TO  x AT (x,y)
    ##########################################################################
    def derx(self,X,Y):

        """This function returns the result of the interpolation
        of the derivative at the given point (x,y). 
        x and y must be floats."""
              
        # define upper and lower x that are plugged in __interpol_x__
        Xplus = X + self.__dX__/2.
        Xmins = X - self.__dX__/2.

        # and calculate the result for the derivative
        return (self.__interpol_x__(Xplus,Y).item(0) \
              - self.__interpol_x__(Xmins,Y).item(0)) / self.__dX__
       

    ##########################################################################
    # FUNCTION FOR THE EVALUATION OF DERIVATIVE WITH RESPECT TO  y AT (x,y)
    ##########################################################################
    def dery(self,X,Y):

        """This function returns the result of the interpolation
        of the derivative at the given point (x,y). 
        x and y must be floats."""
              
        # define upper and lower y that are plugged in __interpol_y__
        Yplus = Y + self.__dY__/2.
        Ymins = Y - self.__dY__/2.

        # and calculate the result for the derivative
        return (self.__interpol_y__(X,Yplus).item(0) \
              - self.__interpol_y__(X,Ymins).item(0)) / self.__dY__
      

#
# End of class BiSpline



# This class provides an interpolation object which is meant for the density
# profile in tokamak geometry, i.e., when the density is a function
# of the normalized poloidal flux only.
# From outside, this object behaves exactly an instance of the class BiSpline,
# but internally univariate spline are used for the density and BiSpline is
# used for the normalized poloidal flux.  
class UniBiSpline(object):

    """In tokamak geometry the density depends only on the poloidal flux, so it
    is usually reprenented as a one dimensional array.

    This class provides an interpolation object for functions F(x,y) that 
    actaully depends only on a scalar psi(x,y). The interpolation object 
    behaves exactly as an instance of the class BiSpline, for fully 2D profiles.

    Internally, the 1D array representing the 1D profile of F(psi) is 
    interpolated by UnivariateSpline, while the scalar psi is interpolated 
    by BiSpline. 

    Derivatives of the profile in the radial (R) and vertical (z)
    directions are constructed from its derivative with respect to psi and
    the gradient of psi itself.

    Attributes and "public" methods mirror those of the class BiSpline."""
    
    ##########################################################################
    # INITIALIZATION OF THE CLASS
    # DEFINITION OF BASIC PROPERTIES AND FUNCTIONS.
    ##########################################################################
    def __init__(self, psi_F, data_F, psi_interpolator):

        """ Constructor method: Initialize the interpolation objects for
        the profile and the scalar psi. """

        # Interpolation object for the poloidal flux
        self.__psi__ = psi_interpolator

        # Interpolation object for the density
        self.__profile__ = spl.UnivariateSpline(psi_F, data_F, s=0,
                                                ext='extrapolate',
                                                bbox=[0.0, psi_F[-1]])
        
        # return from constructor
        return
 

    ##########################################################################
    # FUNCTION FOR THE EVALUATION OF PROFILE AT A GIVEN POINT (x,y)
    ##########################################################################
    def eval(self, x, y):

        """This function returns the result of the interpolation
        at the given point (x,y). The coordinates x and y must be floats.

        This is done in two steps: 1) given (x,y) one evaluates psi(x,y);
        2) at last evaluate F(x,y) = F(psi(x,y)).
        """

        psiloc = self.__psi__.eval(x, y)
        Floc = self.__profile__(psiloc)
        return Floc

    def eval_vec(self, x_arr, y_arr):
        """Evaluate the profile at N scattered (x, y) points.

        Parameters
        ----------
        x_arr, y_arr : 1-D ndarrays of length N

        Returns
        -------
        1-D ndarray of length N
        """
        psi_arr = self.__psi__.__interpol__.ev(x_arr, y_arr)
        return self.__profile__(psi_arr)

    ##########################################################################
    # FUNCTION FOR THE EVALUATION OF DERIVATIVE dF/dpsi
    ##########################################################################
    def __dF_dpsi__(self, x, y):

        """Evaluate the derivative of the profiles with respect to the
        scalar psi at (x,y). Coordinates x and y must be float."""

        # Local value of the normalized poloidal flux
        psiloc = self.__psi__.eval(x, y)

        # Derivative of the density profile
        try:
            results = self.__profile__.derivatives(psiloc)
            dF_dpsi = results.item(1)
        except:
            dF_dpsi = 0.
        
        return dF_dpsi

    ##########################################################################
    # FUNCTION FOR THE EVALUATION OF DERIVATIVE WITH RESPECT TO x AT (x,y)
    ##########################################################################
    def derx(self, x, y):

        """This function returns the x-derivative of the interpolant
        at the given point (x,y). Coordinates x and y must be floats."""

        # Derivative of the profile
        dF_dpsi = self.__dF_dpsi__(x,y)
        
        # Derivative of the scalar psi
        dpsi_dx = self.__psi__.derx(x,y)
        
        # Final results
        dF_dx = dF_dpsi * dpsi_dx

        return dF_dx

    ##########################################################################
    # FUNCTION FOR THE EVALUATION OF DERIVATIVE WITH RESPECT TO y AT (x,y)
    ##########################################################################
    def dery(self, x, y):

        """This function returns the vertical derivative of the interpolant
        at the given point (R, z). Coordinates R and z must be floats."""

        # Derivative of the profile
        dF_dpsi = self.__dF_dpsi__(x,y)

        # Derivative of the scalar psi
        dpsi_dy = self.__psi__.dery(x,y)

        # Final results
        dF_dy = dF_dpsi * dpsi_dy

        return dF_dy
              
#
# End of class UniBiSpline
