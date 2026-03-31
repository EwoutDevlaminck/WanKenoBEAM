""" Define a function which takes the data from ray tracing and does the binning.
The results are written in an array in the parameter list.
"""

# Load standard modules
import numpy as np
import sys
cimport numpy as np

# and some c-functions
cdef extern from "math.h":
    double sqrt( double )
    double exp( double )
    double log( double )
    double floor( double )



############################################################################
# DEFINE SOME MINOR FUNCTIONS USED BELOW
############################################################################
# function which calculates the weighted average of quant0 and quant1
# where xi gives the weight. xi=0 means take purely quant0,
# xi=1 takes quant1
cdef inline linint(double xi, double quant0, double quant1):
   return xi*quant1 + (1.-xi)*quant0


cdef inline int find_bin_index(np.ndarray[double, ndim=1] a, int n, double b):
   # Returns the index of the bin that a point belongs to. This is done by
   # evaluating the point w.r.t. the bin boundaries provided in a (length n).
   # If it lies outside the domain, it is given an index corresponding to a
   # grid that is uniformly extended beyond the boundaries, allowing for a
   # finer grid to work on when interpolating later on.
   # Uses a pure-C binary search (equivalent to np.searchsorted left) to
   # avoid any Python/NumPy overhead in the hot binning loop.

   cdef int lo, hi, mid

   if b < a[0]:  # Extrapolate on the lower end
      return <int>floor((b - a[0]) / (a[1] - a[0]))

   elif b > a[n-1]:  # Extrapolate on the upper end
      return n - 1 + <int>floor((b - a[n-1]) / (a[n-1] - a[n-2]))

   # Binary search: equivalent to np.searchsorted(a, b, side='left') - 1
   lo = 0
   hi = n
   while lo < hi:
      mid = (lo + hi) >> 1
      if a[mid] < b:
         lo = mid + 1
      else:
         hi = mid
   return lo - 1




# bin... : The bin edges along a certain dimension.
# nmbrRays: number of rays which were traced
# data_sim: data of the ray tracing
# data_sim_NxNyNz: if available ray tracing data on NxNyNz
# data_sim_NparallelphiN: the same for Nparallel, phiN
# Wfct: array where the results of the binning will be written, 5 dimensional:
#       index: 1st --> 1st direction
#              2nd -->
#              3rd -->
#              4th --> 4th direction
#              5th --> 0: data from the binning,
#                      1: uncertainty

# data_sim_weight:
# if an array gives the appropriate weight
# if size=0, weight 1 is used

# scatterparameter: 0 --> do the binning for both, scattered and unscattered rays
#                   1 --> binning only for unscattered rays
#                   2 --> binning only for scattered rays

# data_sim_scattered:
# index: rays, contains the number of scattering kicks a ray has experienced.

# absorption == 1 --> use absorption binning strategy,
#               0 --> all normal binning

# time: time along the rays, array


# returns: number of rays which are binned
############################################################################
# DEFINE FUNCTION WHICH DOES THE BINNING
############################################################################
cpdef int binning(np.ndarray [double, ndim=3] data_sim,
                  np.ndarray [double, ndim=2] data_sim_Wfct,
                  np.ndarray [double, ndim=2] data_sim_CorrectionFactor,
                  np.ndarray [double, ndim=2] data_sim_weight,
      	          np.ndarray [double, ndim=5] Wfct,
                  np.ndarray [double, ndim=1] bin1,
                  np.ndarray [double, ndim=1] bin2,
                  np.ndarray [double, ndim=1] bin3,
                  np.ndarray [double, ndim=1] bin4,
                  np.ndarray [double, ndim=2] time,
                  int scatterparameter,
                  np.ndarray [np.int_t, ndim=1] data_sim_scattered,
                  int absorption):


   """ Crucial binning function. See source file for more information.
   """


   ############################################################################
   # check if the given parameters do make sense
   ############################################################################

   cdef int CorrectionFactorGiven
   if data_sim_CorrectionFactor.size > 0:
      CorrectionFactorGiven = 1
   else:
      CorrectionFactorGiven = 0
      print('The correction factor for transforming the results to the ones corresponding to the physical Hamiltonian is not given. It is assumed to be 1.\n')
      sys.stdout.flush()

   cdef int weightGiven
   if data_sim_weight.size == 0:
      weightGiven = 0
   else:
      weightGiven = 1

   ############################################################################
   # print some information
   ############################################################################
   print('binning started ...\n')
   sys.stdout.flush()

   ############################################################################
   # estimate some more information out of the data_sim array given
   ############################################################################
   # number of traces
   cdef int nTraces = data_sim.shape[0]

   # number of points per ray
   cdef int nPoints = data_sim.shape[2]

   # get number of bins and bin array lengths in each direction
   cdef int nmbrdir1 = len(bin1)-1
   cdef int nmbrdir2 = len(bin2)-1
   cdef int nmbrdir3 = len(bin3)-1
   cdef int nmbrdir4 = len(bin4)-1
   cdef int nbin1 = nmbrdir1 + 1
   cdef int nbin2 = nmbrdir2 + 1
   cdef int nbin3 = nmbrdir3 + 1
   cdef int nbin4 = nmbrdir4 + 1

   # Step sizes outside of the binned domain, used for linear interpolation in case
   # the ray traverses the edge of the domain. These are also used in the find_bin_index function
   cdef double Deltadir1m = bin1[1] - bin1[0]   # on the minus side
   cdef double Deltadir1p = bin1[-1] - bin1[-2] # on the plus side
   cdef double Deltadir2m = bin2[1] - bin2[0]
   cdef double Deltadir2p = bin2[-1] - bin2[-2]
   cdef double Deltadir3m = bin3[1] - bin3[0]
   cdef double Deltadir3p = bin3[-1] - bin3[-2]
   cdef double Deltadir4m = bin4[1] - bin4[0]
   cdef double Deltadir4p = bin4[-1] - bin4[-2]

   # define some variables which will be needed
   # (...0: starting point, ...1: destination point)
   cdef int index
   cdef double t0, dir10, dir20, dir30, dir40, Wfct0, GrpVel0=1., CorrectionFactor0=1., weight0=1.
   cdef double t1, dir11, dir21, dir31, dir41, Wfct1, GrpVel1=1., CorrectionFactor1=1., weight1=1.
   cdef double WfctInt

   # define variables where the corresponding bin indices can be stored
   cdef int ndir10, ndir20, ndir30, ndir40
   cdef int ndir11, ndir21, ndir31, ndir41

   # define variable where the Wfct contribution of a ray in one bin is temporarily stored
   cdef double DeltaWfct

   # define variable, where the uncertainty for the recent bin is temporarily stored
   cdef double DeltaUncertainty

   # define parameter to parametrice the ray in between two points
   cdef double xidir1, xidir2, xidir3, xidir4
   cdef double dir1bound, dir2bound, dir3bound, dir4bound

   # Boolean parameter to skip fragment of a ray if it is completely out of the domain
   cdef int skip_fragment


   # define temporarily used variables
   cdef double xiUsed      # in order to store the xi which actually is in use
   cdef int whichxiUsed    # in order to store information on which xi is used
                           # 0 --> xidir1, 1 --> xidir2, 2 --> xidir3, 3 --> xidir4

                           # in order to temporarily store those quantities
   cdef double tempWfct, tempGrpVel=1., tempCorrectionFactor=1., tempweight=1.

   # define variable where the binned rays are counted
   cdef int raycounter = 0

   ############################################################################
   # loop over the rays
   ############################################################################
   for i in range(0,nTraces):

      # sometimes print out some information.
      if i % 500 == 0:
         print('progress: binning ray %i / %i\n' %(i,nTraces))
         sys.stdout.flush()

      # see if the recent ray has to be binned or not
      if scatterparameter == 0:     # means: bin all rays
          pass
      elif scatterparameter == 1:   # means: only unscattered rays
          if data_sim_scattered[i] > 0:   # if ray is scattered: skip it
              continue
      elif scatterparameter == 2:   # means: only scattered rays
          if data_sim_scattered[i] == 0:  # if ray is not scattered: skipt it
              continue

      # if the ray is binned, increment the ray counter
      raycounter += 1


      # read the first ray point
      t0 = 0.
      index = 0
      dir10 = data_sim[i,0,index]
      dir20 = data_sim[i,1,index]
      dir30 = data_sim[i,2,index]
      dir40 = data_sim[i,3,index]
      Wfct0 = data_sim_Wfct[i,index]

      if CorrectionFactorGiven == 1:
         CorrectionFactor0 = data_sim_CorrectionFactor[i,index]
      if weightGiven == 0:
         weight0 = 1.
      else:
         weight0 = data_sim_weight[i,1]

      # and calculate the bin indices. the data in bin1 etc are the bin boundaries
      ndir10 = find_bin_index(bin1, nbin1, dir10)
      ndir20 = find_bin_index(bin2, nbin2, dir20)
      ndir30 = find_bin_index(bin3, nbin3, dir30)
      ndir40 = find_bin_index(bin4, nbin4, dir40)

      # for the first bin, for the moment, the uncertainty to add is 0
      DeltaUncertainty = 0.


      ############################################################################
      # loop over the ray points
      ############################################################################
      # start with the third line, because in the first, there is the extra information
      # and the first ray point in the second line is already stored in point 0
      for j in range(1, nPoints):
         # read the next ray point
         t1 = time[i,j]
         dir11 = data_sim[i,0,j]
         dir21 = data_sim[i,1,j]
         dir31 = data_sim[i,2,j]
         dir41 = data_sim[i,3,j]
         Wfct1 = data_sim_Wfct[i,j]


         # if Wfct vanishes, this is because the ray tracing has been stopped for some reasons.
         # in this case, don't further consider the recent ray
         if Wfct1 == 0.:
             break

         if CorrectionFactorGiven == 1:
            CorrectionFactor1 = data_sim_CorrectionFactor[i,j]
         if weightGiven == 0:
            weight1 = 1.
         else:
            weight1 = data_sim_weight[i,j]

         # and calculate the bin indices
         ndir11 = find_bin_index(bin1, nbin1, dir11)
         ndir21 = find_bin_index(bin2, nbin2, dir21)
         ndir31 = find_bin_index(bin3, nbin3, dir31)
         ndir41 = find_bin_index(bin4, nbin4, dir41)

         # At the start, don't skip fragment yet
         skip_fragment = 0


         ############################################################################
         # loop over intersections with bin boundaries in between the two recent ray
         # points
         ############################################################################
         while True:


            ############################################################################
            # if start and end point are in the same bin, the binning can directly be
            # done and one can continue with the next ray point
            ############################################################################
            # if the starting point and the end point are in the same bin, do the binning directly
            if ndir10 == ndir11 and ndir20 == ndir21 and ndir30 == ndir31 and ndir40 == ndir41:
               if ndir10 >= 0 and ndir10 < nmbrdir1 \
                   and ndir20 >= 0 and ndir20 < nmbrdir2 \
                   and ndir30 >= 0 and ndir30 < nmbrdir3 \
                   and ndir40 >= 0 and ndir40 < nmbrdir4:
                  # calculate the distance in between the two points (either the spacial distance or the time)
                  dist = (t1-t0)

                  # and add the right value to the bin. It must be properly weighted with the probability of
                  # launching the recent ray. The Wfct which is taken into account corresponds to the average value of
                  if absorption == 0:
                      # the two points in between of whom one does the binning.
                      DeltaWfct = dist * (Wfct0+Wfct1)/2. * (CorrectionFactor0+CorrectionFactor1)/2. * (weight0+weight1)/2.
                  else:   # if absorption is chosen it must be treated in a different way
                      DeltaWfct = (Wfct0 - Wfct1) * (weight0+weight1)/2.


                  Wfct[ndir10,ndir20,ndir30,ndir40,0] += DeltaWfct
                  DeltaUncertainty += DeltaWfct

               # the next starting ray point is the recent end point
               # the bin indices are already the same.
               t0 = t1
               dir10 = dir11
               dir20 = dir21
               dir30 = dir31
               dir40 = dir41
               Wfct0 = Wfct1
               CorrectionFactor0 = CorrectionFactor1
               weight0 = weight1
               break

            #############################################################################
            # if start and end point are not in the same bin, do the binning step by step
            # for all the bins in between both.
            #############################################################################
            # calculate the next intersection with a bin boundary
            # and therefore calculate the relevant boundaries
            # always the boundary which is at least to the right direction of the starting
            # point 0 is taken. Therefore, it is seen if the coordinates of point 1 are
            # superior or not.
            # Many different cases are possible for a ray fragment crossing bin borders.
            # We always want to cut the fragment starting from dir10 towards the nearest boundary

            # For each direction, follow the same procedure
            if dir11 >= dir10:
               if ndir11 < 0 or ndir10 > nmbrdir1 -1:
                  # Then the fragment has either not entered the binning domain yet
                  # or it was already out of the domain at the start
                  skip_fragment = 1
               elif ndir10 < 0:
                  # Start of fragment was not in domain, but end is.
                  # We cut the ray from its starting point ar dir10 to the closest boundary
                  # of a bin (even if outside the domain, we add extra bins with Deltadir)
                  dir1bound = bin1[0] + (ndir10 + 1)*Deltadir1m
               else:
                  # Start of fragment was already in domain, and we look for it's closest
                  # boundary at a higher value.
                  dir1bound = bin1[ndir10+1]
            else:
               # Fragment goes from high to low value
               if ndir10 < 0 or ndir11 > nmbrdir1 -1:
                  # Fragment already out of domain, or hasn't entered yet
                  skip_fragment = 1
               elif ndir10 > nmbrdir1 - 1:
                  # Fragment enters the domain 'from the right'
                  dir1bound = bin1[-1] + (ndir10 - nmbrdir1)*Deltadir1p
               else:
                  # Start of fragment was already in domain, and we look for it's closest
                  # boundary at a lower value.
                  dir1bound = bin1[ndir10]

            if dir21 >= dir20:
               if ndir21 < 0 or ndir20 > nmbrdir2 -1:
                  skip_fragment = 1
               elif ndir20 < 0:
                  dir2bound = bin2[0] + (ndir20 + 1)*Deltadir2m
               else:
                  dir2bound = bin2[ndir20+1]
            else:
               if ndir20 < 0 or ndir21 > nmbrdir2 -1:
                  skip_fragment = 1
               elif ndir20 > nmbrdir2 - 1:
                  dir2bound = bin2[-1] + (ndir20 - nmbrdir2)*Deltadir2p
               else:
                  dir2bound = bin2[ndir20]

            if dir31 >= dir30:
               if ndir31 < 0 or ndir30 > nmbrdir3 -1:
                  skip_fragment = 1
               elif ndir30 < 0:
                  dir3bound = bin3[0] + (ndir30 + 1)*Deltadir3m
               else:
                  dir3bound = bin3[ndir30+1]
            else:
               if ndir30 < 0 or ndir31 > nmbrdir3 -1:
                  skip_fragment = 1
               elif ndir30 > nmbrdir3 - 1:
                  dir3bound = bin3[-1] + (ndir30 - nmbrdir3)*Deltadir3p
               else:
                  dir3bound = bin3[ndir30]

            if dir41 >= dir40:
               if ndir41 < 0 or ndir40 > nmbrdir4 -1:
                  skip_fragment = 1
               elif ndir40 < 0:
                  dir4bound = bin4[0] + (ndir40 + 1)*Deltadir4m
               else:
                  dir4bound = bin4[ndir40+1]
            else:
               if ndir40 < 0 or ndir41 > nmbrdir4 -1:
                  skip_fragment = 1
               elif ndir40 > nmbrdir4 - 1:
                  dir4bound = bin4[-1] + (ndir40 - nmbrdir4)*Deltadir4p
               else:
                  dir4bound = bin4[ndir40]


            if skip_fragment == 1:
               # We don't have to consider this fragment at all, and can continue
               # go one with the next ray point
               t0 = t1
               dir10 = dir11
               dir20 = dir21
               dir30 = dir31
               dir40 = dir41
               Wfct0 = Wfct1
               CorrectionFactor0 = CorrectionFactor1
               weight0 = weight1
               ndir10 = ndir11
               ndir20 = ndir21
               ndir30 = ndir31
               ndir40 = ndir41
               break

            # the nearest boundary is taken into account. Therefore, parameters
            # xi... along the ray are calculated giving the intersection with the
            # next bin boundary in ... direction. xi... = 0 corresponds to point 0,
            # xi... = 1 gives point 1.
            # If the coordinates are the same, the xi cannot be calculated. Then
            # it is set to 20 which is a value > 1 and thus this xi for sure is not
            # the next one (because even not in between point 0 and point 1) and
            # ignored.
            if dir11-dir10 != 0.:
               xidir1 = (dir1bound-dir10)/(dir11-dir10)
            else:
               xidir1 = 20.
            if dir21-dir20 != 0.:
               xidir2 = (dir2bound-dir20)/(dir21-dir20)
            else:
               xidir2 = 20.
            if dir31-dir30 != 0.:
               xidir3 = (dir3bound-dir30)/(dir31-dir30)
            else:
               xidir3 = 20.
            if dir41-dir40 != 0.:
               xidir4 = (dir4bound-dir40)/(dir41-dir40)
            else:
               xidir4 = 20.


            # do the binning for the recent bin (which means in between point 0 and
            # an intermediate point at the next smallest reasonable xi)
            # look for the smallest reasonable xi and do the binning for the
            # corresponding part of the ray. For further explanation on the binning
            # look where the binning within one bin is done.
            if xidir1 < xidir2 and xidir1 < xidir3 and xidir1 < xidir4: # and xidir1 <= 1.:
               xiUsed = xidir1
               whichxiUsed = 0
            elif xidir2 < xidir3 and xidir2 < xidir4: #and xidir2 < 1.:
               xiUsed = xidir2
               whichxiUsed = 1
            elif xidir3 < xidir4: #and xidir3 < 1.:
               xiUsed = xidir3
               whichxiUsed = 2
            elif xidir4 < 1.:
               xiUsed = xidir4
               whichxiUsed = 3

            # if there was no reasonable xi this means, that there was some problem due to rounding issues which occurs
            # whenever a ray is oriented exactly along a bin boundary and thus practically only for the vacuum case
            # and the central ray.
            else:
               # print a warning message.
               dist = t1-t0
               print("WARNING: a problem in the binning occured. For ray number=%i, dest. point number=%i. dist=%f[gen.time unit] could not be binned.\n" %(i,j-1,dist))
               sys.stdout.flush()

               # go one with the next ray point
               t0 = t1
               dir10 = dir11
               dir20 = dir21
               dir30 = dir31
               dir40 = dir41
               Wfct0 = Wfct1
               CorrectionFactor0 = CorrectionFactor1
               weight0 = weight1
               ndir10 = ndir11
               ndir20 = ndir21
               ndir30 = ndir31
               ndir40 = ndir41
               break


            # compute the intermediate quantities and save in temporare variables
            tempWfct = linint(xiUsed, Wfct0, Wfct1)
            tempCorrectionFactor = linint(xiUsed, CorrectionFactor0, CorrectionFactor1)
            tempweight = linint(xiUsed, weight0, weight1)

            # and do the binning
            if absorption == 0:
               dist = (t1-t0)*xiUsed
               DeltaWfct = dist * (Wfct0+tempWfct) / 2. * (CorrectionFactor0+tempCorrectionFactor)/2. * (weight0+tempweight)/2.
            else:   # for absorption different computation
               if Wfct1 > 0.:
                  WfctInt = Wfct0 * exp(log(Wfct1/Wfct0)*xiUsed)
               else:
                  WfctInt = 0.
               DeltaWfct = (Wfct0 - WfctInt) * (weight0+tempweight)/2.

            # if no problem occured, see if indices are inside the boundaries and if yes do the binning.
            if ndir10 >= 0 and ndir10 < nmbrdir1 \
                and ndir20 >= 0 and ndir20 < nmbrdir2 \
                and ndir30 >= 0 and ndir30 < nmbrdir3 \
                and ndir40 >= 0 and ndir40 < nmbrdir4:
                # If this is the case, the -now cut from start point 0 to a boundary-
                # fragment is in the domain.

               Wfct[ndir10,ndir20,ndir30,ndir40,0] += DeltaWfct
               Wfct[ndir10,ndir20,ndir30,ndir40,1] += (DeltaWfct+DeltaUncertainty)**2

            DeltaUncertainty = 0.

            # the recent intermediate point is the next starting point
            t0 = linint(xiUsed, t0, t1)
            dir10 = linint(xiUsed, dir10, dir11)
            dir20 = linint(xiUsed, dir20, dir21)
            dir30 = linint(xiUsed, dir30, dir31)
            dir40 = linint(xiUsed, dir40, dir41)
            if absorption == 0:
                Wfct0 = linint(xiUsed, Wfct0, Wfct1)
            else:
                Wfct0 = WfctInt
            CorrectionFactor0 = linint(xiUsed, CorrectionFactor0, CorrectionFactor1)
            weight0 = linint(xiUsed, weight0, weight1)

            # and the bin index of the recent bin is moved into the right neighbour bin.
            # therefore, see which bin index has to be changed
            if whichxiUsed == 0:
               if dir11 > dir10:
                  ndir10 += 1
               else:
                  ndir10 -= 1
            elif whichxiUsed == 1:
               if dir21 > dir20:
                  ndir20 += 1
               else:
                  ndir20 -= 1
            elif whichxiUsed == 2:
               if dir31 > dir30:
                  ndir30 += 1
               else:
                  ndir30 -= 1
            elif whichxiUsed == 3:
               if dir41 > dir40:
                  ndir40 += 1
               else:
                  ndir40 -= 1


   # return the number of rays binned
   return raycounter
