import numpy as np
import pandas as pd
from math import sin, cos, asin, acos, atan2, sqrt, radians

def sph2cart(lon, lat, om=1.):
    """ Transforms spherical to cartesian coordinates
    
    Parameters
    ----------
    lat : array[1xn]
        Latitude in [degrees]
    lon : array[1xn]
        Longitude in [degrees]
    om : array[1xn], optional
        Angular magnitude in [degrees] or [degrees/Myr].
    
    Returns
    ----------
    x,y,z : floats
        Cartesian coordinates in same unit of measurement as 
        om ([degrees] or [degrees/Myr]).
    """
    
    m = np.size(lat)

    x = np.zeros(m)
    y = np.zeros(m)
    z = np.zeros(m)
    
    for ii in range(m):
        
        if np.size(lat) > 1:
            
            if isinstance(lat, type(pd.Series(dtype="float64"))):
                
                lonSph = lon.iloc[ii]
                latSph = lat.iloc[ii]
                
                if np.size(om) > 1: # om is given and is a pd.Series
                    omScalar = om.iloc[ii]
            
            else:
                
                lonSph = lon[ii]
                latSph = lat[ii]
                
                if np.size(om) > 1: # om is given and is a list
                    omScalar = om[ii]
            
        else:
                
            lonSph = lon
            latSph = lat
        
        
        omScalar = om
        lonRad = radians(lonSph)
        latRad = radians(latSph)
        
        if np.size(lat) > 1:
            x[ii] = omScalar * cos(lonRad) * cos(latRad)
            y[ii] = omScalar * cos(latRad) * sin(lonRad)
            z[ii] = omScalar * sin(latRad)     
        
        else:
            x = omScalar * cos(lonRad) * cos(latRad)
            y = omScalar * cos(latRad) * sin(lonRad)
            z = omScalar * sin(latRad)  
    
    return x, y, z


def cart2sph(x=[], y=[], z=[], 
             xyz = [],
             degreesFormat = True,
             onlyOm = False): 
    
    """ Transforms cartesian to spherical coordinates
    
    Parameters
    ----------
    x,y,z : array[1xn]
        Cartesian coordinates
    degreesFormat : boolean, optional
        Allows for output in degrees (True) or radians (False)
    
    Returns
    ----------
    lat : list if floats
        Latitude in degrees
    lon : list if floats
        Longitude in degrees
    om : list if floats
        Angle (magnitude) in degrees
        
    """
    
    # If variable xyz is not empty, and array with the three cartesian was given
    if len(xyz) != 0:
        if isinstance(xyz, (tuple, list, np.ndarray)):
            x = xyz[0]
            y = xyz[1]
            z = xyz[2]
            
        elif isinstance(xyz, pd.core.frame.DataFrame):
            x = xyz["x"]
            y = xyz["y"]
            z = xyz["z"]
        else:
            print ('xyz type not recognised')
      
            
    # If x, y and z are integers
    if isinstance(x, (int, float)):

        om = sqrt( (x**2 + y**2) + z**2 )
        lat = atan2( z, sqrt(x**2 + y**2) )
        lon = atan2(y, x)
        
        # Turns lat, lon output from radians to degrees        
        if degreesFormat:
            lat = np.round(np.degrees(lat), 7)
            lon = np.round(np.degrees(lon), 7)
            #if lon < 0:
                #lon = lon + 360
    
    
    else:
        # Set lat, lon and omega empty lists
        lat = []
        lon = []    
        om = []
               
        if onlyOm == False:
            
            # Iterate and extend lists
            lat.extend( [atan2(Az, sqrt(Ax**2 + Ay**2)) for Ax, Ay, Az in zip(x,y,z)] )
            lon.extend( [atan2(Ay, Ax) for Ax, Ay in zip(x,y)] )
            
            # Turns lat, lon output from radians to degrees        
            if degreesFormat:
                lat = list(np.degrees(lat))
                lon = list(np.degrees(lon))
                
        #for i in range(len(lon)):
            #if lon[i] < 0:
                #lon[i] = lon[i] + 360
        
        
        om.extend( [sqrt(Ax**2 + Ay**2 + Az**2) for Ax, Ay, Az in zip(x,y,z)] )
        
    return lon, lat, om


def pole2matrix(lon, lat, om=0):
    """ Transforms rotation vector in spherical coordinates to rotation matrix
    
    Parameters
    ----------
    lon : float
        Longitude in degrees
    lat : float
        Latitude in degrees
    om : float
        Angle (magnitude) in degrees    
    
    Returns
    ----------
    rotMatrix : array[3x3]
        Rotation matrix
    
    """
    
    lon = radians(lon)
    lat = radians(lat)
    om = radians(om)
    
    ex = cos(lon) * cos(lat)
    ey = cos(lat) * sin(lon)
    ez = sin(lat)
    
    r11 = (ex**2) * (1. - cos(om)) + cos(om)
    r12 = ex * ey * (1. - cos(om)) - ez * sin(om)
    r13 = ex * ez * (1. - cos(om)) + ey * sin(om)
     
    r21 = ey * ex * (1. - cos(om)) + ez * sin(om)
    r22 = (ey**2) * (1. - cos(om)) + cos(om)
    r23 = ey * ez * (1. - cos(om)) - ex * sin(om)
    
    r31 = ez * ex * (1. - cos(om)) - ey * sin(om)
    r32 = ez * ey * (1. - cos(om)) + ex * sin(om)
    r33 = (ez**2.) * (1. - cos(om)) + cos(om)
    
    rotMatrix = np.array([[r11,r12,r13],[r21,r22,r23],[r31,r32,r33]])
    
    return rotMatrix


def matrix2pole(rot_matrix, degreesFormat=True):
    
    """ Transforms rotation matrix to rotation vector in spherical coordinates
    
    Parameters
    ----------
    rotMatrix : array[3x3]
        Rotation matrix
    degreesFormat : boolean, optional
        Allows for output in degrees (True) or radians (False)
    
    Returns
    ----------
    lon : float
        Longitude in degrees
    lat : float
        Latitude in degrees
    om : float
        Angle (magnitude) in degrees    
    
    """
    
    r = rot_matrix
    
    r11 = r[0][0]
    r12 = r[0][1]
    r13 = r[0][2]
    
    r21 = r[1][0]
    r22 = r[1][1]
    r23 = r[1][2]
    
    r31 = r[2][0]
    r32 = r[2][1]
    r33 = r[2][2]
    
    tmp = sqrt((r32-r23)**2. + (r13-r31)**2. + (r21-r12)**2.)
    if tmp == 0:
        lat = 0.0
    else: 
        lat = asin((r21-r12)/tmp)
        
    lon = atan2((r13-r31), (r32-r23))
    om = atan2(sqrt((r32-r23)**2. + (r13-r31)**2. + (r21-r12)**2.),
                  (r11 + r22 + r33 - 1.))
    
    if degreesFormat:
        lon = np.degrees(lon)
        lat = np.degrees(lat)
        om = np.degrees(om)

        if om < 0:
            om = 180. + om
    
    return lon, lat, om


def rotation_matrix(z_angle, y_angle, x_angle = 0):
    
    #Turn vector to matrices
    rot_vector_z = pole2matrix(0, 90, z_angle)
    rot_vector_y = pole2matrix(90, 0, y_angle)
    rot_vector_x = pole2matrix(0, 0, x_angle)

    # Multiply rotation matrices
    return np.dot(rot_vector_x, np.dot(rot_vector_y, rot_vector_z))


def rotation_quaternion(rot_matrix):
    
    lon, lat, angle = matrix2pole(rot_matrix, degreesFormat=True)
    ax, ay, az = sph2cart(lon, lat, 1.0)
    
    qx = ax * sin(np.radians(angle)/2)
    qy = ay * sin(np.radians(angle)/2)
    qz = az * sin(np.radians(angle)/2)
    qw = cos(np.radians(angle)/2)
    
    return qx, qy, qz, qw

