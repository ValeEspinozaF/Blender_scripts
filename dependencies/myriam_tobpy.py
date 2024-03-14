import pandas as pd
import numpy as np

import spherical_functions
import importlib
importlib.reload(spherical_functions)
from spherical_functions import sph2cart

def myriam_contour_tobpy(filepath, radius):
    
    # Load contour coordinates
    cntrDf = pd.read_csv(filepath, delimiter=' ', header=None, names=['lon', 'lat'])
    
    # Create emtpy lists
    verts = []
    edges = []
    faces = []

    # Split Dataframes where nan rows appear (when multiple contours are stacked)
    cntrDf["cntrNr"] = cntrDf.isnull().all(axis=1).cumsum()

    for n, rows in cntrDf.groupby("cntrNr").groups.items():
        
        verts_n = []
        indCntr = cntrDf.iloc[rows].drop(columns="cntrNr", axis=1).dropna()
        
        # Transform spherical to cartesian coordinates
        x,y,z = sph2cart(indCntr["lon"], indCntr["lat"], radius)
        verts_n = [[xi,yi,zi] for xi,yi,zi in zip(x,y,z)]
        #xt,yt,zt = sph2cart(-41, -16, topoDepth*0.9)
        #verts_n.append([xt,yt,zt])
        verts.append(verts_n)
                
        edges_n = [(i, i+1) for i in range(len(z)-1)]
        edges_n.append((len(z)-1, 0))        
        edges.append(edges_n)
        
        #faces_n = tuple([i for i in range(len(z))])
        #faces_n = (0, 60, int(len(z)))
        #faces.append(faces_n)
        
    return verts, edges#, faces